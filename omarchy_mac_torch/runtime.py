"""GPU memory and command recording for the omarchy-mac-torch backend.

Storage     a pooled GPU buffer of 4-byte words. Freed buffers go back to a
            size-class pool and are reused (the queue is in order, so reuse is
            safe even while earlier recorded work is unsubmitted).
Recorder    ops append compute dispatches to a list; flush() submits them all
            as one compute pass. Reading data back (or uploading) is the only
            thing that waits for the GPU.
"""

import collections
import os

import numpy as np
import torch

from omarchy_mac_torch.conv import gpu

_FLOAT_BITS = np.float32


class _Pool:
    def __init__(self):
        self.free = {}

    def alloc(self, words):
        cls = max(64, 1 << (max(words, 1) - 1).bit_length())
        lst = self.free.get(cls)
        return (lst.pop() if lst else gpu().empty(cls)), cls

    def release(self, buf, cls):
        self.free.setdefault(cls, []).append(buf)


POOL = _Pool()


class Storage:
    __slots__ = ("buf", "cls", "__weakref__")

    def __init__(self, words):
        self.buf, self.cls = POOL.alloc(words)

    def __del__(self):
        try:
            POOL.release(self.buf, self.cls)
        except Exception:
            pass


class _Recorder:
    def __init__(self):
        self.passes = []
        self.groups = {}  # (pipeline, buffer ids) -> bind group
        self.uniforms = {}
        self.dispatches = 0
        # The GPU is shared with the desktop compositor. Keeping the queue shallow (at most `depth`
        # chunks of `chunk` dispatches in flight) bounds how long a frame can wait behind our work,
        # without idling the GPU: the CPU only waits for the *older* chunk to finish.
        self.chunk = int(os.environ.get("OMARCHY_MAC_TORCH_CHUNK", "64"))
        self.depth = int(os.environ.get("OMARCHY_MAC_TORCH_QUEUE_DEPTH", "3"))
        self.pending = collections.deque()

    def uniform(self, *values):
        """A cached uniform buffer holding ints (u32) and floats (f32)."""
        key = tuple(values)
        buf = self.uniforms.get(key)
        if buf is None:
            g = gpu()
            words = np.zeros(max(4, -(-len(values) // 4) * 4), np.uint32)
            for i, v in enumerate(values):
                words[i] = np.float32(v).view(np.uint32) if isinstance(v, float) else v
            buf = self.uniforms[key] = g.device.create_buffer_with_data(
                data=words, usage=g.wgpu.BufferUsage.UNIFORM)
        return buf

    def add(self, pipe, bufs, dims):
        self.passes.append((pipe, bufs, dims))
        if len(self.passes) >= self.chunk:  # start the GPU while Python keeps recording
            self.flush()

    def _encode(self):
        g = gpu()
        enc = g.device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cache = self.groups
        last = None
        for pipe, bufs, dims in self.passes:
            if isinstance(bufs, list):
                key = (id(pipe), *map(id, bufs))
                grp = cache.get(key)
                if grp is None:
                    grp = cache[key] = (g.device.create_bind_group(
                        layout=pipe.get_bind_group_layout(0),
                        entries=[{"binding": i, "resource": {"buffer": b}} for i, b in enumerate(bufs)]),
                        bufs)
                bufs = grp[0]
            if pipe is not last:
                cp.set_pipeline(pipe)
                last = pipe
            cp.set_bind_group(0, bufs)
            cp.dispatch_workgroups(*dims)
        cp.end()
        self.dispatches += len(self.passes)
        self.passes = []
        return enc

    def flush(self):
        if self.passes:
            cmd = self._encode().finish()
            while len(self.pending) >= self.depth:
                self.pending.popleft().sync_wait()  # the oldest chunk only; newer ones keep the GPU busy
            queue = gpu().device.queue
            queue.submit([cmd])
            if self.depth < 1000:
                self.pending.append(queue.on_submitted_work_done_async())

    def read_words(self, buf, words):
        """Wait for everything recorded so far, then copy `words` words of `buf` to the CPU."""
        out = gpu().read(self._encode(), [(buf, words)])[0]
        self.pending.clear()  # reading waited for everything
        return out

    def write_words(self, buf, array):
        self.flush()  # earlier recorded work may still read the old contents
        gpu().device.queue.write_buffer(buf, 0, array)


REC = _Recorder()
