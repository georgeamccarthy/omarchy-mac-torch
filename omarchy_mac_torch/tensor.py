"""OmtTensor: a torch.Tensor subclass whose data lives in a GPU Storage.

Metadata (shape, strides, offset, dtype) is the wrapper's own, so views are
free: they share the Storage. float32 is stored as f32 words; int64/int32 as
i32 words; bool as u32 0/1. Every op arrives via __torch_dispatch__ (ops.py).
"""

import numpy as np
import torch

from omarchy_mac_torch.runtime import REC, Storage

_NP = {torch.float32: np.float32, torch.int64: np.int32, torch.int32: np.int32, torch.bool: np.uint32}
DEVICE = torch.device("omarchy_mac_torch", 0)


def contiguous_strides(shape):
    out, acc = [], 1
    for s in reversed(shape):
        out.append(acc)
        acc *= max(s, 1)
    return tuple(reversed(out))


class OmtTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, st, shape, stride, offset, dtype, requires_grad=False):
        t = torch.Tensor._make_wrapper_subclass(
            cls, shape, strides=stride, storage_offset=offset, dtype=dtype, device=DEVICE,
            requires_grad=requires_grad)
        t.st = st
        return t

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        from omarchy_mac_torch.ops import dispatch
        return dispatch(func, args, kwargs or {})

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        from omarchy_mac_torch.ops import FUSED
        impl = FUSED.get(func)
        if impl is not None:
            r = impl(*args, **(kwargs or {}))
            if r is not NotImplemented:
                return r
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **(kwargs or {}))

    def __reduce_ex__(self, protocol):
        # torch.save / pickle: an device tensor is stored as its CPU values (and loads as a CPU tensor)
        cpu = to_cpu(self)
        if self.requires_grad:
            cpu.requires_grad_()
        return cpu.__reduce_ex__(protocol)

    def __deepcopy__(self, memo):
        from omarchy_mac_torch.ops import clone
        out = clone(self.detach())
        if out is NotImplemented:  # integer / bool tensors: no GPU copy kernel
            out = from_cpu(to_cpu(self))
        out.requires_grad_(self.requires_grad)
        memo[id(self)] = out
        return out

    def __repr__(self, *a, **k):
        return f"OmtTensor({to_cpu(self)!r})"


def new(shape, dtype=torch.float32):
    shape = tuple(shape)
    n = int(np.prod(shape)) if shape else 1
    return OmtTensor(Storage(n), shape, contiguous_strides(shape), 0, dtype)


def view_of(t, shape, stride, offset=None):
    return OmtTensor(t.st, tuple(shape), tuple(stride), t.storage_offset() if offset is None else offset, t.dtype)


def from_cpu(x):
    """Upload a CPU tensor to a new device tensor."""
    x = x.detach().contiguous()
    arr = x.numpy()
    if x.dtype not in _NP:
        raise TypeError(f"omarchy-mac-torch doesn't support {x.dtype}")
    arr = np.ascontiguousarray(arr.astype(_NP[x.dtype], copy=False))
    t = new(x.shape, x.dtype)
    if arr.size:
        REC.write_words(t.st.buf, arr.reshape(-1))
    return t


def to_cpu(t):
    from omarchy_mac_torch.ops import contig
    c = contig(t)
    n = c.numel()
    if n == 0:
        return torch.empty(c.shape, dtype=c.dtype)
    off = c.storage_offset()
    raw = REC.read_words(c.st.buf, off + n)[off:off + n].clone()
    if c.dtype == torch.float32:
        out = raw
    elif c.dtype == torch.bool:
        out = raw.view(torch.int32) != 0
    else:
        out = raw.view(torch.int32).to(c.dtype)
    return out.reshape(c.shape)
