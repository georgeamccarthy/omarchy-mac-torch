"""GPU Conv2d for PyTorch on the M1 under Asahi Linux (Vulkan via wgpu).

Tensors stay on the CPU; each conv call uploads its inputs, runs a tiled
implicit-GEMM compute shader on the GPU and reads the result back. Forward,
grad-input and grad-weight are wired into autograd, so a model trains as
usual. Everything other than the conv layers runs on the CPU.

    import omarchy_mac_torch
    omarchy_mac_torch.accelerate(model)   # swaps eligible nn.Conv2d layers in place

Eligible: float32, groups=1, stride 1, dilation 1, square 1x1 or 3x3 kernels
with "same" padding. Others (e.g. depthwise) stay on the CPU.
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# --- shaders ---------------------------------------------------------------

_HEADER = """
struct P { n: u32, ci: u32, co: u32, h: u32, w: u32, M: u32, N: u32, K: u32,
           kslice: u32, slices: u32, ks: u32, kk: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
const BM = TILE_Mu; const BN = TILE_Nu; const BK = 16u;
var<workgroup> As: array<f32, AS_LEN>;  // [k][m]
var<workgroup> Bs: array<f32, BS_LEN>;  // [k][n]

// im2col of an [N, C, H, W] tensor: row m = (n,y,x), column k = (c,ky,kx).
fn im2col(t_m: u32, k: u32, c: u32) -> f32 {
  let hw = p.h * p.w;
  let x = t_m % p.w; let y = (t_m / p.w) % p.h; let n = t_m / hw;
  let kx = k % p.ks; let ky = (k / p.ks) % p.ks; let ch = k / p.kk;
  let pad = i32(p.ks / 2u);
  let yy = i32(y) + i32(ky) - pad; let xx = i32(x) + i32(kx) - pad;
  if (yy < 0 || yy >= i32(p.h) || xx < 0 || xx >= i32(p.w)) { return 0.0; }
  return SRC[(n * c + ch) * hw + u32(yy) * p.w + u32(xx)];
}
"""

# C[M,N] = sum_k A[m,k] B[k,n]; 256 threads, each a 4x4 block in registers.
_GEMM = """
@compute @workgroup_size(WG_X, WG_Y)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.y * (BN / 4u) + lid.x;
  let m0 = wg.y * BM; let n0 = wg.x * BN;
  let kbeg = wg.z * p.kslice; let kend = min(p.K, kbeg + p.kslice);
  var c0 = vec4<f32>(); var c1 = vec4<f32>(); var c2 = vec4<f32>(); var c3 = vec4<f32>();
  for (var k0 = kbeg; k0 < kend; k0 += BK) {
    for (var e = t; e < BK * BM; e += 256u) {
      A_TILE_INDEX
      let m = m0 + mm; let k = k0 + ka;
      var v = 0.0;
      if (m < p.M && k < kend) { v = LOAD_A; }
      As[ka * BM + mm] = v;
    }
    for (var e = t; e < BK * BN; e += 256u) {
      B_TILE_INDEX
      let n = n0 + nn; let k2 = k0 + kb;
      var u = 0.0;
      if (n < p.N && k2 < kend) { u = LOAD_B; }
      Bs[kb * BN + nn] = u;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < BK; kk++) {
      let ab = kk * BM + lid.y * 4u; let bb = kk * BN + lid.x * 4u;
      let rb = vec4<f32>(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);
      c0 += As[ab] * rb; c1 += As[ab + 1u] * rb; c2 += As[ab + 2u] * rb; c3 += As[ab + 3u] * rb;
    }
    workgroupBarrier();
  }
  let rows = array<vec4<f32>, 4>(c0, c1, c2, c3);
  for (var i = 0u; i < 4u; i++) {
    let row = rows[i];
    for (var j = 0u; j < 4u; j++) {
      let m = m0 + lid.y * 4u + i; let n = n0 + lid.x * 4u + j;
      if (m < p.M && n < p.N) { let v = row[j]; STORE }
    }
  }
}
"""

_M_FAST = "let mm = e % BM; let ka = e / BM;"
_K_FAST_A = "let ka = e % BK; let mm = e / BK;"
_N_FAST = "let nn = e % BN; let kb = e / BN;"
_K_FAST_B = "let kb = e % BK; let nn = e / BK;"
_NCHW_STORE = "let hw = p.h * p.w; out[((m / hw) * p.N + n) * hw + m % hw] = v;"

# forward:     m=(n,y,x)  k=(ci,ky,kx)  n=co           A=im2col(x)   B=w
# grad input:  m=(n,y,x)  k=(co,ky,kx)  n=ci           A=im2col(gy)  B=w flipped
# grad weight: m=co       k=(n,y,x)     n=(ci,ky,kx)   A=gy          B=im2col(x), split-K
_KINDS = {
    "forward": ("a", _M_FAST, "im2col(m, k, p.ci)", _N_FAST, "b[n * p.K + k2]", _NCHW_STORE),
    "grad_input": ("a", _M_FAST, "im2col(m, k, p.co)", _N_FAST,
                   "b[((k2 / p.kk) * p.ci + n) * p.kk + (p.kk - 1u) - k2 % p.kk]", _NCHW_STORE),
    "grad_weight": ("b", _K_FAST_A, "a[((k / (p.h * p.w)) * p.co + m) * p.h * p.w + k % (p.h * p.w)]",
                    _K_FAST_B, "im2col(k2, n, p.ci)", "out[(wg.z * p.M + m) * p.N + n] = v;"),
}

_REDUCE = """
struct P { n: u32, ci: u32, co: u32, h: u32, w: u32, M: u32, N: u32, K: u32,
           kslice: u32, slices: u32, ks: u32, kk: u32 };
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let total = p.M * p.N;
  if (i >= total) { return; }
  var s = 0.0;
  for (var z = 0u; z < p.slices; z++) { s += part[z * total + i]; }
  out[i] = s;
}
"""

_SHAPES = [(64, 64), (256, 16), (16, 256), (128, 32), (32, 128)]


def _cdiv(a, b):
    return -(-a // b)


def _source(kind, bm, bn):
    src_name, a_order, load_a, b_order, load_b, store = _KINDS[kind]
    body = (_GEMM.replace("A_TILE_INDEX", a_order).replace("LOAD_A", load_a)
            .replace("B_TILE_INDEX", b_order).replace("LOAD_B", load_b).replace("STORE", store))
    src = _HEADER.replace("SRC", src_name) + body
    return (src.replace("WG_X", str(bn // 4)).replace("WG_Y", str(bm // 4))
            .replace("AS_LEN", str(bm * 16)).replace("BS_LEN", str(bn * 16))
            .replace("TILE_Mu", f"{bm}u").replace("TILE_Nu", f"{bn}u"))


# --- GPU runtime -------------------------------------------------------------

class _Gpu:
    def __init__(self):
        import wgpu

        self.wgpu = wgpu
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.name = adapter.info["device"]
        self.device = adapter.request_device_sync(required_limits={
            "max-storage-buffer-binding-size": adapter.limits["max-storage-buffer-binding-size"],
            "max-buffer-size": adapter.limits["max-buffer-size"],
        })
        self.pipelines = {}
        self.packs = {}
        self.reduce = self._pipeline(_REDUCE)

    def _pipeline(self, src):
        d = self.device
        return d.create_compute_pipeline(
            layout="auto", compute={"module": d.create_shader_module(code=src), "entry_point": "main"})

    def gemm_pipeline(self, kind, M, N, tile=None):
        # Default: the tile shape that wastes least work on padding. `tuned` picks by timing.
        bm, bn = tile or min(_SHAPES, key=lambda s: _cdiv(M, s[0]) * s[0] * _cdiv(N, s[1]) * s[1])
        key = (kind, bm, bn)
        if key not in self.pipelines:
            self.pipelines[key] = self._pipeline(_source(kind, bm, bn))
        return self.pipelines[key], bm, bn

    def upload(self, t):
        U = self.wgpu.BufferUsage
        arr = t.detach().contiguous().numpy()
        return self.device.create_buffer_with_data(data=arr, usage=U.STORAGE | U.COPY_SRC)

    def empty(self, numel):
        U = self.wgpu.BufferUsage
        return self.device.create_buffer(size=max(4, numel * 4),
                                         usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)

    def write(self, buf, t):
        self.device.queue.write_buffer(buf, 0, t.detach().contiguous().numpy())

    def bind(self, passes):
        # Create bind groups once; encode() accepts the result directly.
        d = self.device
        def entry(i, b):  # a buffer, or a (buffer, byte offset, byte size) range of one
            if isinstance(b, tuple):
                return {"binding": i, "resource": {"buffer": b[0], "offset": b[1], "size": b[2]}}
            return {"binding": i, "resource": {"buffer": b}}

        return [(pipe, d.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            entry(i, b) for i, b in enumerate(buffers)]), dims)
            for pipe, buffers, dims in passes]

    def params(self, *values):
        data = np.zeros(12, np.uint32)
        data[:len(values)] = values
        return self.device.create_buffer_with_data(data=data, usage=self.wgpu.BufferUsage.UNIFORM)

    def encode(self, passes):
        # passes: [(pipeline, [buffers], (gx, gy, gz))], recorded in order. Each
        # compute pass ends with a barrier, so later passes see earlier writes.
        # One compute pass: wgpu tracks buffer usage per dispatch and inserts
        # the barriers between dependent dispatches.
        d = self.device
        enc = d.create_command_encoder()
        cp = enc.begin_compute_pass()
        for pipe, group, dims in passes:
            if isinstance(group, list):
                group = d.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
                    {"binding": i, "resource": {"buffer": b}} for i, b in enumerate(group)])
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, group)
            cp.dispatch_workgroups(*dims)
        cp.end()
        return enc

    def read(self, enc, items):
        # Copy every (buffer, numel) into one staging buffer, submit, wait once,
        # and return a list of 1-D CPU tensors.
        U = self.wgpu.BufferUsage
        total = sum(k for _, k in items)
        pack = self.packs.get(total)
        if pack is None:
            pack = self.packs[total] = self.device.create_buffer(
                size=max(4, total * 4), usage=U.COPY_DST | U.COPY_SRC)
        off = 0
        for b, k in items:
            enc.copy_buffer_to_buffer(b, 0, pack, off * 4, k * 4)
            off += k
        self.device.queue.submit([enc.finish()])
        flat = torch.frombuffer(bytearray(self.device.queue.read_buffer(pack)), dtype=torch.float32)
        return list(flat.split([k for _, k in items]))

    def run(self, passes, out, shape):
        numel = int(np.prod(shape))
        return self.read(self.encode(passes), [(out, numel)])[0].view(shape)


_gpu = None


def gpu():
    global _gpu
    if _gpu is None:
        _gpu = _Gpu()
    return _gpu


def _grid(pipe_bm_bn, M, N, z=1):
    _, bm, bn = pipe_bm_bn
    return (_cdiv(N, bn), _cdiv(M, bm), z)


# --- conv ops ------------------------------------------------------------------

def forward_passes(g, xb, wb, out, n, ci, co, h, wd, ks):
    M, N, K = n * h * wd, co, ci * ks * ks
    pl = g.gemm_pipeline("forward", M, N, _tuned("forward", n, ci, co, h, wd, ks))
    pr = g.params(n, ci, co, h, wd, M, N, K, _cdiv(K, 16) * 16, 1, ks, ks * ks)
    return [(pl[0], [xb, wb, out, pr], _grid(pl, M, N))]


def grad_input_passes(g, gyb, wb, out, n, ci, co, h, wd, ks):
    M, N, K = n * h * wd, ci, co * ks * ks
    pl = g.gemm_pipeline("grad_input", M, N, _tuned("grad_input", n, ci, co, h, wd, ks))
    pr = g.params(n, ci, co, h, wd, M, N, K, _cdiv(K, 16) * 16, 1, ks, ks * ks)
    return [(pl[0], [gyb, wb, out, pr], _grid(pl, M, N))]


def grad_weight_passes(g, xb, gyb, out, n, ci, co, h, wd, ks):
    # Split the n*h*w sum across workgroups, then add the partial sums.
    M, N, K = co, ci * ks * ks, n * h * wd
    pl = g.gemm_pipeline("grad_weight", M, N, _tuned("grad_weight", n, ci, co, h, wd, ks))
    _, bm, bn = pl
    tiles = _cdiv(M, bm) * _cdiv(N, bn)
    slices = max(1, min(512 // tiles, K // 256))
    kslice = _cdiv(_cdiv(K, slices), 16) * 16
    slices = _cdiv(K, kslice)
    part = g.empty(slices * M * N)
    pr = g.params(n, ci, co, h, wd, M, N, K, kslice, slices, ks, ks * ks)
    return [(pl[0], [gyb, xb, part, pr], _grid(pl, M, N, slices)),
            (g.reduce, [part, out, pr], (_cdiv(M * N, 256), 1, 1))]


_TILES_FILE = os.path.expanduser("~/.cache/omarchy_mac_torch/tiles.json")
_tiles = None
_tuning = False


def _tuned(kind, n, ci, co, h, wd, ks):
    if _FORCED is not None:
        return _FORCED
    """The fastest tile shape for this conv GEMM on this GPU, found by timing each
    candidate once and cached on disk (set OMARCHY_MAC_TORCH_TUNE=0 to skip tuning)."""
    global _tiles, _tuning
    if _tuning or os.environ.get("OMARCHY_MAC_TORCH_TUNE", "1") == "0":
        return None
    if _tiles is None:
        try:
            _tiles = json.load(open(_TILES_FILE))
        except (OSError, ValueError):
            _tiles = {}
    key = f"{kind}:{n},{ci},{co},{h},{wd},{ks}"
    if key not in _tiles:
        _tuning = True
        try:
            _tiles[key] = _tune(kind, n, ci, co, h, wd, ks)
        finally:
            _tuning = False
        os.makedirs(os.path.dirname(_TILES_FILE), exist_ok=True)
        json.dump(_tiles, open(_TILES_FILE, "w"))
    return tuple(_tiles[key])


def _tune(kind, n, ci, co, h, wd, ks):
    import time

    g = gpu()
    build = {"forward": forward_passes, "grad_input": grad_input_passes, "grad_weight": grad_weight_passes}[kind]
    x, wt, gy = g.empty(n * ci * h * wd), g.empty(co * ci * ks * ks), g.empty(n * co * h * wd)
    out = g.empty(max(n * ci * h * wd, n * co * h * wd))
    best, best_t = None, 1e9
    for tile in _SHAPES:
        global _FORCED
        _FORCED = tile
        passes = build(g, *{"forward": (x, wt, out), "grad_input": (gy, wt, out),
                            "grad_weight": (x, gy, out)}[kind], n, ci, co, h, wd, ks)
        _FORCED = None
        times = []
        for _ in range(3):
            g.read(g.encode(passes * 6), [(out, 1)])
            t0 = time.perf_counter()
            g.read(g.encode(passes * 6), [(out, 1)])
            times.append(time.perf_counter() - t0)
        if min(times) < best_t:
            best, best_t = tile, min(times)
    return list(best)


_FORCED = None


def conv2d_forward(x, w):
    g = gpu()
    n, ci, h, wd = x.shape
    co, _, ks, _ = w.shape
    out = g.empty(n * co * h * wd)
    return g.run(forward_passes(g, g.upload(x), g.upload(w), out, n, ci, co, h, wd, ks),
                 out, (n, co, h, wd))


def conv2d_grad_input(gy, w):
    g = gpu()
    n, co, h, wd = gy.shape
    _, ci, ks, _ = w.shape
    out = g.empty(n * ci * h * wd)
    return g.run(grad_input_passes(g, g.upload(gy), g.upload(w), out, n, ci, co, h, wd, ks),
                 out, (n, ci, h, wd))


def conv2d_grad_weight(x, gy, ks):
    g = gpu()
    n, ci, h, wd = x.shape
    co = gy.shape[1]
    out = g.empty(co * ci * ks * ks)
    return g.run(grad_weight_passes(g, g.upload(x), g.upload(gy), out, n, ci, co, h, wd, ks),
                 out, (co, ci, ks, ks))


class _Conv2dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return conv2d_forward(x, w)

    @staticmethod
    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        gy = gy.contiguous()
        gx = conv2d_grad_input(gy, w) if ctx.needs_input_grad[0] else None
        gw = conv2d_grad_weight(x, gy, w.shape[-1]) if ctx.needs_input_grad[1] else None
        return gx, gw


def supported(conv):
    ks = conv.kernel_size
    return (type(conv) in (nn.Conv2d, FusedConv2d) and conv.groups == 1
            and conv.stride == (1, 1) and conv.dilation == (1, 1)
            and ks[0] == ks[1] and ks[0] in (1, 3, 5) and conv.padding == (ks[0] // 2,) * 2
            and conv.padding_mode == "zeros" and conv.weight.dtype == torch.float32)


class FusedConv2d(nn.Conv2d):
    # Same parameters and state_dict as nn.Conv2d; the conv itself runs on the GPU.
    def forward(self, x):
        if x.device.type != "cpu" or x.dtype != torch.float32:
            return super().forward(x)
        y = _Conv2dFn.apply(x, self.weight)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


def tune_cpu(threads=2):
    """Keep training from crowding out the desktop: run on 6 of the M1 Pro's 8
    performance cores (leaving 2 and the efficiency cores to the compositor and
    other apps), at lower priority, with `threads` threads. PyTorch's default
    (10 threads on all cores) also lets the efficiency cores hold up the rest."""
    import os

    if os.cpu_count() == 10 and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, range(4, 10))
    try:
        os.nice(5)
    except OSError:
        pass
    torch.set_num_threads(threads)


def set_queue_depth(depth=3, chunk=64):
    """The GPU is shared with the desktop compositor. Training keeps at most `depth` chunks of
    `chunk` dispatches queued, so a frame never waits behind more than a few milliseconds of
    our work. Larger values are slightly faster but can make the desktop lag."""
    from omarchy_mac_torch.runtime import REC

    REC.depth, REC.chunk = depth, chunk


def accelerate(model, fuse=True, convs=False):
    """Move what we can of `model` to the GPU, in place. Returns a summary string.

    With `fuse`, every nn.Sequential whose layers all have GPU kernels (conv,
    BN+ReLU, 2x2 max pool) runs as one GPU op while training (fused.py).
    With `convs`, remaining eligible nn.Conv2d layers run one by one on the
    GPU; off by default, since the copies around each call make it slower
    than the CPU.
    """
    from omarchy_mac_torch.fused import FusedSequential, fusable

    stacks, inside = 0, set()
    if fuse:
        for m in model.modules():
            if type(m) is nn.Sequential and fusable(m):
                m.__class__ = FusedSequential
                stacks += 1
                inside |= {id(c) for c in m.modules()}
    convs = 0
    for m in model.modules():
        if convs and type(m) is nn.Conv2d and supported(m) and id(m) not in inside:
            m.__class__ = FusedConv2d
            convs += 1
    return f"{stacks} fused GPU stacks, {convs} separate GPU convs on {gpu().name}"


def check(seed=0):
    """Compare GPU conv forward and gradients against PyTorch's CPU conv."""
    torch.manual_seed(seed)
    worst = 0.0
    for n, ci, co, h, ks in [(4, 3, 8, 11, 3), (4, 16, 32, 14, 1), (2, 5, 7, 9, 3)]:
        x = torch.randn(n, ci, h, h, requires_grad=True)
        w = torch.randn(co, ci, ks, ks, requires_grad=True)
        gy = torch.randn(n, co, h, h)
        ref = F.conv2d(x, w, padding=ks // 2)
        rgx, rgw = torch.autograd.grad(ref, (x, w), gy)
        got = _Conv2dFn.apply(x, w)
        ggx, ggw = torch.autograd.grad(got, (x, w), gy)
        for r, o in [(ref, got), (rgx, ggx), (rgw, ggw)]:
            worst = max(worst, ((r - o).abs().max() / r.abs().max()).item())
    return worst
