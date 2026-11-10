"""Run a whole conv -> BN -> ReLU -> pool stack on the GPU as one autograd op.

The per-layer GPU conv (conv.py) copies data to the GPU and back around every
call, because BN, ReLU and pooling run on the CPU in between. Here an
nn.Sequential of supported layers runs entirely on the GPU while training:

  forward:  upload the batch and parameters, record every layer into one
            command buffer, read back the output and the BN batch statistics
            (for the running averages) in one packed copy;
  backward: upload the output gradient, record every layer's backward, read
            back all parameter gradients in one packed copy.

Supported layers: Conv2d (see conv.supported, no bias), BatchNorm2d followed
by ReLU (fused), MaxPool2d(2, 2). In eval mode or under no_grad the stack runs
on the CPU as usual (BN then uses its running statistics).
"""

import weakref

import torch
from torch import nn

from omarchy_mac_torch.runtime import REC
from omarchy_mac_torch.conv import (_cdiv, forward_passes, gpu, grad_input_passes,
                            grad_weight_passes, supported)

_Q = "struct Q { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };\n"

_REDUCE_FN = """
var<workgroup> red: array<f32, 256>;
fn wg_sum(t: u32, v: f32) -> f32 {
  red[t] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) { red[t] += red[t + s]; }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}
"""

_EW_INDEX = """
  let i = gid.x + gid.y * nwg.x * 256u;
  let hw = q.h * q.w;
"""

_SOURCES = {
    # Batch mean and (biased) variance per channel: one workgroup per channel.
    "bn_stats": _Q + """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> q: Q;
""" + _REDUCE_FN + """
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let ch = wg.x; let hw = q.h * q.w; let m = f32(q.n * hw);
  let shift = x[ch * hw];  // shifted sums keep the one-pass variance accurate
  var s1 = 0.0; var s2 = 0.0;
  for (var nn = 0u; nn < q.n; nn++) {
    let base = (nn * q.c + ch) * hw;
    for (var e = t; e < hw; e += 256u) { let d = x[base + e] - shift; s1 += d; s2 += d * d; }
  }
  let r1 = wg_sum(t, s1) / m;
  let r2 = wg_sum(t, s2) / m;
  let mean = shift + r1;
  let vr = max(r2 - r1 * r1, 0.0);
  if (t == 0u) { stats[ch] = mean; stats[q.c + ch] = vr; }
}
""",
    # y = relu(gamma * (x - mean) / sqrt(var + eps) + beta)
    "bn_relu": _Q + """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> stats: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
""" + _EW_INDEX + """
  if (i >= q.n * q.c * hw) { return; }
  let ch = (i / hw) % q.c;
  let z = (x[i] - stats[ch]) * inverseSqrt(stats[q.c + ch] + 1e-5) * gamma[ch] + beta[ch];
  y[i] = max(z, 0.0);
}
""",
    # Per channel: dbeta = sum(gz), dgamma = sum(gz * xhat), gz = g masked by ReLU.
    "bn_bwd_reduce": _Q + """
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> dgb: array<f32>;
@group(0) @binding(5) var<uniform> q: Q;
""" + _REDUCE_FN + """
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let ch = wg.x; let hw = q.h * q.w; let m = q.n * hw;
  let mean = stats[ch]; let inv = inverseSqrt(stats[q.c + ch] + 1e-5);
  var s1 = 0.0; var s2 = 0.0;
  for (var nn = 0u; nn < q.n; nn++) {
    let base = (nn * q.c + ch) * hw;
    for (var e = t; e < hw; e += 256u) {
      let i = base + e;
      let gz = select(0.0, g[i], y[i] > 0.0);
      s1 += gz;
      s2 += gz * (x[i] - mean) * inv;
    }
  }
  let r1 = wg_sum(t, s1);
  let r2 = wg_sum(t, s2);
  if (t == 0u) { dgb[ch] = r2; dgb[q.c + ch] = r1; }
}
""",
    # dx = gamma * inv / m * (m * gz - dbeta - xhat * dgamma)
    "bn_bwd_apply": _Q + """
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> y: array<f32>;
@group(0) @binding(4) var<storage, read> gamma: array<f32>;
@group(0) @binding(5) var<storage, read> dgb: array<f32>;
@group(0) @binding(6) var<storage, read_write> dx: array<f32>;
@group(0) @binding(7) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
""" + _EW_INDEX + """
  if (i >= q.n * q.c * hw) { return; }
  let ch = (i / hw) % q.c;
  let m = f32(q.n * hw);
  let inv = inverseSqrt(stats[q.c + ch] + 1e-5);
  let gz = select(0.0, g[i], y[i] > 0.0);
  let xhat = (x[i] - stats[ch]) * inv;
  dx[i] = gamma[ch] * inv / m * (m * gz - dgb[q.c + ch] - xhat * dgb[ch]);
}
""",
    # 2x2 max pool, stride 2 (odd edges dropped, like PyTorch); keeps argmax 0..3.
    "pool_fwd": _Q + """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> idx: array<u32>;
@group(0) @binding(3) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
""" + _EW_INDEX + """
  if (i >= q.n * q.c * q.oh * q.ow) { return; }
  let ox = i % q.ow; let oy = (i / q.ow) % q.oh; let nc = i / (q.oh * q.ow);
  let base = nc * hw + 2u * oy * q.w + 2u * ox;
  var best = x[base]; var bi = 0u;
  let v1 = x[base + 1u]; if (v1 > best) { best = v1; bi = 1u; }
  let v2 = x[base + q.w]; if (v2 > best) { best = v2; bi = 2u; }
  let v3 = x[base + q.w + 1u]; if (v3 > best) { best = v3; bi = 3u; }
  out[i] = best;
  idx[i] = bi;
}
""",
    "pool_bwd": _Q + """
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> gx: array<f32>;
@group(0) @binding(3) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
""" + _EW_INDEX + """
  if (i >= q.n * q.c * hw) { return; }
  let xx = i % q.w; let yy = (i / q.w) % q.h; let nc = i / hw;
  let oy = yy / 2u; let ox = xx / 2u;
  var v = 0.0;
  if (oy < q.oh && ox < q.ow) {
    let o = (nc * q.oh + oy) * q.ow + ox;
    if (idx[o] == (yy % 2u) * 2u + xx % 2u) { v = g[o]; }
  }
  gx[i] = v;
}
""",
    # Depthwise 3x3 conv (groups = channels), stride 1, padding 1.
    "dw_fwd": _Q + """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
""" + _EW_INDEX + """
  if (i >= q.n * q.c * hw) { return; }
  let xx = i32(i % q.w); let yy = i32((i / q.w) % q.h); let nc = i / hw; let ch = nc % q.c;
  var s = 0.0;
  let ks = i32(q.p0); let pad = ks / 2;
  for (var ky = 0; ky < ks; ky++) {
    let sy = yy + ky - pad;
    if (sy < 0 || sy >= i32(q.h)) { continue; }
    for (var kx = 0; kx < ks; kx++) {
      let sx = xx + kx - pad;
      if (sx < 0 || sx >= i32(q.w)) { continue; }
      s += x[nc * hw + u32(sy) * q.w + u32(sx)] * wt[ch * q.p0 * q.p0 + u32(ky * ks + kx)];
    }
  }
  out[i] = s;
}
""",
    "dw_gin": _Q + """
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> gx: array<f32>;
@group(0) @binding(3) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
""" + _EW_INDEX + """
  if (i >= q.n * q.c * hw) { return; }
  let xx = i32(i % q.w); let yy = i32((i / q.w) % q.h); let nc = i / hw; let ch = nc % q.c;
  var s = 0.0;
  let ks = i32(q.p0); let pad = ks / 2;
  for (var ky = 0; ky < ks; ky++) {
    let sy = yy - ky + pad;
    if (sy < 0 || sy >= i32(q.h)) { continue; }
    for (var kx = 0; kx < ks; kx++) {
      let sx = xx - kx + pad;
      if (sx < 0 || sx >= i32(q.w)) { continue; }
      s += g[nc * hw + u32(sy) * q.w + u32(sx)] * wt[ch * q.p0 * q.p0 + u32(ky * ks + kx)];
    }
  }
  gx[i] = s;
}
""",
    # One workgroup per weight (channel, tap): sum over n*h*w.
    "dw_gwt": _Q + """
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> gw: array<f32>;
@group(0) @binding(3) var<uniform> q: Q;
""" + _REDUCE_FN + """
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let kk = q.p0 * q.p0;
  let j = wg.x; let ch = j / kk; let k = j % kk;
  let ky = i32(k / q.p0) - i32(q.p0 / 2u); let kx = i32(k % q.p0) - i32(q.p0 / 2u);
  let hw = q.h * q.w; let m = q.n * hw;
  var s = 0.0;
  for (var e = t; e < m; e += 256u) {
    let n = e / hw; let r = e % hw;
    let yy = i32(r / q.w) + ky; let xx = i32(r % q.w) + kx;
    if (yy >= 0 && yy < i32(q.h) && xx >= 0 && xx < i32(q.w)) {
      let base = (n * q.c + ch) * hw;
      s += g[base + r] * x[base + u32(yy) * q.w + u32(xx)];
    }
  }
  let total = wg_sum(t, s);
  if (t == 0u) { gw[j] = total; }
}
""",
    # Global average pool to [N, C, 1, 1].
    "gap_fwd": _Q + """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
""" + _EW_INDEX + """
  if (i >= q.n * q.c) { return; }
  var s = 0.0;
  for (var r = 0u; r < hw; r++) { s += x[i * hw + r]; }
  out[i] = s / f32(hw);
}
""",
    "gap_bwd": _Q + """
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read_write> gx: array<f32>;
@group(0) @binding(2) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
""" + _EW_INDEX + """
  if (i >= q.n * q.c * hw) { return; }
  gx[i] = g[i / hw] / f32(hw);
}
""",
}

_kernels = {}


# bn_stats that also updates the BatchNorm running averages (one dispatch instead of two).
_SOURCES["bn_stats_run"] = (_SOURCES["bn_stats"]
    .replace("@group(0) @binding(2) var<uniform> q: Q;",
             "@group(0) @binding(2) var<storage, read_write> rm: array<f32>;\n"
             "@group(0) @binding(3) var<storage, read_write> rv: array<f32>;\n"
             "@group(0) @binding(4) var<uniform> q: Q;\n"
             "struct R { mom: f32, corr: f32, p0: u32, p1: u32 };\n"
             "@group(0) @binding(5) var<uniform> r: R;")
    .replace("if (t == 0u) { stats[ch] = mean; stats[q.c + ch] = vr; }",
             "if (t == 0u) {\n"
             "    stats[ch] = mean; stats[q.c + ch] = vr;\n"
             "    rm[ch] = rm[ch] + r.mom * (mean - rm[ch]);\n"
             "    rv[ch] = rv[ch] + r.mom * (vr * r.corr - rv[ch]);\n"
             "  }"))


# bn_relu for eval mode: reads the BatchNorm running mean and variance buffers directly.
_SOURCES["bn_relu_rv"] = (_SOURCES["bn_relu"]
    .replace("@group(0) @binding(1) var<storage, read> stats: array<f32>;",
             "@group(0) @binding(1) var<storage, read> rm: array<f32>;\n"
             "@group(0) @binding(6) var<storage, read> rv: array<f32>;")
    .replace("stats[q.c + ch]", "rv[ch]").replace("stats[ch]", "rm[ch]"))


def _kernel(name):
    if name not in _kernels:
        _kernels[name] = gpu()._pipeline(_SOURCES[name])
    return _kernels[name]


def _ew(total):
    # Grid for one thread per element, 256 per workgroup.
    groups = _cdiv(total, 256)
    gx = min(groups, 65535)
    return (gx, _cdiv(groups, gx), 1)


def _is_pool(m):
    pair = lambda v: v if isinstance(v, tuple) else (v, v)  # noqa: E731
    return (type(m) is nn.MaxPool2d and pair(m.kernel_size) == (2, 2)
            and pair(m.stride) == (2, 2) and pair(m.padding) == (0, 0)
            and pair(m.dilation) == (1, 1) and not m.ceil_mode and not m.return_indices)


def _is_avgpool(m):
    pair = lambda v: v if isinstance(v, tuple) else (v, v)  # noqa: E731
    return (type(m) is nn.AvgPool2d and pair(m.kernel_size) == (2, 2) and pair(m.stride) == (2, 2)
            and pair(m.padding) == (0, 0) and not m.ceil_mode and m.divisor_override is None)


def _is_depthwise(m):
    return (type(m) is nn.Conv2d and m.groups == m.in_channels == m.out_channels > 1
            and m.kernel_size[0] == m.kernel_size[1] and m.kernel_size[0] in (3, 5, 7)
            and m.stride == (1, 1) and m.padding == (m.kernel_size[0] // 2,) * 2
            and m.dilation == (1, 1) and m.padding_mode == "zeros"
            and m.weight.dtype == torch.float32)


def fusable(seq):
    """Whether every layer of nn.Sequential `seq` has a GPU implementation."""
    mods = list(seq)
    if not mods or not isinstance(mods[0], nn.Conv2d):
        return False
    i = 0
    while i < len(mods):
        m = mods[i]
        if isinstance(m, nn.Conv2d):
            if not ((supported(m) or _is_depthwise(m)) and m.bias is None):
                return False
        elif type(m) is nn.AdaptiveAvgPool2d:
            if m.output_size not in (1, (1, 1)):
                return False
        elif type(m) is nn.BatchNorm2d:
            if not (m.affine and m.track_running_stats and m.momentum is not None
                    and m.eps == 1e-5 and i + 1 < len(mods) and type(mods[i + 1]) is nn.ReLU):
                return False
            i += 1  # the ReLU is fused into the BN kernel
        elif not (_is_pool(m) or _is_avgpool(m)):
            return False
        i += 1
    return True


def _cpu(t):
    # An device tensor wraps a CPU tensor (see device.py); the GPU code wants that.
    return getattr(t, "data_", t)


def _params(seq):
    out = []
    for m in seq:
        if isinstance(m, nn.Conv2d):
            out.append(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            out += [m.weight, m.bias]
    return out


class _Plan:
    """Every GPU buffer, bind group and dispatch for one input shape, built once.

    A step only writes the new input and parameters into the plan's buffers,
    encodes the prebuilt passes and reads the results back.
    """

    def __init__(self, seq, shape, train, input_grad, bufs=None, running=None):
        g = gpu()
        n, c, h, w = shape
        self.train, self.input_grad, self.busy = train, input_grad, False
        self.in_shape = tuple(shape)
        params = _params(seq)
        self.x = g.empty(n * c * h * w)
        self.pbufs = bufs if bufs is not None else [g.empty(p.numel()) for p in params]
        self.shapes = [p.shape for p in params]
        self.running = []  # eval: (bn, stats buffer) to fill from running averages
        fwd, recs, self.bns = [], [], []
        cur, mods, i, pi, bi = self.x, list(seq), 0, 0, 0
        while i < len(mods):
            m = mods[i]
            if isinstance(m, nn.Conv2d) and m.groups > 1:  # depthwise k x k
                ks = m.kernel_size[0]
                out, q = g.empty(n * c * h * w), g.params(n, c, h, w, 0, 0, ks)
                fwd.append((_kernel("dw_fwd"), [cur, self.pbufs[pi], out, q], _ew(n * c * h * w)))
                recs.append(("dw", pi, cur, self.pbufs[pi], q, (n, c, h, w), ks))
                cur, pi = out, pi + 1
            elif isinstance(m, nn.Conv2d):
                co, ks = m.out_channels, m.kernel_size[0]
                out = g.empty(n * co * h * w)
                fwd += forward_passes(g, cur, self.pbufs[pi], out, n, c, co, h, w, ks)
                recs.append(("conv", pi, cur, self.pbufs[pi], (n, c, co, h, w, ks)))
                cur, c, pi = out, co, pi + 1
            elif isinstance(m, nn.BatchNorm2d):
                stats = g.empty(2 * c) if (train or running is None) else None
                y, q = g.empty(n * c * h * w), g.params(n, c, h, w)
                if train:
                    self.bns.append((m, stats, c, n * h * w))
                    if running is not None:  # GPU-resident: batch statistics and running averages in one kernel
                        count = n * h * w
                        r = REC.uniform(float(m.momentum), float(count / max(count - 1, 1)), 0, 0)
                        fwd.append((_kernel("bn_stats_run"), [cur, stats, running[bi][0], running[bi][1], q, r],
                                    (c, 1, 1)))
                    else:
                        fwd.append((_kernel("bn_stats"), [cur, stats, q], (c, 1, 1)))
                else:
                    self.running.append((m, stats))
                bi += 1
                if not train and running is not None:  # eval, GPU-resident: running stats read in place
                    fwd.append((_kernel("bn_relu_rv"), [cur, running[bi - 1][0], self.pbufs[pi], self.pbufs[pi + 1],
                                                        y, q, running[bi - 1][1]], _ew(n * c * h * w)))
                else:
                    fwd.append((_kernel("bn_relu"), [cur, stats, self.pbufs[pi], self.pbufs[pi + 1], y, q],
                                _ew(n * c * h * w)))
                recs.append(("bn", pi, cur, stats, self.pbufs[pi], y, q, (n, c, h, w)))
                cur, pi, i = y, pi + 2, i + 1  # skip the fused ReLU
            elif isinstance(m, nn.AdaptiveAvgPool2d):
                out, q = g.empty(n * c), g.params(n, c, h, w)
                fwd.append((_kernel("gap_fwd"), [cur, out, q], _ew(n * c)))
                recs.append(("gap", q, (n, c, h, w)))
                cur, h, w = out, 1, 1
            elif _is_avgpool(m):
                from omarchy_mac_torch.ops import pipe
                oh, ow = h // 2, w // 2
                out, q = g.empty(n * c * oh * ow), g.params(n, c, h, w, oh, ow)
                fwd.append((pipe("avgpool_fwd"), [cur, out, q], _ew(n * c * oh * ow)))
                recs.append(("apool", q, (n, c, h, w)))
                cur, h, w = out, oh, ow
            else:  # 2x2 max pool
                oh, ow = h // 2, w // 2
                out, idx = g.empty(n * c * oh * ow), g.empty(n * c * oh * ow)
                q = g.params(n, c, h, w, oh, ow)
                fwd.append((_kernel("pool_fwd"), [cur, out, idx, q], _ew(n * c * oh * ow)))
                recs.append(("pool", idx, q, (n, c, h, w)))
                cur, h, w = out, oh, ow
            i += 1
        self.out_shape, self.out_buf = (n, c, h, w), cur
        self.fwd = g.bind(fwd)
        self.fwd_reads = [(cur, n * c * h * w)] + [(st, 2 * k) for _, st, k, _ in self.bns]
        if train:
            self._build_backward(g, recs, n * c * h * w)

    def _galloc(self, numel):
        """A (marker, word offset, byte size) range of this plan's gradient storage."""
        off = self._gwords
        self._gwords += -(-max(numel, 1) // 64) * 64  # 256-byte aligned
        return ("G", off, max(numel, 1) * 4)

    def _resolve(self, passes):
        """Replace gradient-range markers by ranges of the current gradient storage."""
        buf = self.gstore.buf
        return [(pipe, [(buf, b[1] * 4, b[2]) if isinstance(b, tuple) and b[0] == "G" else b for b in bufs], dims)
                for pipe, bufs, dims in passes]

    def _build_backward(self, g, recs, out_numel):
        from omarchy_mac_torch.runtime import Storage
        self.gout = g.empty(out_numel)
        self._gwords = 0
        cur, bwd, self.reads = self.gout, [], []
        for r_i, rec in reversed(list(enumerate(recs))):
            kind = rec[0]
            if kind == "pool":
                _, idx, q, (n, c, h, w) = rec
                gx = g.empty(n * c * h * w)
                bwd.append((_kernel("pool_bwd"), [cur, idx, gx, q], _ew(n * c * h * w)))
                cur = gx
            elif kind == "apool":
                from omarchy_mac_torch.ops import pipe
                _, q, (n, c, h, w) = rec
                gx = g.empty(n * c * h * w)
                bwd.append((pipe("avgpool_bwd"), [cur, gx, q], _ew(n * c * h * w)))
                cur = gx
            elif kind == "gap":
                _, q, (n, c, h, w) = rec
                gx = g.empty(n * c * h * w)
                bwd.append((_kernel("gap_bwd"), [cur, gx, q], _ew(n * c * h * w)))
                cur = gx
            elif kind == "bn":
                _, pi, xb, stats, gamma, y, q, (n, c, h, w) = rec
                dgb, gx = self._galloc(2 * c), g.empty(n * c * h * w)
                bwd.append((_kernel("bn_bwd_reduce"), [cur, xb, stats, y, dgb, q], (c, 1, 1)))
                bwd.append((_kernel("bn_bwd_apply"), [cur, xb, stats, y, gamma, dgb, gx, q],
                            _ew(n * c * h * w)))
                self.reads.append(("bn", pi, dgb, 2 * c))
                cur = gx
            elif kind == "dw":
                _, pi, xb, wb, q, (n, c, h, w), ks = rec
                gw = self._galloc(c * ks * ks)
                bwd.append((_kernel("dw_gwt"), [cur, xb, gw, q], (c * ks * ks, 1, 1)))
                self.reads.append(("conv", pi, gw, c * ks * ks))
                if r_i > 0 or self.input_grad:
                    gx = g.empty(n * c * h * w)
                    bwd.append((_kernel("dw_gin"), [cur, wb, gx, q], _ew(n * c * h * w)))
                    cur = gx
                self.in_numel = n * c * h * w
            else:  # conv
                _, pi, xb, wb, (n, ci, co, h, w, ks) = rec
                gw = self._galloc(co * ci * ks * ks)
                bwd += grad_weight_passes(g, xb, cur, gw, n, ci, co, h, w, ks)
                self.reads.append(("conv", pi, gw, co * ci * ks * ks))
                if r_i > 0 or self.input_grad:
                    gx = g.empty(n * ci * h * w)
                    bwd += grad_input_passes(g, cur, wb, gx, n, ci, co, h, w, ks)
                    cur = gx
                self.in_numel = n * ci * h * w
        self._bwd_raw = bwd
        self._gsize = max(self._gwords, 64)
        self.gstore = Storage(self._gsize)
        self.bwd = g.bind(self._resolve(bwd))
        self.gin_buf = cur

    def gpu_forward(self, x):
        """Record the forward pass on device tensor `x`; returns the output device tensor."""
        from omarchy_mac_torch.ops import copy_words
        from omarchy_mac_torch.runtime import REC
        from omarchy_mac_torch.tensor import new
        copy_words(x.st.buf, self.x, x.numel())
        for p in self.fwd:
            REC.add(*p)
        out = new(self.out_shape)
        copy_words(self.out_buf, out.st.buf, out.numel())
        return out

    def gpu_backward(self, gout):
        """Record the backward pass; returns (input grad or None, parameter grads) as device tensors.

        Parameter gradients are views of this plan's persistent gradient storage (no copies). If the
        previous step's gradients are still alive (e.g. accumulating without zero_grad), a fresh
        storage is used so they aren't overwritten."""
        import sys
        from omarchy_mac_torch.ops import copy_words
        from omarchy_mac_torch.runtime import REC, Storage
        from omarchy_mac_torch.tensor import OmtTensor, contiguous_strides, new
        if sys.getrefcount(self.gstore) > 2:
            self.gstore = Storage(self._gsize)
            self.bwd = gpu().bind(self._resolve(self._bwd_raw))
        copy_words(gout.st.buf, self.gout, gout.numel())
        for p in self.bwd:
            REC.add(*p)

        def view(off, shape):
            return OmtTensor(self.gstore, tuple(shape), contiguous_strides(tuple(shape)), off, torch.float32)

        grads = [None] * len(self.shapes)
        for kind, pi, marker, k in self.reads:
            off = marker[1]
            if kind == "bn":
                half = k // 2
                grads[pi], grads[pi + 1] = view(off, (half,)), view(off + half, (half,))
            else:
                grads[pi] = view(off, self.shapes[pi])
        gx = None
        if self.input_grad:
            gx = new(self.in_shape)
            copy_words(self.gin_buf, gx.st.buf, self.in_numel)
        return gx, grads


class _StackFn(torch.autograd.Function):
    # A conv stack on device tensors as ONE autograd op: the plan's passes are
    # recorded straight into the command stream; parameters stay on the GPU.
    @staticmethod
    def forward(ctx, x, seq, *params):
        plan = _gpu_plan(seq, x.shape, True, ctx.needs_input_grad[0], params)
        out = plan.gpu_forward(x)
        plan.busy = True
        ctx.plan = plan
        return out

    @staticmethod
    def backward(ctx, gout):
        from omarchy_mac_torch.ops import contig
        plan = ctx.plan
        gx, grads = plan.gpu_backward(contig(gout))
        plan.busy = False
        return (gx, None, *grads)


def _running_bufs(seq):
    return [(m.running_mean.st.buf, m.running_var.st.buf) for m in seq if isinstance(m, nn.BatchNorm2d)]


_plans = weakref.WeakKeyDictionary()  # module -> {key: _Plan}


def _gpu_plan(seq, shape, train, input_grad, params):
    cache = _plans.setdefault(seq, {})
    key = (tuple(shape), train, input_grad)
    plan = cache.get(key)
    if plan is None or plan.busy:
        plan = _Plan(seq, shape, train, input_grad, bufs=[p.st.buf for p in params],
                     running=_running_bufs(seq))
        if key not in cache:
            cache[key] = plan
    return plan


class FusedSequential(nn.Sequential):
    # Same modules and state_dict as the nn.Sequential it replaces. On device
    # tensors, training (with autograd) and inference (eval mode, no_grad) run
    # the whole stack from prebuilt GPU passes; anything else runs per op.
    def forward(self, x):
        from omarchy_mac_torch.ops import contig, _zero_off
        from omarchy_mac_torch.tensor import OmtTensor
        if isinstance(x, OmtTensor) and x.dtype == torch.float32 and x.dim() == 4:
            x = _zero_off(x)
            if self.training and torch.is_grad_enabled():
                return _StackFn.apply(x, self, *_params(self))
            if not self.training and not torch.is_grad_enabled():
                return _gpu_plan(self, x.shape, False, False, _params(self)).gpu_forward(x)
        return super().forward(x)
