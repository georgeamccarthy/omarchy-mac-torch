"""Op implementations for the device device: WGSL kernels + the aten dispatch table.

Anything not in TABLE falls back to the CPU (download, run, upload), so every
PyTorch program runs; FALLBACKS counts what fell back (each one is a sync).
"""

import collections
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import _pytree as pytree

from omarchy_mac_torch import conv as _conv
from omarchy_mac_torch.conv import _cdiv, forward_passes, gpu, grad_input_passes, grad_weight_passes
from omarchy_mac_torch.fused import _EW_INDEX, _Q, _REDUCE_FN, _ew, _kernel as fused_kernel
from omarchy_mac_torch.runtime import REC
from omarchy_mac_torch.tensor import OmtTensor, contiguous_strides, from_cpu, new, to_cpu, view_of

A = torch.ops.aten
TABLE = {}
FUSED = {}
FALLBACKS = collections.Counter()
_g = gpu()
# conv.py's uniform buffers: share the recorder's cache.
_conv._Gpu.params = lambda self, *v: REC.uniform(*v, *([0] * (12 - len(v))))


def op(*overloads):
    def deco(fn):
        for o in overloads:
            TABLE[o] = fn
        return fn
    return deco


# --- kernels ------------------------------------------------------------------

_SRC = {}
_SRC["ew"] = """
struct E { n: u32, op: u32, bs: u32, alpha: f32, sval: f32, p0: u32, p1: u32, p2: u32,
           shape: vec4<u32>, sa: vec4<u32>, sb: vec4<u32>, oa: u32, ob: u32, oo: u32, p3: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> e: E;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= e.n) { return; }
  var r = i;
  let c3 = r % e.shape.w; r = r / e.shape.w;
  let c2 = r % e.shape.z; r = r / e.shape.z;
  let c1 = r % e.shape.y; let c0 = r / e.shape.y;
  let ia = e.oa + c0 * e.sa.x + c1 * e.sa.y + c2 * e.sa.z + c3 * e.sa.w;
  let ib = e.ob + c0 * e.sb.x + c1 * e.sb.y + c2 * e.sb.z + c3 * e.sb.w;
  let x = a[ia];
  var y = e.sval;
  if (e.bs == 0u) { y = b[ib]; }
  var v = 0.0;
  switch (e.op) {
    case 0u: { v = x + e.alpha * y; }
    case 1u: { v = x - e.alpha * y; }
    case 2u: { v = x * y; }
    case 3u: { v = x / y; }
    case 4u: { v = -x; }
    case 5u: { v = max(x, 0.0); }
    case 6u: { v = exp(x); }
    case 7u: { v = sqrt(x); }
    case 8u: { v = select(0.0, x, y > e.alpha); }
    case 9u: { v = x; }
    case 10u: { v = e.sval; }
    case 11u: { v = e.alpha * x; }
    case 12u: { v = select(x * log(y), 0.0, x == 0.0); }
    case 13u: { v = 1.0 / (1.0 + exp(-x)); }                       // sigmoid
    case 14u: { v = x / (1.0 + exp(-x)); }                         // silu
    case 15u: { v = x * clamp(x + 3.0, 0.0, 6.0) / 6.0; }          // hardswish
    case 16u: { v = x * y * (1.0 - y); }                           // sigmoid_backward(grad=x, output=y)
    case 17u: { let s = 1.0 / (1.0 + exp(-y)); v = x * s * (1.0 + y * (1.0 - s)); }   // silu_backward(grad, self)
    case 18u: { v = select(select(x * (2.0 * y + 3.0) / 6.0, x, y > 3.0), 0.0, y < -3.0); }  // hardswish_backward
    case 19u: { v = clamp(x + 3.0, 0.0, 6.0) / 6.0; }              // hardsigmoid
    case 20u: { v = select(0.0, x / 6.0, y > -3.0 && y < 3.0); }   // hardsigmoid_backward(grad, self)
    case 21u: { v = tanh(x); }
    case 22u: { v = x * (1.0 - y * y); }                           // tanh_backward(grad, output)
    case 23u: { v = clamp(x, e.alpha, e.sval); }
    case 24u: { v = log(x); }
    case 25u: { v = abs(x); }
    case 26u: { v = inverseSqrt(x); }
    case 27u: { v = pow(x, e.alpha); }
    default: { v = x; }
  }
  o[e.oo + i] = v;
}
"""
# integer add of a scalar (i32 words): step counters such as num_batches_tracked.
_SRC["iadd"] = """
@group(0) @binding(0) var<storage, read_write> o: array<i32>;
@group(0) @binding(1) var<uniform> u: vec4<i32>;   // n, unused, out offset, k
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u32(u.x)) { return; }
  o[u32(u.z) + i] = o[u32(u.z) + i] + u.w;
}
"""
_SRC["icopy"] = """
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read_write> o: array<i32>;
@group(0) @binding(2) var<uniform> u: vec4<i32>;   // n, a offset, out offset, unused
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u32(u.x)) { return; }
  o[u32(u.z) + i] = a[u32(u.y) + i];
}
"""
# sum over the middle axis of x viewed as [outer, red, inner]
_SRC["reduce"] = """
struct R { outer: u32, red: u32, inner: u32, xo: u32, oo: u32, scale: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> r: R;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 64u;
  if (i >= r.outer * r.inner) { return; }
  let ou = i / r.inner; let in = i % r.inner;
  var s = 0.0;
  for (var k = 0u; k < r.red; k++) { s += x[r.xo + (ou * r.red + k) * r.inner + in]; }
  o[r.oo + i] = s * r.scale;
}
"""
_SRC["mm"] = """
struct M { m: u32, n: u32, k: u32, hb: u32, ao: u32, bo: u32, co: u32, bio: u32, sa0: u32, sa1: u32, sb0: u32, sb1: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<uniform> u: M;
var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let row = wg.y * 16u + l.y; let col = wg.x * 16u + l.x;
  var acc = 0.0;
  for (var k0 = 0u; k0 < u.k; k0 += 16u) {
    let ak = k0 + l.x; let bk = k0 + l.y;
    var av = 0.0; var bv = 0.0;
    if (row < u.m && ak < u.k) { av = a[u.ao + row * u.sa0 + ak * u.sa1]; }
    if (bk < u.k && col < u.n) { bv = b[u.bo + bk * u.sb0 + col * u.sb1]; }
    As[l.y * 16u + l.x] = av; Bs[l.y * 16u + l.x] = bv;
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) { acc += As[l.y * 16u + kk] * Bs[kk * 16u + l.x]; }
    workgroupBarrier();
  }
  if (row < u.m && col < u.n) {
    var v = acc;
    if (u.hb == 1u) { v += bias[u.bio + col]; }
    c[u.co + row * u.n + col] = v;
  }
}
"""
# BatchNorm without the fused ReLU. stats = [mean(c); biased var(c)] from bn_stats.
_SRC["bn_final"] = """
struct R { c: u32, mom: f32, corr: f32, eps: f32, upd: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> stats: array<f32>;
@group(0) @binding(1) var<storage, read_write> rm: array<f32>;
@group(0) @binding(2) var<storage, read_write> rv: array<f32>;
@group(0) @binding(3) var<storage, read_write> sm: array<f32>;
@group(0) @binding(4) var<storage, read_write> si: array<f32>;
@group(0) @binding(5) var<uniform> r: R;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= r.c) { return; }
  let mean = stats[i]; let v = stats[r.c + i];
  sm[i] = mean; si[i] = inverseSqrt(v + r.eps);
  if (r.upd == 1u) {
    rm[i] = rm[i] + r.mom * (mean - rm[i]);
    rv[i] = rv[i] + r.mom * (v * r.corr - rv[i]);
  }
}
"""
_SRC["bn_eval_inv"] = """
struct R { c: u32, eps: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> rv: array<f32>;
@group(0) @binding(1) var<storage, read_write> si: array<f32>;
@group(0) @binding(2) var<uniform> r: R;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < r.c) { si[gid.x] = inverseSqrt(rv[gid.x] + r.eps); }
}
"""
_SRC["bn_apply"] = _Q + """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> mean: array<f32>;
@group(0) @binding(2) var<storage, read> inv: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
""" + _EW_INDEX + """
  if (i >= q.n * q.c * hw) { return; }
  let ch = (i / hw) % q.c;
  y[i] = (x[i] - mean[ch]) * inv[ch] * gamma[ch] + beta[ch];
}
"""
_SRC["bnb_reduce"] = _Q + """
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> mean: array<f32>;
@group(0) @binding(3) var<storage, read> inv: array<f32>;
@group(0) @binding(4) var<storage, read_write> dgb: array<f32>;
@group(0) @binding(5) var<uniform> q: Q;
""" + _REDUCE_FN + """
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let ch = wg.x; let hw = q.h * q.w; let m = q.n * hw;
  var s1 = 0.0; var s2 = 0.0;
  for (var nn = 0u; nn < q.n; nn++) {
    let base = (nn * q.c + ch) * hw;
    for (var e = t; e < hw; e += 256u) {
      let i = base + e;
      s1 += g[i];
      s2 += g[i] * (x[i] - mean[ch]) * inv[ch];
    }
  }
  let r1 = wg_sum(t, s1);
  let r2 = wg_sum(t, s2);
  if (t == 0u) { dgb[ch] = r2; dgb[q.c + ch] = r1; }
}
"""
_SRC["bnb_apply"] = _Q + """
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> mean: array<f32>;
@group(0) @binding(3) var<storage, read> inv: array<f32>;
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
  let xhat = (x[i] - mean[ch]) * inv[ch];
  dx[i] = gamma[ch] * inv[ch] / m * (m * g[i] - dgb[q.c + ch] - xhat * dgb[ch]);
}
"""
# Row-wise softmax / log-softmax over [n, c], and their backward (mode 0 softmax, 1 log-softmax).
_SRC["smax"] = """
struct S { n: u32, c: u32, mode: u32, p: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> s: S;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= s.n) { return; }
  var mx = -1e30;
  for (var k = 0u; k < s.c; k++) { mx = max(mx, x[r * s.c + k]); }
  var sum = 0.0;
  for (var k = 0u; k < s.c; k++) { sum += exp(x[r * s.c + k] - mx); }
  let lse = log(sum) + mx;
  for (var k = 0u; k < s.c; k++) {
    let lp = x[r * s.c + k] - lse;
    o[r * s.c + k] = select(exp(lp), lp, s.mode == 1u);
  }
}
"""
_SRC["smax_bwd"] = """
struct S { n: u32, c: u32, mode: u32, p: u32 };
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> s: S;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= s.n) { return; }
  var t = 0.0;
  for (var k = 0u; k < s.c; k++) {
    t += select(g[r * s.c + k] * y[r * s.c + k], g[r * s.c + k], s.mode == 1u);
  }
  for (var k = 0u; k < s.c; k++) {
    let i = r * s.c + k;
    o[i] = select(y[i] * (g[i] - t), g[i] - exp(y[i]) * t, s.mode == 1u);
  }
}
"""
# Integer / bool helpers (values are 4-byte words: int64 and int32 as i32, bool as u32 0/1).
_SRC["argmax_row"] = """
struct S { n: u32, c: u32, xo: u32, p: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<i32>;
@group(0) @binding(2) var<uniform> s: S;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= s.n) { return; }
  var best = x[s.xo + r * s.c]; var bi = 0u;
  for (var k = 1u; k < s.c; k++) { let v = x[s.xo + r * s.c + k]; if (v > best) { best = v; bi = k; } }
  o[r] = i32(bi);
}
"""
_SRC["eqw"] = """
struct S { n: u32, ao: u32, bo: u32, oo: u32, bs: u32, bval: i32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read> b: array<i32>;
@group(0) @binding(2) var<storage, read_write> o: array<u32>;
@group(0) @binding(3) var<uniform> s: S;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= s.n) { return; }
  var bv = s.bval;
  if (s.bs == 0u) { bv = b[s.bo + i]; }
  o[s.oo + i] = select(0u, 1u, a[s.ao + i] == bv);
}
"""
_SRC["iaddw"] = """
struct S { n: u32, ao: u32, bo: u32, oo: u32, bs: u32, bval: i32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> a: array<i32>;
@group(0) @binding(1) var<storage, read> b: array<i32>;
@group(0) @binding(2) var<storage, read_write> o: array<i32>;
@group(0) @binding(3) var<uniform> s: S;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= s.n) { return; }
  var bv = s.bval;
  if (s.bs == 0u) { bv = b[s.bo + i]; }
  o[s.oo + i] = a[s.ao + i] + bv;
}
"""
_SRC["isum"] = """
struct S { n: u32, xo: u32, oo: u32, p: u32 };
@group(0) @binding(0) var<storage, read> x: array<i32>;
@group(0) @binding(1) var<storage, read_write> o: array<i32>;
@group(0) @binding(2) var<uniform> s: S;
var<workgroup> part: array<i32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32) {
  var acc = 0;
  for (var i = t; i < s.n; i += 64u) { acc += x[s.xo + i]; }
  part[t] = acc;
  workgroupBarrier();
  if (t == 0u) { var total = 0; for (var k = 0u; k < 64u; k++) { total += part[k]; } o[s.oo] = total; }
}
"""
_SRC["avgpool_fwd"] = _Q + """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= q.n * q.c * q.oh * q.ow) { return; }
  let ox = i % q.ow; let oy = (i / q.ow) % q.oh; let nc = i / (q.ow * q.oh);
  let b = nc * q.h * q.w + (oy * 2u) * q.w + ox * 2u;
  out[i] = 0.25 * (x[b] + x[b + 1u] + x[b + q.w] + x[b + q.w + 1u]);
}
"""
_SRC["avgpool_bwd"] = _Q + """
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read_write> gx: array<f32>;
@group(0) @binding(2) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= q.n * q.c * q.h * q.w) { return; }
  let x = i % q.w; let y = (i / q.w) % q.h; let nc = i / (q.w * q.h);
  if (x / 2u >= q.ow || y / 2u >= q.oh) { gx[i] = 0.0; return; }
  gx[i] = 0.25 * g[nc * q.oh * q.ow + (y / 2u) * q.ow + x / 2u];
}
"""
_SRC["ce"] = """
struct L { n: u32, c: u32, eps: f32, p: f32 };
@group(0) @binding(0) var<storage, read> z: array<f32>;
@group(0) @binding(1) var<storage, read> labels: array<i32>;
@group(0) @binding(2) var<storage, read_write> gout: array<f32>;
@group(0) @binding(3) var<storage, read_write> losses: array<f32>;
@group(0) @binding(4) var<uniform> l: L;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= l.n) { return; }
  var mx = -1e30;
  for (var k = 0u; k < l.c; k++) { mx = max(mx, z[r * l.c + k]); }
  var s = 0.0;
  for (var k = 0u; k < l.c; k++) { s += exp(z[r * l.c + k] - mx); }
  let lse = log(s) + mx;
  let y = u32(labels[r]);
  var loss = 0.0;
  for (var k = 0u; k < l.c; k++) {
    let logp = z[r * l.c + k] - lse;
    let tg = l.eps / f32(l.c) + select(0.0, 1.0 - l.eps, k == y);
    loss -= tg * logp;
    gout[r * l.c + k] = (exp(logp) - tg) / f32(l.n);
  }
  losses[r] = loss;
}
"""
_SRC["adam"] = """
struct H { lr: f32, b1: f32, b2: f32, eps: f32, bc1: f32, sbc2: f32, wd: f32, p: f32 };
struct A { count: u32, po: u32, go: u32, mo: u32, vo: u32, so: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read_write> p: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> a: A;
@group(0) @binding(5) var<uniform> h: H;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= a.count) { return; }
  let gr = g[a.go + i] + h.wd * p[a.po + i];
  let mi = m[a.mo + i] + (1.0 - h.b1) * (gr - m[a.mo + i]);
  let vi = h.b2 * v[a.vo + i] + (1.0 - h.b2) * gr * gr;
  m[a.mo + i] = mi; v[a.vo + i] = vi;
  p[a.po + i] = p[a.po + i] - (h.lr / h.bc1) * mi / (sqrt(vi) / h.sbc2 + h.eps);
}
"""

def _adam4_source():
    """Adam over 4 parameter tensors in one dispatch (WGSL can't index an array of buffers, so the
    per-slot buffer access is generated)."""
    L = ["struct H { lr: f32, b1: f32, b2: f32, eps: f32, bc1: f32, sbc2: f32, wd: f32, p: f32 };",
         "struct A { c: vec4<u32>, po: vec4<u32>, go: vec4<u32>, mo: vec4<u32>, vo: vec4<u32> };"]
    for k in range(4):
        L.append(f"@group(0) @binding({k}) var<storage, read_write> p{k}: array<f32>;")
        L.append(f"@group(0) @binding({4 + k}) var<storage, read> g{k}: array<f32>;")
        L.append(f"@group(0) @binding({8 + k}) var<storage, read_write> m{k}: array<f32>;")
        L.append(f"@group(0) @binding({12 + k}) var<storage, read_write> v{k}: array<f32>;")
    L += ["@group(0) @binding(16) var<uniform> a: A;", "@group(0) @binding(17) var<uniform> h: H;",
          "@compute @workgroup_size(256)",
          "fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {",
          "  let i = gid.x + gid.y * nwg.x * 256u;",
          "  if (i >= a.c.x + a.c.y + a.c.z + a.c.w) { return; }",
          "  var j = i; var s = 0u;",
          "  if (j >= a.c.x) { j -= a.c.x; s = 1u; if (j >= a.c.y) { j -= a.c.y; s = 2u; if (j >= a.c.z) { j -= a.c.z; s = 3u; } } }",
          "  var gr = 0.0; var pv = 0.0; var mv = 0.0; var vv = 0.0;",
          "  switch (s) {"]
    for k in range(4):
        L.append(f"    case {k}u: {{ gr = g{k}[a.go[{k}] + j]; pv = p{k}[a.po[{k}] + j]; mv = m{k}[a.mo[{k}] + j]; vv = v{k}[a.vo[{k}] + j]; }}")
    L += ["    default: { }", "  }",
          "  gr = gr + h.wd * pv;",
          "  let mi = mv + (1.0 - h.b1) * (gr - mv);",
          "  let vi = h.b2 * vv + (1.0 - h.b2) * gr * gr;",
          "  let pn = pv - (h.lr / h.bc1) * mi / (sqrt(vi) / h.sbc2 + h.eps);",
          "  switch (s) {"]
    for k in range(4):
        L.append(f"    case {k}u: {{ p{k}[a.po[{k}] + j] = pn; m{k}[a.mo[{k}] + j] = mi; v{k}[a.vo[{k}] + j] = vi; }}")
    L += ["    default: { }", "  }", "}"]
    return "\n".join(L)


_SRC["adam4"] = _adam4_source()

_pipes = {}
_DUMMY = _g.empty(64)


def pipe(name):
    p = _pipes.get(name)
    if p is None:
        p = _pipes[name] = _g._pipeline(_SRC[name])
    return p


# --- helpers ------------------------------------------------------------------

def _is_dev(x):
    return isinstance(x, OmtTensor)


def _pad4(v):
    return [1] * (4 - len(v)) + list(v)


def _bstrides(t, shape):
    """Strides of `t` broadcast to `shape`, padded to rank 4."""
    shape = list(shape)
    st = [0] * (len(shape) - t.dim()) + [s if sz > 1 else 0 for s, sz in zip(t.stride(), t.shape)]
    st = [s if sz > 1 else 0 for s, sz in zip(st, shape)]
    return [0] * (4 - len(shape)) + st


def _grid64(total):
    groups = _cdiv(total, 64)
    gx = min(groups, 65535)
    return (gx, _cdiv(groups, gx), 1)


def copy_pass(src, dst, n, src_off=0, dst_off=0):
    """A dispatch that copies n 4-byte words between GPU buffers."""
    return (pipe("icopy"), [src, dst, REC.uniform(n, src_off, dst_off, 0)], (_cdiv(n, 64), 1, 1))


def copy_words(src, dst, n, src_off=0, dst_off=0):
    REC.add(*copy_pass(src, dst, n, src_off, dst_off))


def _emit(name, bufs, dims):
    REC.add(pipe(name), bufs, dims)


def _launch_ew(opc, a, b, out, shape, alpha=1.0, sval=0.0, oo=0):
    n = int(np.prod(shape)) if len(shape) else 1
    if n == 0:
        return
    sh = _pad4(shape)
    sb = _bstrides(b, shape) if b is not None else [0, 0, 0, 0]
    if opc == 10:  # fill: reads nothing (a scratch buffer keeps in/out bindings distinct)
        sa, abuf, aoff = [0, 0, 0, 0], _DUMMY, 0
    else:
        sa, abuf, aoff = _bstrides(a, shape), a.st.buf, a.storage_offset()
    u = REC.uniform(n, opc, 1 if b is None else 0, float(alpha), float(sval), 0, 0, 0,
                    *sh, *sa, *sb, aoff, b.storage_offset() if b is not None else 0, oo, 0)
    _emit("ew", [abuf, abuf if b is None else b.st.buf, out.st.buf, u], _ew(n))


def _f32(*ts):
    return all(_is_dev(t) and t.dtype == torch.float32 for t in ts) and all(t.dim() <= 4 for t in ts)


def contig(t):
    if t.is_contiguous():
        return t
    out = new(t.shape, t.dtype)
    if t.dtype == torch.float32 and t.dim() <= 4:
        _launch_ew(9, t, None, out, t.shape)
        return out
    return from_cpu(to_cpu_raw(t).contiguous())


def to_cpu_raw(t):
    # Fallback for non-contiguous / high-rank tensors: gather via the CPU as a flat strided view.
    flat = REC.read_words(t.st.buf, t.st.cls)
    raw = flat if t.dtype == torch.float32 else flat.view(torch.int32).to(t.dtype)
    return raw.as_strided(t.shape, t.stride(), t.storage_offset())


def _operand(x):
    """(device tensor or None, python scalar or None)"""
    if _is_dev(x):
        return x, None
    if isinstance(x, torch.Tensor) and x.dim() == 0:
        return None, float(x.item())
    return None, float(x)


def _int_add(a, b):
    """Integer tensor + integer tensor / python int (same shape), e.g. accumulating counts on the GPU."""
    if not a.is_contiguous():
        return NotImplemented
    if _is_dev(b):
        if b.dtype not in (torch.int64, torch.int32) or b.shape != a.shape or not b.is_contiguous():
            return NotImplemented
        bo, bs, bval, bbuf = b.storage_offset(), 0, 0, b.st.buf
    elif isinstance(b, (int, bool)) or (isinstance(b, torch.Tensor) and b.dim() == 0 and not b.is_floating_point()):
        bo, bs, bval, bbuf = 0, 1, int(b) & 0x7FFFFFFF if int(b) >= 0 else int(b), a.st.buf
    else:
        return NotImplemented
    n = a.numel()
    out = new(a.shape, torch.int64)
    u = REC.uniform(n, a.storage_offset(), bo, 0, bs, bval & 0xFFFFFFFF, 0, 0)
    _emit("iaddw", [a.st.buf, bbuf, out.st.buf, u], (_cdiv(n, 64), 1, 1))
    return out


def binary(opc, a, b, alpha=1.0):
    if opc == 0 and alpha == 1 and _is_dev(a) and a.dtype in (torch.int64, torch.int32):
        return _int_add(a, b)
    bt, bs = _operand(b)
    if not _f32(a) or (bt is not None and not _f32(bt)):
        return NotImplemented
    shape = torch.broadcast_shapes(a.shape, bt.shape if bt is not None else ())
    out = new(shape)
    _launch_ew(opc, a, bt, out, shape, alpha, bs or 0.0)
    return out


def unary(opc, a, alpha=1.0):
    if not _f32(a):
        return NotImplemented
    out = new(a.shape)
    _launch_ew(opc, a, None, out, a.shape, alpha)
    return out


def inplace(fn):
    def run(self, *args, **kw):
        res = fn(self, *args, **kw)
        if res is NotImplemented or not self.is_contiguous():
            return NotImplemented
        _launch_ew(9, res, None, self, self.shape, oo=self.storage_offset())
        return self
    return run


def _cpu_scalar_tensor_ok(x):
    return isinstance(x, torch.Tensor) and not _is_dev(x)


# --- elementwise ----------------------------------------------------------------

op(A.add.Tensor)(lambda a, b, alpha=1: binary(0, a, b, alpha))
op(A.add.Scalar)(lambda a, b, alpha=1: binary(0, a, b, alpha))
op(A.sub.Tensor)(lambda a, b, alpha=1: binary(1, a, b, alpha))
op(A.sub.Scalar)(lambda a, b, alpha=1: binary(1, a, b, alpha))
op(A.mul.Tensor, A.mul.Scalar)(lambda a, b: binary(2, a, b))
op(A.div.Tensor, A.div.Scalar)(lambda a, b: binary(3, a, b))
op(A.neg.default)(lambda a: unary(4, a))
op(A.relu.default)(lambda a: unary(5, a))
op(A.exp.default)(lambda a: unary(6, a))
op(A.sqrt.default)(lambda a: unary(7, a))
op(A.sigmoid.default)(lambda a: unary(13, a))
op(A.silu.default)(lambda a: unary(14, a))
op(A.hardswish.default)(lambda a: unary(15, a))
op(A.hardsigmoid.default)(lambda a: unary(19, a))
op(A.tanh.default)(lambda a: unary(21, a))
op(A.add_.Tensor)(inplace(lambda a, b, alpha=1: binary(0, a, b, alpha)))
op(A.add_.Scalar)(inplace(lambda a, b, alpha=1: binary(0, a, b, alpha)))
op(A.mul_.Tensor, A.mul_.Scalar)(inplace(lambda a, b: binary(2, a, b)))
op(A.sub_.Tensor)(inplace(lambda a, b, alpha=1: binary(1, a, b, alpha)))
op(A.div_.Tensor, A.div_.Scalar)(inplace(lambda a, b: binary(3, a, b)))


def _backward_ew(opc):
    def run(grad, y, *rest):
        if not _f32(grad, y):
            return NotImplemented
        out = new(grad.shape)
        _launch_ew(opc, grad, y, out, grad.shape)
        return out
    return run


op(A.sigmoid_backward.default)(_backward_ew(16))
op(A.silu_backward.default)(_backward_ew(17))
op(A.hardswish_backward.default)(_backward_ew(18))
op(A.hardsigmoid_backward.default)(_backward_ew(20))
op(A.tanh_backward.default)(_backward_ew(22))


@op(A.threshold_backward.default)
def threshold_backward(grad, x, threshold):
    if not _f32(grad, x):
        return NotImplemented
    out = new(grad.shape)
    _launch_ew(8, grad, x, out, grad.shape, alpha=float(threshold))
    return out


@op(A.add_.Tensor)
def _int_add_(self, other, alpha=1):
    # integer counters (num_batches_tracked); float tensors use the elementwise kernel
    if self.dtype == torch.float32:
        return inplace(lambda a, b, alpha=1: binary(0, a, b, alpha))(self, other, alpha=alpha)
    if self.dtype in (torch.int64, torch.int32) and not isinstance(other, OmtTensor) and self.is_contiguous():
        k = int(other.item() if isinstance(other, torch.Tensor) else other) * int(alpha)
        n = self.numel()
        off = self.storage_offset()
        u = REC.uniform(n, 0, off, k)
        _emit("iadd", [self.st.buf, u], (_cdiv(n, 64), 1, 1))
        return self
    return NotImplemented


@op(A.zero_.default)
def zero_(self):
    if self.dtype != torch.float32 or not self.is_contiguous():
        return NotImplemented
    _launch_ew(10, self, None, self, self.shape, sval=0.0, oo=self.storage_offset())
    return self


@op(A.fill_.Scalar)
def fill_(self, v):
    if self.dtype != torch.float32 or not self.is_contiguous():
        return NotImplemented
    _launch_ew(10, self, None, self, self.shape, sval=float(v), oo=self.storage_offset())
    return self


@op(A.clone.default)
def clone(self, memory_format=None):
    if not _f32(self):
        return NotImplemented
    out = new(self.shape)
    _launch_ew(9, self, None, out, self.shape)
    return out


@op(A.ones_like.default)
def ones_like(self, **kw):
    out = new(self.shape, kw.get("dtype") or self.dtype)
    if out.dtype != torch.float32:
        return NotImplemented
    _launch_ew(10, out, None, out, out.shape, sval=1.0)
    return out


@op(A.empty_like.default)
def empty_like(self, **kw):
    return new(self.shape, kw.get("dtype") or self.dtype)


@op(A.zeros_like.default)
def zeros_like(self, **kw):
    out = new(self.shape, kw.get("dtype") or self.dtype)
    if out.dtype != torch.float32:
        return NotImplemented
    _launch_ew(10, out, None, out, out.shape, sval=0.0)
    return out


@op(A.copy_.default)
def copy_(dst, src, non_blocking=False):
    if not _is_dev(dst):  # device -> cpu
        dst.copy_(to_cpu(src))
        return dst
    if not _is_dev(src):  # cpu -> device
        s = from_cpu(src.to(dst.dtype) if src.dtype != dst.dtype else src)
        src = s.expand(dst.shape) if s.shape != dst.shape else s
    if dst.dtype != torch.float32 or src.dtype != torch.float32:
        if dst.dtype == src.dtype and dst.is_contiguous() and src.is_contiguous():
            n = dst.numel()  # same-size words: use the integer add of 0 as a copy
            u = REC.uniform(n, src.storage_offset(), dst.storage_offset(), 0)
            _emit("icopy", [src.st.buf, dst.st.buf, u], (_cdiv(n, 64), 1, 1))
            return dst
        return NotImplemented
    if not dst.is_contiguous() or dst.dim() > 4 or src.dim() > 4:
        return NotImplemented
    _launch_ew(9, src, None, dst, dst.shape, oo=dst.storage_offset())
    return dst


@op(A._to_copy.default)
def _to_copy(self, dtype=None, layout=None, device=None, pin_memory=None, non_blocking=False,
             memory_format=None):
    if not _is_dev(self):
        return NotImplemented
    if device is not None and torch.device(device).type == "cpu":
        out = to_cpu(self)
        return out.to(dtype) if dtype is not None and dtype != out.dtype else out
    dt = dtype or self.dtype
    if dt == self.dtype:
        return clone(self) if self.dtype == torch.float32 else from_cpu(to_cpu(self))
    return from_cpu(to_cpu(self).to(dt))


@op(A._local_scalar_dense.default)
def _local_scalar_dense(self):
    return to_cpu(self).item()


# --- views (metadata only) -------------------------------------------------------

def _infer(shape, numel):
    shape = list(shape)
    if -1 in shape:
        known = math.prod(s for s in shape if s != -1)
        shape[shape.index(-1)] = numel // max(known, 1)
    return shape


@op(A.view.default, A._unsafe_view.default, A.reshape.default)
def view(self, shape):
    shape = _infer(shape, self.numel())
    src = self if self.is_contiguous() else contig(self)
    return view_of(src, shape, contiguous_strides(shape), src.storage_offset())


op(A.detach.default, A.alias.default)(lambda self: view_of(self, self.shape, self.stride()))
op(A.t.default)(lambda self: view_of(self, self.shape[::-1], self.stride()[::-1]) if self.dim() == 2 else self)


@op(A.transpose.int)
def transpose(self, d0, d1):
    sh, st = list(self.shape), list(self.stride())
    sh[d0], sh[d1], st[d0], st[d1] = sh[d1], sh[d0], st[d1], st[d0]
    return view_of(self, sh, st)


@op(A.permute.default)
def permute(self, dims):
    return view_of(self, [self.shape[d] for d in dims], [self.stride()[d] for d in dims])


@op(A.unsqueeze.default)
def unsqueeze(self, dim):
    dim = dim % (self.dim() + 1)
    sh, st = list(self.shape), list(self.stride())
    sh.insert(dim, 1)
    st.insert(dim, sh[dim + 1] * st[dim] if dim < self.dim() else 1)
    return view_of(self, sh, st)


@op(A.squeeze.dim)
def squeeze(self, dim):
    dim = dim % max(self.dim(), 1)
    if self.dim() == 0 or self.shape[dim] != 1:
        return view_of(self, self.shape, self.stride())
    return view_of(self, self.shape[:dim] + self.shape[dim + 1:], self.stride()[:dim] + self.stride()[dim + 1:])


@op(A.expand.default)
def expand(self, size, implicit=False):
    size = list(size)
    lead = len(size) - self.dim()
    sh, st = [], []
    for i, s in enumerate(size):
        if i < lead:
            sh.append(s); st.append(0)
        else:
            d = i - lead
            s = self.shape[d] if s == -1 else s
            sh.append(s); st.append(0 if self.shape[d] == 1 and s != 1 else self.stride()[d])
    return view_of(self, sh, st)


@op(A.select.int)
def select(self, dim, index):
    index = index % self.shape[dim]
    return view_of(self, self.shape[:dim] + self.shape[dim + 1:], self.stride()[:dim] + self.stride()[dim + 1:],
                   self.storage_offset() + index * self.stride()[dim])


@op(A.slice.Tensor)
def slice_(self, dim=0, start=None, end=None, step=1):
    n = self.shape[dim]
    start = 0 if start is None else (start + n if start < 0 else start)
    end = n if end is None or end > n else (end + n if end < 0 else end)
    start, end = min(start, n), max(start, min(end, n))
    sh, st = list(self.shape), list(self.stride())
    sh[dim] = -(-(end - start) // step)
    st[dim] *= step
    return view_of(self, sh, st, self.storage_offset() + start * self.stride()[dim])


@op(A.as_strided.default)
def as_strided(self, size, stride, offset=None):
    return view_of(self, size, stride, self.storage_offset() if offset is None else offset)


# --- reductions --------------------------------------------------------------------

def _reduce(x, dims, keepdim, mean):
    if not _f32(x):
        return NotImplemented
    nd = x.dim()
    dims = sorted({d % max(nd, 1) for d in dims}) if dims else list(range(nd))
    if dims != list(range(dims[0], dims[-1] + 1)) if dims else False:
        return NotImplemented
    xc = contig(x)
    if not dims:
        return clone(x)
    outer = math.prod(x.shape[:dims[0]])
    red = math.prod(x.shape[dims[0]:dims[-1] + 1])
    inner = math.prod(x.shape[dims[-1] + 1:])
    oshape = [1 if i in dims else s for i, s in enumerate(x.shape)] if keepdim else \
        [s for i, s in enumerate(x.shape) if i not in dims]
    out = new(oshape)
    n = outer * inner
    u = REC.uniform(outer, red, inner, xc.storage_offset(), 0, float(1.0 / red) if mean else 1.0, 0, 0)
    _emit("reduce", [xc.st.buf, out.st.buf, u], _grid64(n))
    return out


@op(A.sum.dim_IntList)
def sum_dim(x, dim, keepdim=False, dtype=None):
    return _reduce(x, dim, keepdim, False)


@op(A.mean.dim)
def mean_dim(x, dim, keepdim=False, dtype=None):
    return _reduce(x, dim, keepdim, True)


@op(A.mean.default)
def mean_all(x, dtype=None):
    return _reduce(x, None, False, True)


# --- matmul ---------------------------------------------------------------------------

def _mm(a, b, bias=None):
    if not _f32(a, b) or a.dim() != 2 or b.dim() != 2:
        return NotImplemented
    m, k = a.shape
    n = b.shape[1]
    out = new((m, n))
    bt = bias if bias is not None else a
    u = REC.uniform(m, n, k, int(bias is not None), a.storage_offset(), b.storage_offset(), 0,
                    bt.storage_offset(), *a.stride(), *b.stride())
    _emit("mm", [a.st.buf, b.st.buf, bt.st.buf, out.st.buf, u], (_cdiv(n, 16), _cdiv(m, 16), 1))
    return out


op(A.mm.default)(lambda a, b: _mm(a, b))


@op(A.addmm.default)
def addmm(bias, a, b, beta=1, alpha=1):
    if beta != 1 or alpha != 1 or bias.dim() != 1 or not _f32(bias) or not bias.is_contiguous():
        return NotImplemented
    return _mm(a, b, bias)


# --- convolution ------------------------------------------------------------------------

_PASSES = {}


def _passes(key, build):
    p = _PASSES.get(key)
    if p is None:
        p = _PASSES[key] = build()
    return p


def _zero_off(t):
    t = contig(t)
    if t.storage_offset() != 0:
        out = new(t.shape)
        _launch_ew(9, t, None, out, t.shape)
        t = out
    return t


def _conv_kind(x, w, bias, stride, padding, dilation, transposed, groups):
    if bias is not None or transposed or not _f32(x, w) or x.dim() != 4 or w.dim() != 4:
        return None
    if tuple(stride) != (1, 1) or tuple(dilation) != (1, 1) or w.shape[2] != w.shape[3]:
        return None
    ks = w.shape[2]
    if tuple(padding) != (ks // 2, ks // 2):
        return None
    if groups == 1 and ks in (1, 3, 5):
        return "dense"
    if groups == x.shape[1] == w.shape[0] and ks in (3, 5, 7) and groups > 1:
        return "dw"
    return None


@op(A.convolution.default)
def convolution(x, w, bias, stride, padding, dilation, transposed, output_padding, groups):
    kind = _conv_kind(x, w, bias, stride, padding, dilation, transposed, groups)
    if kind is None:
        return NotImplemented
    x, w = _zero_off(x), _zero_off(w)
    n, ci, h, wd = x.shape
    co, _, ks, _ = w.shape
    out = new((n, co, h, wd))
    if kind == "dense":
        ps = _passes(("cf", id(x.st.buf), id(w.st.buf), id(out.st.buf), tuple(x.shape), co, ks),
                     lambda: forward_passes(_g, x.st.buf, w.st.buf, out.st.buf, n, ci, co, h, wd, ks))
        for p in ps:
            REC.add(*p)
    else:
        q = REC.uniform(n, ci, h, wd, 0, 0, ks, 0)
        REC.add(fused_kernel("dw_fwd"), [x.st.buf, w.st.buf, out.st.buf, q], _ew(n * ci * h * wd))
    return out


@op(A.convolution_backward.default)
def convolution_backward(gy, x, w, bias_sizes, stride, padding, dilation, transposed, output_padding,
                         groups, output_mask):
    kind = _conv_kind(x, w, None, stride, padding, dilation, transposed, groups)
    if kind is None or not _f32(gy) or output_mask[2]:
        return NotImplemented
    gy, x, w = _zero_off(gy), _zero_off(x), _zero_off(w)
    n, ci, h, wd = x.shape
    co, _, ks, _ = w.shape
    gx = gw = None
    if kind == "dense":
        if output_mask[0]:
            gx = new(x.shape)
            for p in _passes(("cgi", id(gy.st.buf), id(w.st.buf), id(gx.st.buf), tuple(x.shape), co, ks),
                             lambda: grad_input_passes(_g, gy.st.buf, w.st.buf, gx.st.buf, n, ci, co, h, wd, ks)):
                REC.add(*p)
        if output_mask[1]:
            gw = new(w.shape)
            for p in _passes(("cgw", id(x.st.buf), id(gy.st.buf), id(gw.st.buf), tuple(x.shape), co, ks),
                             lambda: grad_weight_passes(_g, x.st.buf, gy.st.buf, gw.st.buf, n, ci, co, h, wd, ks)):
                REC.add(*p)
    else:
        q = REC.uniform(n, ci, h, wd, 0, 0, ks, 0)
        if output_mask[0]:
            gx = new(x.shape)
            REC.add(fused_kernel("dw_gin"), [gy.st.buf, w.st.buf, gx.st.buf, q], _ew(n * ci * h * wd))
        if output_mask[1]:
            gw = new(w.shape)
            REC.add(fused_kernel("dw_gwt"), [gy.st.buf, x.st.buf, gw.st.buf, q], (ci * ks * ks, 1, 1))
    return gx, gw, None


# --- batch norm ----------------------------------------------------------------------------

@op(A.native_batch_norm.default)
def native_batch_norm(x, weight, bias, rm, rv, training, momentum, eps):
    if not _f32(x, weight, bias) or x.dim() != 4 or rm is None or not _f32(rm, rv):
        return NotImplemented
    x = _zero_off(x)
    n, c, h, w = x.shape
    q = REC.uniform(n, c, h, w, 0, 0, 0, 0)
    y = new(x.shape)
    if training:
        stats, sm, si = new((2 * c,)), new((c,)), new((c,))
        count = n * h * w
        _emit_ext(fused_kernel("bn_stats"), [x.st.buf, stats.st.buf, q], (c, 1, 1))
        r = REC.uniform(c, float(momentum), float(count / max(count - 1, 1)), float(eps), 1, 0, 0, 0)
        _emit("bn_final", [stats.st.buf, rm.st.buf, rv.st.buf, sm.st.buf, si.st.buf, r], (_cdiv(c, 64), 1, 1))
    else:
        sm, si = rm, new((c,))
        r = REC.uniform(c, float(eps), 0, 0)
        _emit("bn_eval_inv", [rv.st.buf, si.st.buf, r], (_cdiv(c, 64), 1, 1))
    _emit("bn_apply", [x.st.buf, sm.st.buf, si.st.buf, weight.st.buf, bias.st.buf, y.st.buf, q], _ew(n * c * h * w))
    if not training:
        return y, new((0,)), new((0,))
    return y, sm, si


def _emit_ext(pipeline, bufs, dims):
    REC.add(pipeline, bufs, dims)


@op(A.native_batch_norm_backward.default)
def native_batch_norm_backward(gy, x, weight, rm, rv, sm, si, train, eps, output_mask):
    if not train or not _f32(gy, x, weight, sm, si):
        return NotImplemented
    gy, x = _zero_off(gy), _zero_off(x)
    n, c, h, w = x.shape
    q = REC.uniform(n, c, h, w, 0, 0, 0, 0)
    dgb = new((2, c))
    _emit("bnb_reduce", [gy.st.buf, x.st.buf, sm.st.buf, si.st.buf, dgb.st.buf, q], (c, 1, 1))
    gx = None
    if output_mask[0]:
        gx = new(x.shape)
        _emit("bnb_apply", [gy.st.buf, x.st.buf, sm.st.buf, si.st.buf, weight.st.buf, dgb.st.buf, gx.st.buf, q],
              _ew(n * c * h * w))
    return gx, select(dgb, 0, 0), select(dgb, 0, 1)


# --- pooling ----------------------------------------------------------------------------------

def _pool_ok(x, k, s, p, d, ceil):
    return (_f32(x) and x.dim() == 4 and tuple(k) == (2, 2) and tuple(s or k) == (2, 2)
            and tuple(p) == (0, 0) and tuple(d) == (1, 1) and not ceil)


@op(A.max_pool2d_with_indices.default)
def max_pool2d_with_indices(x, k, s=(), p=(0, 0), d=(1, 1), ceil=False):
    if not _pool_ok(x, k, s, p, d, ceil):
        return NotImplemented
    x = _zero_off(x)
    n, c, h, w = x.shape
    oh, ow = h // 2, w // 2
    out, idx = new((n, c, oh, ow)), new((n, c, oh, ow), torch.int64)
    q = REC.uniform(n, c, h, w, oh, ow, 0, 0)
    REC.add(fused_kernel("pool_fwd"), [x.st.buf, out.st.buf, idx.st.buf, q], _ew(n * c * oh * ow))
    return out, idx


@op(A.max_pool2d_with_indices_backward.default)
def max_pool2d_backward(gy, x, k, s, p, d, ceil, idx):
    if not _pool_ok(x, k, s, p, d, ceil):
        return NotImplemented
    gy = _zero_off(gy)
    n, c, h, w = x.shape
    gx = new(x.shape)
    q = REC.uniform(n, c, h, w, h // 2, w // 2, 0, 0)
    REC.add(fused_kernel("pool_bwd"), [gy.st.buf, idx.st.buf, gx.st.buf, q], _ew(n * c * h * w))
    return gx


def _softmax_ok(x, dim):
    return _f32(x) and x.dim() == 2 and dim % 2 == 1


def _softmax(x, dim, half_to_float=False, mode=0):
    if not _softmax_ok(x, dim):
        return NotImplemented
    x = _zero_off(x)
    n, c = x.shape
    out = new((n, c))
    _emit("smax", [x.st.buf, out.st.buf, REC.uniform(n, c, mode, 0)], (_cdiv(n, 64), 1, 1))
    return out


op(A._softmax.default)(lambda x, dim, half=False: _softmax(x, dim, half, 0))
op(A._log_softmax.default)(lambda x, dim, half=False: _softmax(x, dim, half, 1))


def _softmax_bwd(mode):
    def run(g, y, dim, dtype):
        if not _softmax_ok(g, dim) or not _f32(y):
            return NotImplemented
        g, y = _zero_off(g), _zero_off(y)
        n, c = g.shape
        out = new((n, c))
        _emit("smax_bwd", [g.st.buf, y.st.buf, out.st.buf, REC.uniform(n, c, mode, 0)], (_cdiv(n, 64), 1, 1))
        return out
    return run


op(A._softmax_backward_data.default)(_softmax_bwd(0))
op(A._log_softmax_backward_data.default)(_softmax_bwd(1))
op(A.xlogy.Tensor)(lambda a, b: binary(12, a, b))


# --- eval helpers: argmax / eq / integer sum stay on the GPU ---------------------------------------

@op(A.argmax.default)
def argmax(x, dim=None, keepdim=False):
    if not _f32(x) or x.dim() != 2 or dim is None or dim % 2 != 1 or keepdim:
        return NotImplemented
    x = contig(x)
    n, c = x.shape
    out = new((n,), torch.int64)
    _emit("argmax_row", [x.st.buf, out.st.buf, REC.uniform(n, c, x.storage_offset(), 0)], (_cdiv(n, 64), 1, 1))
    return out


def _is_word(t):
    return _is_dev(t) and t.dtype in (torch.int64, torch.int32) and t.is_contiguous()


@op(A.eq.Tensor)
def eq_tensor(a, b):
    if not (_is_word(a) and _is_word(b)) or a.shape != b.shape:
        return NotImplemented
    n = a.numel()
    out = new(a.shape, torch.bool)
    u = REC.uniform(n, a.storage_offset(), b.storage_offset(), 0, 0, 0, 0, 0)
    _emit("eqw", [a.st.buf, b.st.buf, out.st.buf, u], (_cdiv(n, 64), 1, 1))
    return out


@op(A.eq.Scalar)
def eq_scalar(a, v):
    if not _is_word(a) or isinstance(v, float):
        return NotImplemented
    n = a.numel()
    out = new(a.shape, torch.bool)
    u = REC.uniform(n, a.storage_offset(), 0, 0, 1, int(v) & 0xFFFFFFFF, 0, 0)
    _emit("eqw", [a.st.buf, a.st.buf, out.st.buf, u], (_cdiv(n, 64), 1, 1))
    return out


@op(A.sum.default)
def sum_all_any(x, dtype=None):
    if _is_dev(x) and x.dtype in (torch.bool, torch.int64, torch.int32) and x.is_contiguous() and dtype in (None, torch.int64):
        n = x.numel()
        out = new((), torch.int64)
        _emit("isum", [x.st.buf, out.st.buf, REC.uniform(n, x.storage_offset(), 0, 0)], (1, 1, 1))
        return out
    return _reduce(x, None, False, False)


# --- average pooling (2x2, stride 2) ---------------------------------------------------------------

def _avgpool_ok(x, k, s, p, ceil, include_pad, divisor):
    return (_f32(x) and x.dim() == 4 and tuple(k) in ((2,), (2, 2)) and tuple(s or k) in ((2,), (2, 2))
            and tuple(p) in ((0,), (0, 0)) and not ceil and divisor is None)


@op(A.avg_pool2d.default)
def avg_pool2d(x, k, s=(), p=(0, 0), ceil=False, include_pad=True, divisor=None):
    if not _avgpool_ok(x, k, s, p, ceil, include_pad, divisor):
        return NotImplemented
    x = _zero_off(x)
    n, c, h, w = x.shape
    out = new((n, c, h // 2, w // 2))
    q = REC.uniform(n, c, h, w, h // 2, w // 2, 0, 0)
    _emit("avgpool_fwd", [x.st.buf, out.st.buf, q], _ew(n * c * (h // 2) * (w // 2)))
    return out


@op(A.avg_pool2d_backward.default)
def avg_pool2d_backward(gy, x, k, s, p, ceil, include_pad, divisor):
    if not _avgpool_ok(x, k, s, p, ceil, include_pad, divisor) or not _f32(gy):
        return NotImplemented
    gy = _zero_off(gy)
    n, c, h, w = x.shape
    gx = new(x.shape)
    q = REC.uniform(n, c, h, w, h // 2, w // 2, 0, 0)
    _emit("avgpool_bwd", [gy.st.buf, gx.st.buf, q], _ew(n * c * h * w))
    return gx


# --- loss (fused, via __torch_function__) ------------------------------------------------------

class _CrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, eps):
        n, c = logits.shape
        gout, losses = new((n, c)), new((n,))
        lg = _zero_off(logits)
        u = REC.uniform(n, c, float(eps), 0.0)
        _emit("ce", [lg.st.buf, target.st.buf, gout.st.buf, losses.st.buf, u], (_cdiv(n, 64), 1, 1))
        ctx.gout = gout
        return mean_all(losses)

    @staticmethod
    def backward(ctx, g):
        return binary(2, ctx.gout, g), None, None


def _cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None,
                   reduction="mean", label_smoothing=0.0):
    if (weight is not None or reduction != "mean" or ignore_index != -100 or not _is_dev(input)
            or not _is_dev(target) or target.dtype != torch.int64 or input.dim() != 2
            or input.dtype != torch.float32 or not target.is_contiguous() or target.storage_offset() != 0):
        return NotImplemented
    return _CrossEntropy.apply(input, target, label_smoothing)


FUSED[F.cross_entropy] = _cross_entropy


# --- fused Adam ---------------------------------------------------------------------------------

_hyper = _g.device.create_buffer(size=32, usage=_g.wgpu.BufferUsage.UNIFORM | _g.wgpu.BufferUsage.COPY_DST)


@op(A._fused_adam_.default)
def fused_adam(params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, *, lr, beta1, beta2,
               weight_decay, eps, amsgrad, maximize, grad_scale=None, found_inf=None):
    if amsgrad or maximize or grad_scale is not None or found_inf is not None:
        return NotImplemented
    steps = {float(s.item()) for s in state_steps}  # CPU tensors: no GPU sync
    if len(steps) != 1:
        return NotImplemented
    step = steps.pop()
    hyper = np.array([lr, beta1, beta2, eps, 1 - beta1 ** step, (1 - beta2 ** step) ** 0.5, weight_decay, 0],
                     np.float32)
    REC.write_words(_hyper, hyper)
    items = list(zip(params, grads, exp_avgs, exp_avg_sqs))
    i = 0
    while i + 4 <= len(items):
        grp = items[i:i + 4]
        rw = [id(t.st.buf) for (p, _, m, v) in grp for t in (p, m, v)]
        if len(set(rw)) == len(rw):  # one dispatch for four parameters (rw buffers must be distinct)
            counts = [p.numel() for p, *_ in grp]
            u = REC.uniform(*counts, *[p.storage_offset() for p, *_ in grp], *[g_.storage_offset() for _, g_, *_ in grp],
                            *[m.storage_offset() for _, _, m, _ in grp], *[v.storage_offset() for *_, v in grp])
            bufs = ([p.st.buf for p, *_ in grp] + [g_.st.buf for _, g_, *_ in grp]
                    + [m.st.buf for _, _, m, _ in grp] + [v.st.buf for *_, v in grp] + [u, _hyper])
            _emit("adam4", bufs, _ew(sum(counts)))
            i += 4
        else:
            break
    for p, g_, m, v in items[i:]:
        n = p.numel()
        u = REC.uniform(n, p.storage_offset(), g_.storage_offset(), m.storage_offset(), v.storage_offset(),
                        0, 0, 0)
        _emit("adam", [p.st.buf, g_.st.buf, m.st.buf, v.st.buf, u, _hyper], _ew(n))
    return None


_add_inplace = inplace(lambda a, b: binary(0, a, b))


@op(A._foreach_add_.Scalar)
def foreach_add_(tensors, scalar):
    if not all(_f32(t) and t.is_contiguous() for t in tensors):
        return NotImplemented
    for t in tensors:
        _add_inplace(t, scalar)
    return None


# --- dispatch and CPU fallback ---------------------------------------------------------------------

def dispatch(func, args, kwargs):
    impl = TABLE.get(func)
    if impl is not None:
        r = impl(*args, **kwargs)
        if r is not NotImplemented:
            return r
    return fallback(func, args, kwargs)


def fallback(func, args, kwargs):
    FALLBACKS[func] += 1
    flat, spec = pytree.tree_flatten((args, kwargs))
    mapping = {}

    def down(a):
        if _is_dev(a):
            c = to_cpu(a)
            mapping[id(c)] = a
            return c
        return a

    try:
        cpu_flat = [down(a) for a in flat]
    except AttributeError:
        raise RuntimeError(f"device fallback for {func}: tensor without storage: "
                           f"{[(type(a).__name__, tuple(a.shape)) for a in flat if isinstance(a, torch.Tensor)]}")
    cpu_args, cpu_kwargs = pytree.tree_unflatten(cpu_flat, spec)
    out = func(*cpu_args, **cpu_kwargs)
    # in-place results: copy mutated CPU tensors back into their device originals
    schema_args = func._schema.arguments
    for i, a in enumerate(cpu_args):
        if i < len(schema_args) and schema_args[i].alias_info is not None and schema_args[i].alias_info.is_write \
                and id(a) in mapping:
            copy_(mapping[id(a)], a)

    def up(o):
        if isinstance(o, torch.Tensor) and not _is_dev(o):
            if id(o) in mapping:
                return mapping[id(o)]
            return from_cpu(o) if o.dtype in (torch.float32, torch.int64, torch.int32, torch.bool) else o
        return o

    return pytree.tree_map(up, out)
