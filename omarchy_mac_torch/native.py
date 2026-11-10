"""Native (C++) backend setup: loads _native, registers WGSL kernel sources and the conv tile planner,
renames the PrivateUse1 device to "omarchy_mac_torch" and installs the small PyTorch patches the backend relies on."""

import ctypes
import json
import os
import pathlib
import sys
import types
from unittest import mock

import torch

_HERE = pathlib.Path(__file__).parent
ctypes.CDLL(str(_HERE / "libwgpu_native-release.so"), mode=ctypes.RTLD_GLOBAL)
from omarchy_mac_torch import _native as core  # noqa: E402

DEVICE = "omarchy_mac_torch"
torch.utils.rename_privateuse1_backend(DEVICE)

# ---- kernel sources: reuse the generators in ops.py / fused.py / conv.py without touching wgpu-py ----
import omarchy_mac_torch.conv as _conv  # noqa: E402

_real_gpu = _conv.gpu
_conv.gpu = lambda: mock.MagicMock()
import omarchy_mac_torch.fused as _fused  # noqa: E402
import omarchy_mac_torch.ops as _ops  # noqa: E402

_conv.gpu = _real_gpu

for _name, _src in {**_fused._SOURCES, **_ops._SRC}.items():
    core.register_source(_name, _src)
core.register_source("gemm_reduce", _conv._REDUCE)

_Q, _RED = _fused._Q, _fused._REDUCE_FN
_EXTRA = {
    "csum_nchw": _Q + """
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> q: Q;
""" + _RED + """
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let ch = wg.x; let hw = q.h * q.w; var s = 0.0;
  for (var nn = 0u; nn < q.n; nn++) {
    let base = (nn * q.c + ch) * hw;
    for (var e = t; e < hw; e += 256u) { s += g[base + e]; }
  }
  let r = wg_sum(t, s);
  if (t == 0u) { out[ch] = r; }
}
""",
    "cmpw": """
struct S { n: u32, op: u32, bs: u32, ao: u32, bo: u32, p0: u32, sval: f32, p1: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<u32>;
@group(0) @binding(3) var<uniform> s: S;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= s.n) { return; }
  let x = a[s.ao + i];
  var y = s.sval;
  if (s.bs == 0u) { y = b[s.bo + i]; }
  var r = false;
  switch (s.op) {
    case 0u: { r = x < y; }
    case 1u: { r = x <= y; }
    case 2u: { r = x > y; }
    case 3u: { r = x >= y; }
    case 4u: { r = x != y; }
    default: { r = x == y; }
  }
  o[i] = select(0u, 1u, r);
}
""",
    "where3": """
struct S { n: u32, co: u32, ao: u32, bo: u32, p0: u32, p1: u32, p2: u32, p3: u32 };
@group(0) @binding(0) var<storage, read> c: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<uniform> s: S;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= s.n) { return; }
  o[i] = select(b[s.bo + i], a[s.ao + i], c[s.co + i] != 0u);
}
""",
    "gather0": """
struct S { n: u32, row: u32, xo: u32, io: u32 };
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> o: array<u32>;
@group(0) @binding(3) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.n * s.row) { return; }
  let r = i / s.row; let c = i % s.row;
  o[i] = x[s.xo + u32(idx[s.io + r]) * s.row + c];
}
""",
    "dropout": """
struct S { n: u32, seed: u32, p: f32, scale: f32, xo: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<u32>;
@group(0) @binding(3) var<uniform> s: S;
fn hash(v: u32) -> u32 {
  var x = v * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (x >> 22u) ^ x;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= s.n) { return; }
  let h = hash(i ^ hash(s.seed));
  let u = f32(h >> 8u) * (1.0 / 16777216.0);
  let keep = u >= s.p;
  o[i] = select(0.0, x[s.xo + i] * s.scale, keep);
  m[i] = select(0u, 1u, keep);
}
""",
    "mulmask": """
struct S { n: u32, scale: f32, go: u32, p: u32 };
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> m: array<u32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> s: S;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= s.n) { return; }
  o[i] = select(0.0, g[s.go + i] * s.scale, m[i] != 0u);
}
""",
    "ln_fwd": """
struct S { R: u32, D: u32, eps: f32, aff: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<storage, read_write> mean: array<f32>;
@group(0) @binding(5) var<storage, read_write> rstd: array<f32>;
@group(0) @binding(6) var<uniform> s: S;
""" + _RED + """
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let r = wg.x; let dd = s.D;
  var sum = 0.0;
  for (var e = t; e < dd; e += 256u) { sum += x[r * dd + e]; }
  let mu = wg_sum(t, sum) / f32(dd);
  var v = 0.0;
  for (var e = t; e < dd; e += 256u) { let d = x[r * dd + e] - mu; v += d * d; }
  let variance = wg_sum(t, v) / f32(dd);
  let rs = inverseSqrt(variance + s.eps);
  for (var e = t; e < dd; e += 256u) {
    var y = (x[r * dd + e] - mu) * rs;
    if (s.aff == 1u) { y = y * g[e] + b[e]; }
    o[r * dd + e] = y;
  }
  if (t == 0u) { mean[r] = mu; rstd[r] = rs; }
}
""",
    "ln_bwd_x": """
struct S { R: u32, D: u32, eps: f32, aff: u32 };
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> rstd: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;
@group(0) @binding(6) var<uniform> s: S;
""" + _RED + """
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let r = wg.x; let dd = s.D; let mu = mean[r]; let rs = rstd[r];
  var s1 = 0.0; var s2 = 0.0;
  for (var e = t; e < dd; e += 256u) {
    let xh = (x[r * dd + e] - mu) * rs;
    var dg = dy[r * dd + e];
    if (s.aff == 1u) { dg = dg * g[e]; }
    s1 += dg; s2 += dg * xh;
  }
  let a1 = wg_sum(t, s1) / f32(dd);
  let a2 = wg_sum(t, s2) / f32(dd);
  for (var e = t; e < dd; e += 256u) {
    let xh = (x[r * dd + e] - mu) * rs;
    var dg = dy[r * dd + e];
    if (s.aff == 1u) { dg = dg * g[e]; }
    dx[r * dd + e] = rs * (dg - a1 - xh * a2);
  }
}
""",
    "ln_bwd_w": """
struct S { R: u32, D: u32, eps: f32, aff: u32 };
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> mean: array<f32>;
@group(0) @binding(3) var<storage, read> rstd: array<f32>;
@group(0) @binding(4) var<storage, read_write> dg: array<f32>;
@group(0) @binding(5) var<storage, read_write> db: array<f32>;
@group(0) @binding(6) var<uniform> s: S;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  if (d >= s.D) { return; }
  var a = 0.0; var b = 0.0;
  for (var r = 0u; r < s.R; r++) {
    let g = dy[r * s.D + d];
    a += g * (x[r * s.D + d] - mean[r]) * rstd[r];
    b += g;
  }
  dg[d] = a; db[d] = b;
}
""",
}

core.register_source("sgd", """
struct H { lr: f32, mom: f32, damp: f32, wd: f32, nesterov: u32, first: u32, maximize: u32, p: u32 };
struct A { count: u32, po: u32, go: u32, mo: u32 };
@group(0) @binding(0) var<storage, read_write> p: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<uniform> a: A;
@group(0) @binding(4) var<uniform> h: H;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= a.count) { return; }
  var gr = g[a.go + i];
  if (h.maximize == 1u) { gr = -gr; }
  let pv = p[a.po + i];
  gr = gr + h.wd * pv;
  var d = gr;
  if (h.mom != 0.0) {
    var v = gr;
    if (h.first == 0u) { v = h.mom * m[a.mo + i] + (1.0 - h.damp) * gr; }
    m[a.mo + i] = v;
    d = select(v, gr + h.mom * v, h.nesterov == 1u);
  }
  p[a.po + i] = pv - h.lr * d;
}
""")

_EXTRA2 = {
    "bn_stats_part": _Q + """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> part: array<f32>;
@group(0) @binding(2) var<uniform> q: Q;
""" + _RED + """
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let ch = wg.x; let sl = wg.y; let hw = q.h * q.w; let ns = q.p0;
  let shift = x[ch * hw];
  var s1 = 0.0; var s2 = 0.0;
  for (var nn = sl; nn < q.n; nn += ns) {
    let base = (nn * q.c + ch) * hw;
    for (var e = t; e < hw; e += 256u) { let d = x[base + e] - shift; s1 += d; s2 += d * d; }
  }
  let r1 = wg_sum(t, s1);
  let r2 = wg_sum(t, s2);
  if (t == 0u) { part[(ch * ns + sl) * 2u] = r1; part[(ch * ns + sl) * 2u + 1u] = r2; }
}
""",
    "bn_stats_fin": _Q + """
struct R { mom: f32, corr: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> part: array<f32>;
@group(0) @binding(2) var<storage, read_write> stats: array<f32>;
@group(0) @binding(3) var<storage, read_write> rm: array<f32>;
@group(0) @binding(4) var<storage, read_write> rv: array<f32>;
@group(0) @binding(5) var<uniform> q: Q;
@group(0) @binding(6) var<uniform> r: R;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x;
  if (ch >= q.c) { return; }
  let hw = q.h * q.w; let ns = q.p0; let m = f32(q.n * hw);
  var s1 = 0.0; var s2 = 0.0;
  for (var k = 0u; k < ns; k++) { s1 += part[(ch * ns + k) * 2u]; s2 += part[(ch * ns + k) * 2u + 1u]; }
  let a = s1 / m; let mean = x[ch * hw] + a; let vr = max(s2 / m - a * a, 0.0);
  stats[ch] = mean; stats[q.c + ch] = vr;
  rm[ch] = rm[ch] + r.mom * (mean - rm[ch]);
  rv[ch] = rv[ch] + r.mom * (vr * r.corr - rv[ch]);
}
""",
    "bn_bwd_part": _Q + """
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> part: array<f32>;
@group(0) @binding(5) var<uniform> q: Q;
""" + _RED + """
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let ch = wg.x; let sl = wg.y; let hw = q.h * q.w; let ns = q.p0;
  let mean = stats[ch]; let inv = inverseSqrt(stats[q.c + ch] + 1e-5);
  var s1 = 0.0; var s2 = 0.0;
  for (var nn = sl; nn < q.n; nn += ns) {
    let base = (nn * q.c + ch) * hw;
    for (var e = t; e < hw; e += 256u) {
      let i = base + e;
      let gz = select(0.0, g[i], y[i] > 0.0);
      s1 += gz; s2 += gz * (x[i] - mean) * inv;
    }
  }
  let r1 = wg_sum(t, s1);
  let r2 = wg_sum(t, s2);
  if (t == 0u) { part[(ch * ns + sl) * 2u] = r1; part[(ch * ns + sl) * 2u + 1u] = r2; }
}
""",
    "bn_bwd_fin": _Q + """
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> dgb: array<f32>;
@group(0) @binding(2) var<uniform> q: Q;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x;
  if (ch >= q.c) { return; }
  let ns = q.p0;
  var s1 = 0.0; var s2 = 0.0;
  for (var k = 0u; k < ns; k++) { s1 += part[(ch * ns + k) * 2u]; s2 += part[(ch * ns + k) * 2u + 1u]; }
  dgb[ch] = s2; dgb[q.c + ch] = s1;
}
""",
}
for _name, _src in _EXTRA2.items():
    core.register_source(_name, _src)


core.register_source("mm4", """
struct M { m: u32, n: u32, k: u32, hb: u32, ao: u32, bo: u32, co: u32, bio: u32, sa0: u32, sa1: u32, sb0: u32, sb1: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@group(0) @binding(4) var<uniform> u: M;
var<workgroup> As: array<f32, 1024>;   // [k][row]
var<workgroup> Bs: array<f32, 1024>;   // [k][col]
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = l.y * 16u + l.x;
  let row0 = wg.y * 64u; let col0 = wg.x * 64u;
  var c0 = vec4<f32>(); var c1 = vec4<f32>(); var c2 = vec4<f32>(); var c3 = vec4<f32>();
  for (var k0 = 0u; k0 < u.k; k0 += 16u) {
    for (var i = 0u; i < 4u; i++) {
      let e = t + i * 256u;
      let kk = e % 16u; let r = e / 16u;
      var av = 0.0;
      if (row0 + r < u.m && k0 + kk < u.k) { av = a[u.ao + (row0 + r) * u.sa0 + (k0 + kk) * u.sa1]; }
      As[kk * 64u + r] = av;
      let cc = e % 64u; let kb = e / 64u;
      var bv = 0.0;
      if (k0 + kb < u.k && col0 + cc < u.n) { bv = b[u.bo + (k0 + kb) * u.sb0 + (col0 + cc) * u.sb1]; }
      Bs[kb * 64u + cc] = bv;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      let ab = kk * 64u + l.y * 4u; let bb = kk * 64u + l.x * 4u;
      let rb = vec4<f32>(Bs[bb], Bs[bb + 1u], Bs[bb + 2u], Bs[bb + 3u]);
      c0 += As[ab] * rb; c1 += As[ab + 1u] * rb; c2 += As[ab + 2u] * rb; c3 += As[ab + 3u] * rb;
    }
    workgroupBarrier();
  }
  let rows = array<vec4<f32>, 4>(c0, c1, c2, c3);
  for (var i = 0u; i < 4u; i++) {
    for (var j = 0u; j < 4u; j++) {
      let m = row0 + l.y * 4u + i; let nn = col0 + l.x * 4u + j;
      if (m < u.m && nn < u.n) {
        var v = rows[i][j];
        if (u.hb == 1u) { v += bias[u.bio + nn]; }
        c[u.co + m * u.n + nn] = v;
      }
    }
  }
}
""")

_BN_BINDINGS = ("@group(0) @binding(4) var<storage, read> bnm: array<f32>;\n"
                "@group(0) @binding(5) var<storage, read> bnv: array<f32>;\n"
                "@group(0) @binding(6) var<storage, read> bng: array<f32>;\n"
                "@group(0) @binding(7) var<storage, read> bnb: array<f32>;\n")
_BN_ACT = "max((%s - bnm[ch]) * inverseSqrt(bnv[ch] + 1e-5) * bng[ch] + bnb[ch], 0.0)"
_dw = _fused._SOURCES["dw_fwd"]
_old = "s += x[nc * hw + u32(sy) * q.w + u32(sx)] * wt[ch * q.p0 * q.p0 + u32(ky * ks + kx)];"
assert _old in _dw
_EXTRA["dw_fwd_bn"] = _dw.replace("@group(0) @binding(3) var<uniform> q: Q;", "@group(0) @binding(3) var<uniform> q: Q;\n" + _BN_BINDINGS).replace(
    _old, "s += " + _BN_ACT % "x[nc * hw + u32(sy) * q.w + u32(sx)]" + " * wt[ch * q.p0 * q.p0 + u32(ky * ks + kx)];")
for _name, _src in _EXTRA.items():
    core.register_source(_name, _src)

from omarchy_mac_torch import gemm as _gemm  # noqa: E402

core.register_source("mm_reduce", _gemm.MM_REDUCE)
core.register_source("maxpool_gen_fwd", _gemm.MAXPOOL_FWD)
core.register_source("maxpool_gen_bwd", _gemm.MAXPOOL_BWD)
core.register_source("adapool_fwd", _gemm.ADAPOOL_FWD)
core.register_source("adapool_bwd", _gemm.ADAPOOL_BWD)
_KINDS = ["forward", "grad_input", "grad_weight", "forward_bn"]


def _planner(kind, n, ci, co, ih, iw, oh, ow, ks, stride, pad, tile_index=-1):
    if kind in (0, 3):
        M, N = n * oh * ow, co
    elif kind == 1:
        M, N = n * ih * iw, ci
    else:
        M, N = co, ci * ks * ks
    if tile_index >= 0:
        tile = _gemm.TILES[tile_index]
    else:  # untuned default: least padding waste, then the larger register block
        tile = min(_gemm.TILES, key=lambda s: (_conv._cdiv(M, s[0]) * s[0] * _conv._cdiv(N, s[1]) * s[1], -s[2] * s[3]))
    bm, bn, tm, tn = tile
    kernel = f"gemm2:{_KINDS[kind]}:{bm}x{bn}x{tm}x{tn}"
    core.register_source(kernel, _gemm.source(kind, bm, bn, tm, tn))
    return (kernel, bm, bn)


core.set_conv_planner(_planner)


def _mm_planner(a_kfast, b_nfast, tile_index):
    bm, bn, tm, tn = _gemm.TILES[tile_index]
    name = f"mm2:{'k' if a_kfast else 'm'}{'n' if b_nfast else 'k'}:{bm}x{bn}x{tm}x{tn}"
    core.register_source(name, _gemm.source(4, bm, bn, tm, tn, a_kfast, b_nfast))
    return (name, bm, bn)


core.set_mm_planner(_mm_planner)
core.set_fold(int(os.environ.get("OMARCHY_MAC_TORCH_FOLD", "1")))
core.set_tuning(os.environ.get("OMARCHY_MAC_TORCH_TUNE", "1") != "0", len(_gemm.TILES))

# ---- device module so torch.omarchy_mac_torch.* works ----
_mod = types.ModuleType("torch.omarchy_mac_torch")
_mod.is_available = lambda: True
_mod.device_count = lambda: 1
_mod.current_device = lambda: 0
_mod.set_device = lambda d: None
_mod._is_in_bad_fork = lambda: False
_mod.manual_seed_all = lambda s: None
_mod.get_amp_supported_dtype = lambda: [torch.float32]
_mod.synchronize = lambda device=None: core.sync()
_mod.empty_cache = lambda: core.empty_cache()
_mod.memory_allocated = lambda device=None: core.memory_allocated()
_mod.memory_reserved = lambda device=None: core.memory_reserved()
torch._register_device_module("omarchy_mac_torch", _mod)

# ---- fused cross-entropy for device tensors ----
import torch.nn.functional as F  # noqa: E402

_cross_entropy = F.cross_entropy


def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean",
                  label_smoothing=0.0):
    if (input.device.type == "omarchy_mac_torch" and weight is None and reduction == "mean" and ignore_index == -100
            and input.dim() == 2 and input.dtype == torch.float32 and target.dtype == torch.int64
            and target.device.type == "omarchy_mac_torch" and size_average is None and reduce is None):
        return core.cross_entropy(input, target, float(label_smoothing))
    return _cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing)


F.cross_entropy = cross_entropy

# ---- Adam: fused kernel by default on the device; step counters stay on the CPU ----
_sgd_init = torch.optim.SGD.__init__


def _sgd_init_fused(self, params, *args, **kwargs):
    params = list(params)
    flat = [p for g in params for p in (g["params"] if isinstance(g, dict) else [g])]
    if flat and all(p.device.type == DEVICE for p in flat) and kwargs.get("fused") is None \
            and kwargs.get("foreach") is None and not kwargs.get("differentiable") and not isinstance(kwargs.get("lr", 1e-3), torch.Tensor):
        kwargs["fused"] = True
    _sgd_init(self, params, *args, **kwargs)


torch.optim.SGD.__init__ = _sgd_init_fused
_adam_init = torch.optim.Adam.__init__


def _init(self, params, *args, **kwargs):
    params = list(params)
    flat = [p for g in params for p in (g["params"] if isinstance(g, dict) else [g])]
    if flat and all(p.device.type == "omarchy_mac_torch" for p in flat) and kwargs.get("fused") is None \
            and kwargs.get("foreach") is None and not kwargs.get("differentiable"):
        kwargs["fused"] = True
    _adam_init(self, params, *args, **kwargs)


torch.optim.Adam.__init__ = _init
_init_group = torch.optim.Adam._init_group


def _init_group_cpu_steps(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps):
    r = _init_group(self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps)
    for i, p in enumerate(params_with_grad):
        st = self.state[p]["step"]
        if st.device.type == "omarchy_mac_torch":
            self.state[p]["step"] = state_steps[i] = torch.tensor(float(st.item()), dtype=torch.float32)
    return r


torch.optim.Adam._init_group = _init_group_cpu_steps


def set_queue_depth(depth=3, chunk=64):
    core.set_queue(chunk, depth)


def synchronize():
    core.sync()

# ---- copy / serialisation of device tensors (storage-level copies don't apply to GPU buffers) ----
_orig_deepcopy = torch.Tensor.__deepcopy__


def _deepcopy(self, memo):
    if self.device.type == "omarchy_mac_torch":
        out = self.detach().clone()
        out.requires_grad_(self.requires_grad)
        memo[id(self)] = out
        return out
    return _orig_deepcopy(self, memo)


torch.Tensor.__deepcopy__ = _deepcopy
_orig_reduce = torch.Tensor.__reduce_ex__


def _reduce(self, proto):
    if self.device.type == "omarchy_mac_torch":
        cpu = self.detach().cpu()
        cpu.requires_grad_(self.requires_grad)
        return cpu.__reduce_ex__(proto)
    return _orig_reduce(self, proto)


torch.Tensor.__reduce_ex__ = _reduce


# ---- CPU-fallback counter (counted inside C++) ----
class _Fallbacks:
    def keys(self):
        return core.fallback_counts().keys()

    def __getitem__(self, k):
        return core.fallback_counts()[k]

    def __iter__(self):
        return iter(core.fallback_counts())

    def items(self):
        return core.fallback_counts().items()

    def __len__(self):
        return len(core.fallback_counts())

    def __repr__(self):
        return repr(core.fallback_counts())


FALLBACKS = _Fallbacks()

# ---- fused conv/BN/ReLU/pool stacks: native plans (C++), driven by a small autograd Function ----
import weakref  # noqa: E402

from torch import nn  # noqa: E402

from omarchy_mac_torch.fused import _is_avgpool, _params, fusable  # noqa: E402


def _layer_specs(seq):
    specs, mods, i = [], list(seq), 0
    while i < len(mods):
        m = mods[i]
        if isinstance(m, nn.Conv2d) and m.groups > 1:
            specs.append(("dw", m.in_channels, m.out_channels, m.kernel_size[0], 0.0))
        elif isinstance(m, nn.Conv2d):
            specs.append(("conv", m.in_channels, m.out_channels, m.kernel_size[0], 0.0))
        elif isinstance(m, nn.BatchNorm2d):
            specs.append(("bn", m.num_features, m.num_features, 0, float(m.momentum)))
            i += 1  # the ReLU is part of the BN kernel
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            specs.append(("gap", 0, 0, 0, 0.0))
        elif _is_avgpool(m):
            specs.append(("apool", 0, 0, 0, 0.0))
        else:
            specs.append(("pool", 0, 0, 0, 0.0))
        i += 1
    return specs


_plans = weakref.WeakKeyDictionary()


def _plan(seq, shape, train, input_grad):
    cache = _plans.setdefault(seq, {})
    key = (tuple(shape), train, input_grad)
    plan = cache.get(key)
    if plan is None or plan.busy:
        bns = [m for m in seq if isinstance(m, nn.BatchNorm2d)]
        plan = core.make_plan(_layer_specs(seq), list(shape), train, input_grad, list(_params(seq)),
                              [m.running_mean for m in bns], [m.running_var for m in bns])
        if key not in cache:
            cache[key] = plan
    return plan


class _StackFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, seq, *params):
        plan = _plan(seq, x.shape, True, ctx.needs_input_grad[0])
        out = plan.forward(x)
        plan.busy = True
        ctx.plan = plan
        return out

    @staticmethod
    def backward(ctx, gout):
        plan = ctx.plan
        gx, grads = plan.backward(gout)
        plan.busy = False
        return (gx, None, *grads)


class FusedSequential(nn.Sequential):
    """The nn.Sequential it replaces (same modules and state_dict), run from a prebuilt native plan."""

    def forward(self, x):
        if x.device.type == "omarchy_mac_torch" and x.dtype == torch.float32 and x.dim() == 4:
            if self.training and torch.is_grad_enabled():
                return _StackFn.apply(x, self, *_params(self))
            if not self.training and not torch.is_grad_enabled():
                return _plan(self, x.shape, False, False).forward(x)
        return super().forward(x)


def accelerate(model, **kwargs):
    """Swap every nn.Sequential made only of supported layers for a native fused stack, in place."""
    count = 0
    for m in model.modules():
        if type(m) is nn.Sequential and fusable(m):
            m.__class__ = FusedSequential
            count += 1
    return f"{count} fused native stacks on {core.adapter_name()}"


_module_to = nn.Module.to


def _to(self, *args, **kwargs):
    out = _module_to(self, *args, **kwargs)
    device = torch._C._nn._parse_to(*args, **kwargs)[0]
    if device is not None and device.type == "omarchy_mac_torch":
        accelerate(out)
    return out


nn.Module.to = _to


def empty_cache():
    core.empty_cache()


def memory_allocated():
    return core.memory_allocated()


def memory_reserved():
    return core.memory_reserved()


# ---- on-disk cache of the GEMM tuning (a few seconds of timing per new convolution shape) ----
import atexit  # noqa: E402

_TUNE_FILE = os.path.expanduser("~/.cache/omarchy-mac-torch/tuning.json")
try:
    core.tune_load({k: tuple(v) for k, v in json.load(open(_TUNE_FILE)).items()})
except (OSError, ValueError, TypeError):
    pass


def _save_tuning():
    try:
        os.makedirs(os.path.dirname(_TUNE_FILE), exist_ok=True)
        old = {}
        try:
            old = json.load(open(_TUNE_FILE))
        except (OSError, ValueError):
            pass
        old.update({k: list(v) for k, v in core.tune_dump().items()})
        json.dump(old, open(_TUNE_FILE, "w"))
    except Exception:
        pass


atexit.register(_save_tuning)
