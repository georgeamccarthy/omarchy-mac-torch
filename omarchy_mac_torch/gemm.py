"""WGSL implicit-GEMM convolution kernels for the native backend.

One generated kernel per (kind, tile): C[M, N] = sum_k A[m, k] B[k, n], where the convolution's im2col is done by
the loaders (any stride and padding), 256 threads per workgroup, each computing a TM x TN block in registers.

  kind 0  forward      m = (n, oy, ox)   k = (ci, ky, kx)   n = co        A = im2col(x)    B = w
  kind 1  grad input   m = (n, y, x)     k = (co, ky, kx)   n = ci        A = im2col(gy)   B = w   (stride 1 only)
  kind 2  grad weight  m = co            k = (n, oy, ox)    n = (ci,ky,kx) A = gy          B = im2col(x), split-K
  kind 3  forward with the previous BatchNorm + ReLU applied in the input loader (eval plans)
"""

# (bm, bn, tm, tn): workgroup tile and per-thread register block; threads = (bm/tm) * (bn/tn) = 256 for all.
TILES = [(64, 64, 4, 4), (256, 16, 4, 4), (16, 256, 4, 4), (128, 32, 4, 4), (32, 128, 4, 4),
         (128, 64, 8, 4), (64, 128, 4, 8), (128, 128, 8, 8), (256, 32, 8, 4), (32, 256, 4, 8)]
BK = 16

PARAMS = """struct P { n: u32, ci: u32, co: u32, ih: u32, iw: u32, oh: u32, ow: u32, ks: u32,
           stride: u32, pad: u32, M: u32, N: u32, K: u32, kslice: u32, slices: u32, p2: u32 };"""

_FWD = """
fn aval(m: u32, k: u32) -> f32 {
  let ohw = p.oh * p.ow; let n = m / ohw; let r = m % ohw; let oy = r / p.ow; let ox = r % p.ow;
  let kk = p.ks * p.ks; let c = k / kk; let kr = k % kk; let ky = kr / p.ks; let kx = kr % p.ks;
  let yy = i32(oy * p.stride + ky) - i32(p.pad); let xx = i32(ox * p.stride + kx) - i32(p.pad);
  if (yy < 0 || yy >= i32(p.ih) || xx < 0 || xx >= i32(p.iw)) { return 0.0; }
  let v = a[((n * p.ci + c) * p.ih + u32(yy)) * p.iw + u32(xx)];
  RETURN
}
fn bval(k: u32, nn: u32) -> f32 { return b[nn * p.K + k]; }
fn store(m: u32, nn: u32, v: f32, z: u32) { let ohw = p.oh * p.ow; out[((m / ohw) * p.co + nn) * ohw + m % ohw] = v; }
"""
_GIN = """
fn aval(m: u32, k: u32) -> f32 {
  let ihw = p.ih * p.iw; let n = m / ihw; let r = m % ihw; let y = r / p.iw; let x = r % p.iw;
  let kk = p.ks * p.ks; let co = k / kk; let kr = k % kk; let ky = kr / p.ks; let kx = kr % p.ks;
  let yy = i32(y) + i32(p.pad) - i32(ky); let xx = i32(x) + i32(p.pad) - i32(kx);
  if (yy < 0 || yy >= i32(p.oh) || xx < 0 || xx >= i32(p.ow)) { return 0.0; }
  return a[((n * p.co + co) * p.oh + u32(yy)) * p.ow + u32(xx)];
}
fn bval(k: u32, nn: u32) -> f32 { let kk = p.ks * p.ks; return b[((k / kk) * p.ci + nn) * kk + k % kk]; }
fn store(m: u32, nn: u32, v: f32, z: u32) { let ihw = p.ih * p.iw; out[((m / ihw) * p.ci + nn) * ihw + m % ihw] = v; }
"""
_GW = """
fn aval(m: u32, k: u32) -> f32 { let ohw = p.oh * p.ow; return a[((k / ohw) * p.co + m) * ohw + k % ohw]; }
fn bval(k: u32, nn: u32) -> f32 {
  let ohw = p.oh * p.ow; let s = k / ohw; let r = k % ohw; let oy = r / p.ow; let ox = r % p.ow;
  let kk = p.ks * p.ks; let c = nn / kk; let kr = nn % kk; let ky = kr / p.ks; let kx = kr % p.ks;
  let yy = i32(oy * p.stride + ky) - i32(p.pad); let xx = i32(ox * p.stride + kx) - i32(p.pad);
  if (yy < 0 || yy >= i32(p.ih) || xx < 0 || xx >= i32(p.iw)) { return 0.0; }
  return b[((s * p.ci + c) * p.ih + u32(yy)) * p.iw + u32(xx)];
}
fn store(m: u32, nn: u32, v: f32, z: u32) { out[(z * p.M + m) * p.N + nn] = v; }
"""
MM_PARAMS = """struct P { M: u32, N: u32, K: u32, hb: u32, ao: u32, bo: u32, co: u32, bio: u32,
           sa0: u32, sa1: u32, sb0: u32, sb1: u32, kslice: u32, slices: u32, p0: u32, p1: u32 };"""
_MM = """
fn aval(m: u32, k: u32) -> f32 { return a[p.ao + m * p.sa0 + k * p.sa1]; }
fn bval(k: u32, nn: u32) -> f32 { return b[p.bo + k * p.sb0 + nn * p.sb1]; }
fn store(m: u32, nn: u32, v: f32, z: u32) {
  if (p.slices == 1u) {
    var r = v;
    if (p.hb == 1u) { r += bias[p.bio + nn]; }
    out[p.co + m * p.N + nn] = r;
  } else {
    out[(z * p.M + m) * p.N + nn] = v;
  }
}
"""
MM_REDUCE = MM_PARAMS + """
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  let total = p.M * p.N;
  if (i >= total) { return; }
  var s = 0.0;
  for (var z = 0u; z < p.slices; z++) { s += part[z * total + i]; }
  if (p.hb == 1u) { s += bias[p.bio + i % p.N]; }
  out[p.co + i] = s;
}
"""

BN_BINDINGS = ("@group(0) @binding(4) var<storage, read> bnm: array<f32>;\n"
               "@group(0) @binding(5) var<storage, read> bnv: array<f32>;\n"
               "@group(0) @binding(6) var<storage, read> bng: array<f32>;\n"
               "@group(0) @binding(7) var<storage, read> bnb: array<f32>;\n")


def source(kind, bm, bn, tm, tn, a_kfast=False, b_nfast=False):
    """WGSL for one kernel. Thread (lid.x, lid.y) owns rows lid.y*tm.. and columns lid.x*tn.. of the workgroup tile."""
    nt = (bm // tm) * (bn // tn)
    fwd = kind in (0, 3)
    body = _MM if kind == 4 else (_FWD if fwd else (_GIN if kind == 1 else _GW))
    if kind == 3:
        body = body.replace("RETURN", "return max((v - bnm[c]) * inverseSqrt(bnv[c] + 1e-5) * bng[c] + bnb[c], 0.0);")
    else:
        body = body.replace("RETURN", "return v;")
    a_m_fast = (not a_kfast) if kind == 4 else kind != 2   # A: consecutive threads along m (kinds 0, 1) or along k (kind 2)
    b_n_fast = b_nfast if kind == 4 else kind == 1         # B: along n for grad input, along k otherwise
    a_idx = "let mm = e % BM; let ka = e / BM;" if a_m_fast else "let ka = e % 16u; let mm = e / 16u;"
    b_idx = "let nn = e % BN; let kb = e / BN;" if b_n_fast else "let kb = e % 16u; let nn = e / 16u;"
    acc = [[f"c{i}_{j}" for j in range(tn // 4)] for i in range(tm)]
    decl = "\n".join(f"  var {c} = vec4<f32>();" for row in acc for c in row)
    inner = "\n".join(f"      let rb{j} = vec4<f32>(Bs[bb + {4 * j}u], Bs[bb + {4 * j + 1}u], Bs[bb + {4 * j + 2}u], Bs[bb + {4 * j + 3}u]);"
                      for j in range(tn // 4))
    inner += "\n" + "\n".join(f"      let ra{i} = As[ab + {i}u];" for i in range(tm))
    inner += "\n" + "\n".join(f"      {acc[i][j]} += ra{i} * rb{j};" for i in range(tm) for j in range(tn // 4))
    epi = []
    for i in range(tm):
        for j in range(tn // 4):
            for c, comp in enumerate("xyzw"):
                epi.append(f"  {{ let m = m0 + lid.y * {tm}u + {i}u; let nn = n0 + lid.x * {tn}u + {4 * j + c}u;\n"
                           f"    if (m < p.M && nn < p.N) {{ store(m, nn, {acc[i][j]}.{comp}, wg.z); }} }}")
    extra = BN_BINDINGS if kind == 3 else ("@group(0) @binding(4) var<storage, read> bias: array<f32>;" if kind == 4 else "")
    return f"""{MM_PARAMS if kind == 4 else PARAMS}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
{extra}
const BM = {bm}u; const BN = {bn}u;
var<workgroup> As: array<f32, {BK * bm}>;
var<workgroup> Bs: array<f32, {BK * bn}>;
{body}
@compute @workgroup_size({bn // tn}, {bm // tm})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
  let t = lid.y * {bn // tn}u + lid.x;
  let m0 = wg.y * BM; let n0 = wg.x * BN;
  let kbeg = wg.z * p.kslice; let kend = min(p.K, kbeg + p.kslice);
{decl}
  for (var k0 = kbeg; k0 < kend; k0 += 16u) {{
    for (var e = t; e < {BK * bm}u; e += {nt}u) {{
      {a_idx}
      let m = m0 + mm; let k = k0 + ka;
      var v = 0.0;
      if (m < p.M && k < kend) {{ v = aval(m, k); }}
      As[ka * BM + mm] = v;
    }}
    for (var e = t; e < {BK * bn}u; e += {nt}u) {{
      {b_idx}
      let nn2 = n0 + nn; let k2 = k0 + kb;
      var u = 0.0;
      if (nn2 < p.N && k2 < kend) {{ u = bval(k2, nn2); }}
      Bs[kb * BN + nn] = u;
    }}
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {{
      let ab = kk * BM + lid.y * {tm}u; let bb = kk * BN + lid.x * {tn}u;
{inner}
    }}
    workgroupBarrier();
  }}
{chr(10).join(epi)}
}}
"""


# ---- pooling kernels (general window / stride / padding; indices in PyTorch's format: flat position in the plane) ----
_S = "struct S { n: u32, c: u32, ih: u32, iw: u32, oh: u32, ow: u32, k: u32, stride: u32, pad: u32, p0: u32, p1: u32, p2: u32 };"
MAXPOOL_FWD = _S + """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> idx: array<i32>;
@group(0) @binding(3) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.n * s.c * s.oh * s.ow) { return; }
  let ox = i % s.ow; let oy = (i / s.ow) % s.oh; let nc = i / (s.ow * s.oh);
  var best = -3.4e38; var bi = 0u;
  for (var ky = 0u; ky < s.k; ky++) {
    let y = i32(oy * s.stride + ky) - i32(s.pad);
    if (y < 0 || y >= i32(s.ih)) { continue; }
    for (var kx = 0u; kx < s.k; kx++) {
      let xx = i32(ox * s.stride + kx) - i32(s.pad);
      if (xx < 0 || xx >= i32(s.iw)) { continue; }
      let pos = u32(y) * s.iw + u32(xx);
      let v = x[nc * s.ih * s.iw + pos];
      if (v > best) { best = v; bi = pos; }
    }
  }
  out[i] = best; idx[i] = i32(bi);
}
"""
MAXPOOL_BWD = _S + """
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> gx: array<f32>;
@group(0) @binding(3) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.n * s.c * s.ih * s.iw) { return; }
  let xx = i32(i % s.iw); let y = i32((i / s.iw) % s.ih); let nc = i / (s.iw * s.ih);
  let st = i32(s.stride); let k = i32(s.k); let pd = i32(s.pad);
  let ny = y + pd - k + 1; let nx = xx + pd - k + 1;
  let ylo = select(0, (ny + st - 1) / st, ny > 0); let yhi = min(i32(s.oh) - 1, (y + pd) / st);
  let xlo = select(0, (nx + st - 1) / st, nx > 0); let xhi = min(i32(s.ow) - 1, (xx + pd) / st);
  let me = y * i32(s.iw) + xx;
  var acc = 0.0;
  for (var oy = ylo; oy <= yhi; oy++) {
    for (var ox = xlo; ox <= xhi; ox++) {
      let o = (nc * s.oh + u32(oy)) * s.ow + u32(ox);
      if (idx[o] == me) { acc += g[o]; }
    }
  }
  gx[i] = acc;
}
"""
ADAPOOL_FWD = _S + """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.n * s.c * s.oh * s.ow) { return; }
  let ox = i % s.ow; let oy = (i / s.ow) % s.oh; let nc = i / (s.ow * s.oh);
  let y0 = (oy * s.ih) / s.oh; let y1 = ((oy + 1u) * s.ih + s.oh - 1u) / s.oh;
  let x0 = (ox * s.iw) / s.ow; let x1 = ((ox + 1u) * s.iw + s.ow - 1u) / s.ow;
  var acc = 0.0;
  for (var y = y0; y < y1; y++) { for (var xx = x0; xx < x1; xx++) { acc += x[(nc * s.ih + y) * s.iw + xx]; } }
  out[i] = acc / f32((y1 - y0) * (x1 - x0));
}
"""
ADAPOOL_BWD = _S + """
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read_write> gx: array<f32>;
@group(0) @binding(2) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.n * s.c * s.ih * s.iw) { return; }
  let xx = i % s.iw; let y = (i / s.iw) % s.ih; let nc = i / (s.iw * s.ih);
  var acc = 0.0;
  for (var oy = 0u; oy < s.oh; oy++) {
    let y0 = (oy * s.ih) / s.oh; let y1 = ((oy + 1u) * s.ih + s.oh - 1u) / s.oh;
    if (y < y0 || y >= y1) { continue; }
    for (var ox = 0u; ox < s.ow; ox++) {
      let x0 = (ox * s.iw) / s.ow; let x1 = ((ox + 1u) * s.iw + s.ow - 1u) / s.ow;
      if (xx < x0 || xx >= x1) { continue; }
      acc += g[(nc * s.oh + oy) * s.ow + ox] / f32((y1 - y0) * (x1 - x0));
    }
  }
  gx[i] = acc;
}
"""
