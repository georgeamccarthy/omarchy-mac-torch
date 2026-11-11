"""WGSL implicit-GEMM convolution kernels for the native backend.

One generated kernel per (kind, tile): C[M, N] = sum_k A[m, k] B[k, n], where the convolution's im2col is done by
the loaders (any stride and padding), 256 threads per workgroup, each computing a TM x TN block in registers.

  kind 0  forward      m = (n, oy, ox)   k = (ci, ky, kx)   n = co        A = im2col(x)    B = w
  kind 1  grad input   m = (n, y, x)     k = (co, ky, kx)   n = ci        A = im2col(gy)   B = w   (any stride)
  kind 2  grad weight  m = co            k = (n, oy, ox)    n = (ci,ky,kx) A = gy          B = im2col(x), split-K
  kind 3  forward with the previous BatchNorm + ReLU applied in the input loader (eval plans)
  kind 5  grad input of a strided conv, one stride phase (py, px) at a time: input pixels y = py + s*i only receive
          taps ky = ky0 + s*j, so each phase is a dense GEMM with K = co * ceil((ks - ky0) / s)^2, no wasted taps

The strided grad-input loader (skip taps that don't land on the output grid) follows TensorFlow.js's WebGPU
conv2d backprop-input kernel (conv_backprop_mm_webgpu.ts, Apache-2.0), adapted from NHWC to NCHW.
"""

# (bm, bn, tm, tn): workgroup tile and per-thread register block; threads = (bm/tm) * (bn/tn) = 256 for all.
TILES = [(64, 64, 4, 4), (256, 16, 4, 4), (16, 256, 4, 4), (128, 32, 4, 4), (32, 128, 4, 4),
         (128, 64, 8, 4), (64, 128, 4, 8), (128, 128, 8, 8), (256, 32, 8, 4), (32, 256, 4, 8)]
BK = 16

PARAMS = """struct P { n: u32, ci: u32, co: u32, ih: u32, iw: u32, oh: u32, ow: u32, ks: u32,
           stride: u32, pad: u32, M: u32, N: u32, K: u32, kslice: u32, slices: u32, hb: u32 };"""

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
fn store(m: u32, nn: u32, v: f32, z: u32) {
  let ohw = p.oh * p.ow;
  var r = v;
  if ((p.hb & 1u) != 0u) { r += bias[nn]; }   // hb bit 0: bias, bit 1: ReLU (fused epilogue)
  if ((p.hb & 2u) != 0u) { r = max(r, 0.0); }
  out[((m / ohw) * p.co + nn) * ohw + m % ohw] = r;
}
"""
_GIN = """
fn aval(m: u32, k: u32) -> f32 {
  let ihw = p.ih * p.iw; let n = m / ihw; let r = m % ihw; let y = r / p.iw; let x = r % p.iw;
  let kk = p.ks * p.ks; let co = k / kk; let kr = k % kk; let ky = kr / p.ks; let kx = kr % p.ks;
  var yy = i32(y) + i32(p.pad) - i32(ky); var xx = i32(x) + i32(p.pad) - i32(kx);
  if (yy < 0 || xx < 0) { return 0.0; }
  if (p.stride > 1u) {  // only taps on the stride grid came from an output pixel
    let st = i32(p.stride);
    if (yy % st != 0 || xx % st != 0) { return 0.0; }
    yy = yy / st; xx = xx / st;
  }
  if (yy >= i32(p.oh) || xx >= i32(p.ow)) { return 0.0; }
  return a[((n * p.co + co) * p.oh + u32(yy)) * p.ow + u32(xx)];
}
fn bval(k: u32, nn: u32) -> f32 { let kk = p.ks * p.ks; return b[((k / kk) * p.ci + nn) * kk + k % kk]; }
fn store(m: u32, nn: u32, v: f32, z: u32) { let ihw = p.ih * p.iw; out[((m / ihw) * p.ci + nn) * ihw + m % ihw] = v; }
"""
_GINP = """
struct PH { py: u32, px: u32, ky0: u32, kx0: u32, qy: u32, qx: u32, hi: u32, wi: u32, jy: u32, jx: u32, p0: u32, p1: u32 };
@group(0) @binding(4) var<uniform> ph: PH;
fn aval(m: u32, k: u32) -> f32 {
  let hw = ph.hi * ph.wi; let n = m / hw; let r = m % hw; let i = r / ph.wi; let j = r % ph.wi;
  let jj = ph.jy * ph.jx; let co = k / jj; let kr = k % jj;
  let oy = i32(i + ph.qy) - i32(kr / ph.jx); let ox = i32(j + ph.qx) - i32(kr % ph.jx);
  if (oy < 0 || oy >= i32(p.oh) || ox < 0 || ox >= i32(p.ow)) { return 0.0; }
  return a[((n * p.co + co) * p.oh + u32(oy)) * p.ow + u32(ox)];
}
fn bval(k: u32, nn: u32) -> f32 {
  let jj = ph.jy * ph.jx; let co = k / jj; let kr = k % jj;
  let ky = ph.ky0 + p.stride * (kr / ph.jx); let kx = ph.kx0 + p.stride * (kr % ph.jx);
  return b[((co * p.ci + nn) * p.ks + ky) * p.ks + kx];
}
fn store(m: u32, nn: u32, v: f32, z: u32) {
  let hw = ph.hi * ph.wi; let n = m / hw; let r = m % hw;
  let y = ph.py + p.stride * (r / ph.wi); let x = ph.px + p.stride * (r % ph.wi);
  out[((n * p.ci + nn) * p.ih + y) * p.iw + x] = v;
}
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
    body = {4: _MM, 5: _GINP, 1: _GIN, 2: _GW}.get(kind, _FWD)
    if kind == 3:  # eval plans: no bias (BN folding), the BN parameters are bound at 4..7 instead
        body = body.replace("  if ((p.hb & 1u) != 0u) { r += bias[nn]; }   // hb bit 0: bias, bit 1: ReLU (fused epilogue)\n", "")
        body = body.replace("RETURN", "return max((v - bnm[c]) * inverseSqrt(bnv[c] + 1e-5) * bng[c] + bnb[c], 0.0);")
    else:
        body = body.replace("RETURN", "return v;")
    a_m_fast = (not a_kfast) if kind == 4 else kind != 2   # A: consecutive threads along m (kinds 0, 1, 5) or along k (kind 2)
    b_n_fast = b_nfast if kind == 4 else kind in (1, 5)    # B: along n for grad input, along k otherwise
    a_idx = "let mm = e % BM; let ka = e / BM;" if a_m_fast else "let ka = e % 16u; let mm = e / 16u;"
    b_idx = "let nn = e % BN; let kb = e / BN;" if b_n_fast else "let kb = e % 16u; let nn = e / 16u;"
    acc = [[f"c{i}_{j}" for j in range(tn // 4)] for i in range(tm)]
    decl = "\n".join(f"  var {c} = vec4<f32>();" for row in acc for c in row)
    # shared tiles are vec4 arrays: per kk a thread reads tm/4 + tn/4 vec4s for tm * tn multiply-adds
    inner = "\n".join(f"      let rb{j} = Bs[bb + {j}u];" for j in range(tn // 4))
    inner += "\n" + "\n".join(f"      let ra{i} = As[ab + {i}u];" for i in range(tm // 4))
    inner += "\n" + "\n".join(f"      {acc[i][j]} += ra{i // 4}.{'xyzw'[i % 4]} * rb{j};" for i in range(tm) for j in range(tn // 4))
    # loaders unrolled (trip counts are compile-time constants). Register-prefetching the next K-tile was tried and
    # halved throughput: the extra registers spill on this GPU.
    # Loaders unrolled (trip counts are compile-time constants). Two things that made it slower, because live
    # registers cost more than integer divisions on this GPU and the kernels start spilling: prefetching the next
    # K-tile into registers (-50%), and hoisting each slot's row decomposition out of the K loop with a K-side index
    # table (-15 to -20%).
    loads = []
    for r in range(BK * bm // nt):
        loads.append(f"""    {{ let e = t + {r * nt}u; {a_idx}
      let m = m0 + mm; let k = k0 + ka; var v = 0.0;
      if (m < p.M && k < kend) {{ v = aval(m, k); }}
      let ai = ka * BM + mm; As[ai / 4u][ai % 4u] = v; }}""")
    for r in range(BK * bn // nt):
        loads.append(f"""    {{ let e = t + {r * nt}u; {b_idx}
      let nn2 = n0 + nn; let k2 = k0 + kb; var u = 0.0;
      if (nn2 < p.N && k2 < kend) {{ u = bval(k2, nn2); }}
      let bi = kb * BN + nn; Bs[bi / 4u][bi % 4u] = u; }}""")
    loaders = "\n".join(loads)
    epi = []
    for i in range(tm):
        for j in range(tn // 4):
            for c, comp in enumerate("xyzw"):
                epi.append(f"  {{ let m = m0 + lid.y * {tm}u + {i}u; let nn = n0 + lid.x * {tn}u + {4 * j + c}u;\n"
                           f"    if (m < p.M && nn < p.N) {{ store(m, nn, {acc[i][j]}.{comp}, wg.z); }} }}")
    extra = BN_BINDINGS if kind == 3 else ("@group(0) @binding(4) var<storage, read> bias: array<f32>;" if kind in (0, 4) else "")
    return f"""{MM_PARAMS if kind == 4 else PARAMS}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
{extra}
const BM = {bm}u; const BN = {bn}u;
var<workgroup> As: array<vec4<f32>, {BK * bm // 4}>;
var<workgroup> Bs: array<vec4<f32>, {BK * bn // 4}>;
{body}
@compute @workgroup_size({bn // tn}, {bm // tm})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
  let t = lid.y * {bn // tn}u + lid.x;
  let m0 = wg.y * BM; let n0 = wg.x * BN;
  let kbeg = wg.z * p.kslice; let kend = min(p.K, kbeg + p.kslice);
{decl}
  for (var k0 = kbeg; k0 < kend; k0 += 16u) {{
{loaders}
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {{
      let ab = (kk * BM + lid.y * {tm}u) / 4u; let bb = (kk * BN + lid.x * {tn}u) / 4u;
{inner}
    }}
    workgroupBarrier();
  }}
{chr(10).join(epi)}
}}
"""
