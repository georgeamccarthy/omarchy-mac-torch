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
