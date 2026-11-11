// Depthwise conv weight gradient, any kernel size / stride / padding: one workgroup per (channel, ky, kx).
struct D { n: u32, c: u32, ih: u32, iw: u32, oh: u32, ow: u32, ks: u32, stride: u32, pad: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> gw: array<f32>;
@group(0) @binding(3) var<uniform> q: D;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let kk = q.ks * q.ks; let j = wg.x; let ch = j / kk; let k = j % kk;
  let ky = i32(k / q.ks) - i32(q.pad); let kx = i32(k % q.ks) - i32(q.pad);
  let ohw = q.oh * q.ow; let m = q.n * ohw;
  var s = 0.0;
  for (var e = t; e < m; e += 256u) {
    let n = e / ohw; let r = e % ohw;
    let yy = i32((r / q.ow) * q.stride) + ky; let xx = i32((r % q.ow) * q.stride) + kx;
    if (yy >= 0 && yy < i32(q.ih) && xx >= 0 && xx < i32(q.iw)) {
      s += g[(n * q.c + ch) * ohw + r] * x[((n * q.c + ch) * q.ih + u32(yy)) * q.iw + u32(xx)];
    }
  }
  red[t] = s;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h = h / 2u) {
    if (t < h) { red[t] += red[t + h]; }
    workgroupBarrier();
  }
  if (t == 0u) { gw[j] = red[0]; }
}
