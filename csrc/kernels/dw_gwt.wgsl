struct Q { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> gw: array<f32>;
@group(0) @binding(3) var<uniform> q: Q;

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
