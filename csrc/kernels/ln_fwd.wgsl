struct S { R: u32, D: u32, eps: f32, aff: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<storage, read_write> mean: array<f32>;
@group(0) @binding(5) var<storage, read_write> rstd: array<f32>;
@group(0) @binding(6) var<uniform> s: S;

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
