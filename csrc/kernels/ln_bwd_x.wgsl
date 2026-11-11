struct S { R: u32, D: u32, eps: f32, aff: u32 };
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> rstd: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;
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
