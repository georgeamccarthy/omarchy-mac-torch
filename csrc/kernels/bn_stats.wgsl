struct Q { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> stats: array<f32>;
@group(0) @binding(2) var<uniform> q: Q;

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
