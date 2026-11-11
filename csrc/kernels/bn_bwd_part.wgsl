struct Q { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> y: array<f32>;
@group(0) @binding(4) var<storage, read_write> part: array<f32>;
@group(0) @binding(5) var<uniform> q: Q;

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
  let ch = wg.x; let sl = wg.y; let hw = q.h * q.w; let ns = q.p0;
  let mean = stats[ch]; let inv = select(inverseSqrt(stats[q.c + ch] + bitcast<f32>(q.p1)), stats[q.c + ch], q.p1 == 0xffffffffu);
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
