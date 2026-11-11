struct Q { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };

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
