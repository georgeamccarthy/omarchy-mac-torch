// nll_loss backward: gx[r, y_r] = -w[y_r] * g * scale (scale = 1 / total weight for "mean"; per-row g for "none").
struct S { n: u32, c: u32, ignore: i32, hw: u32, to: u32, wo: u32, per_row: u32, go: u32 };
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> tg: array<i32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read> scale: array<f32>;   // [1]: 1 / total weight, or 1
@group(0) @binding(4) var<storage, read_write> gx: array<f32>;
@group(0) @binding(5) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.n * s.c) { return; }
  let r = i / s.c; let k = i % s.c;
  let y = tg[2u * (s.to + r)];
  var v = 0.0;
  if (y != s.ignore && y >= 0 && u32(y) == k) {
    var wy = 1.0;
    if (s.hw == 1u) { wy = wt[s.wo + k]; }
    var gg = g[s.go];
    if (s.per_row == 1u) { gg = g[s.go + r]; }
    v = -wy * gg * scale[0];
  }
  gx[i] = v;
}
