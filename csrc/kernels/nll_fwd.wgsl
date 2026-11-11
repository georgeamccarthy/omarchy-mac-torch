// nll_loss forward for log-probabilities (n, c) and int64 targets: per-row loss and weight (0 for ignore_index).
// The host reduces (sum, or sum / total weight for "mean").
struct S { n: u32, c: u32, ignore: i32, hw: u32, xo: u32, to: u32, wo: u32, p: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> tg: array<i32>;
@group(0) @binding(2) var<storage, read> wt: array<f32>;
@group(0) @binding(3) var<storage, read_write> loss: array<f32>;
@group(0) @binding(4) var<storage, read_write> w: array<f32>;
@group(0) @binding(5) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let r = gid.x + gid.y * nwg.x * 256u;
  if (r >= s.n) { return; }
  let y = tg[2u * (s.to + r)];
  if (y == s.ignore || y < 0 || u32(y) >= s.c) { loss[r] = 0.0; w[r] = 0.0; return; }
  var wy = 1.0;
  if (s.hw == 1u) { wy = wt[s.wo + u32(y)]; }
  loss[r] = -wy * x[s.xo + r * s.c + u32(y)];
  w[r] = wy;
}
