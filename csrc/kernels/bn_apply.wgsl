struct Q { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> mean: array<f32>;
@group(0) @binding(2) var<storage, read> inv: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {

  let i = gid.x + gid.y * nwg.x * 256u;
  let hw = q.h * q.w;

  if (i >= q.n * q.c * hw) { return; }
  let ch = (i / hw) % q.c;
  y[i] = (x[i] - mean[ch]) * inv[ch] * gamma[ch] + beta[ch];
}
