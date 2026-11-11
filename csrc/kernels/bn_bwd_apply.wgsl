struct Q { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> stats: array<f32>;
@group(0) @binding(3) var<storage, read> y: array<f32>;
@group(0) @binding(4) var<storage, read> gamma: array<f32>;
@group(0) @binding(5) var<storage, read> dgb: array<f32>;
@group(0) @binding(6) var<storage, read_write> dx: array<f32>;
@group(0) @binding(7) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {

  let i = gid.x + gid.y * nwg.x * 256u;
  let hw = q.h * q.w;

  if (i >= q.n * q.c * hw) { return; }
  let ch = (i / hw) % q.c;
  let m = f32(q.n * hw);
  let inv = inverseSqrt(stats[q.c + ch] + 1e-5);
  let gz = select(0.0, g[i], y[i] > 0.0);
  let xhat = (x[i] - stats[ch]) * inv;
  dx[i] = gamma[ch] * inv / m * (m * gz - dgb[q.c + ch] - xhat * dgb[ch]);
}
