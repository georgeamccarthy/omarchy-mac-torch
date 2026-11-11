struct Q { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= q.n * q.c * q.oh * q.ow) { return; }
  let ox = i % q.ow; let oy = (i / q.ow) % q.oh; let nc = i / (q.ow * q.oh);
  let b = nc * q.h * q.w + (oy * 2u) * q.w + ox * 2u;
  out[i] = 0.25 * (x[b] + x[b + 1u] + x[b + q.w] + x[b + q.w + 1u]);
}
