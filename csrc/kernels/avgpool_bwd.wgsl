struct Q { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read_write> gx: array<f32>;
@group(0) @binding(2) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= q.n * q.c * q.h * q.w) { return; }
  let x = i % q.w; let y = (i / q.w) % q.h; let nc = i / (q.w * q.h);
  if (x / 2u >= q.ow || y / 2u >= q.oh) { gx[i] = 0.0; return; }
  gx[i] = 0.25 * g[nc * q.oh * q.ow + (y / 2u) * q.ow + x / 2u];
}
