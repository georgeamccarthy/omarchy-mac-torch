struct Q { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> idx: array<u32>;
@group(0) @binding(3) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {

  let i = gid.x + gid.y * nwg.x * 256u;
  let hw = q.h * q.w;

  if (i >= q.n * q.c * q.oh * q.ow) { return; }
  let ox = i % q.ow; let oy = (i / q.ow) % q.oh; let nc = i / (q.oh * q.ow);
  let base = nc * hw + 2u * oy * q.w + 2u * ox;
  var best = x[base]; var bi = 0u;
  let v1 = x[base + 1u]; if (v1 > best) { best = v1; bi = 1u; }
  let v2 = x[base + q.w]; if (v2 > best) { best = v2; bi = 2u; }
  let v3 = x[base + q.w + 1u]; if (v3 > best) { best = v3; bi = 3u; }
  out[i] = best;
  idx[i] = bi;
}
