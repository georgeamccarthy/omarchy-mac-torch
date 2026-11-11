struct P { M: u32, N: u32, K: u32, hb: u32, ao: u32, bo: u32, co: u32, bio: u32,
           sa0: u32, sa1: u32, sb0: u32, sb1: u32, kslice: u32, slices: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  let total = p.M * p.N;
  if (i >= total) { return; }
  var s = 0.0;
  for (var z = 0u; z < p.slices; z++) { s += part[z * total + i]; }
  if (p.hb == 1u) { s += bias[p.bio + i % p.N]; }
  out[p.co + i] = s;
}
