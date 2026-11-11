struct P { n: u32, ci: u32, co: u32, h: u32, w: u32, M: u32, N: u32, K: u32,
           kslice: u32, slices: u32, ks: u32, kk: u32 };
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let total = p.M * p.N;
  if (i >= total) { return; }
  var s = 0.0;
  for (var z = 0u; z < p.slices; z++) { s += part[z * total + i]; }
  out[i] = s;
}
