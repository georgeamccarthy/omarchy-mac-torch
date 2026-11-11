// out = mask ? g * scale : 0, with a bool (byte) mask
struct S { n: u32, scale: f32, go: u32, mo: u32 };
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> m: array<u32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.n) { return; }
  let j = s.mo + i;
  let keep = ((m[j / 4u] >> (8u * (j % 4u))) & 255u) != 0u;
  o[i] = select(0.0, g[s.go + i] * s.scale, keep);
}
