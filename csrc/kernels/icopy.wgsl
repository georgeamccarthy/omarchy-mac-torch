// copy n 32-bit words (any dtype): o[out offset + i] = a[a offset + i]
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@group(0) @binding(2) var<uniform> u: vec4<u32>;   // n, a offset, out offset, unused
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= u.x) { return; }
  o[u.z + i] = a[u.y + i];
}
