struct Q { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> dgb: array<f32>;
@group(0) @binding(2) var<uniform> q: Q;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x;
  if (ch >= q.c) { return; }
  let ns = q.p0;
  var s1 = 0.0; var s2 = 0.0;
  for (var k = 0u; k < ns; k++) { s1 += part[(ch * ns + k) * 2u]; s2 += part[(ch * ns + k) * 2u + 1u]; }
  dgb[ch] = s2; dgb[q.c + ch] = s1;
}
