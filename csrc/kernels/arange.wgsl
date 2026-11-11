// out[i] = start + i * step: f32 (wide = 0) or int64 (wide = 1, computed in i32 and sign-extended)
struct S { n: u32, wide: u32, oo: u32, p0: u32, start: f32, step: f32, istart: i32, istep: i32 };
@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@group(0) @binding(1) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.n) { return; }
  if (s.wide == 0u) { o[s.oo + i] = bitcast<u32>(s.start + f32(i) * s.step); return; }
  let v = s.istart + i32(i) * s.istep;
  o[2u * (s.oo + i)] = bitcast<u32>(v);
  o[2u * (s.oo + i) + 1u] = select(0u, 0xffffffffu, v < 0);
}
