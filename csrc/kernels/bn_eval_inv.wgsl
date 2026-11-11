struct R { c: u32, eps: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> rv: array<f32>;
@group(0) @binding(1) var<storage, read_write> si: array<f32>;
@group(0) @binding(2) var<uniform> r: R;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < r.c) { si[gid.x] = inverseSqrt(rv[gid.x] + r.eps); }
}
