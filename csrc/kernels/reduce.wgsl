struct R { outer: u32, red: u32, inner: u32, xo: u32, oo: u32, scale: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> r: R;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 64u;
  if (i >= r.outer * r.inner) { return; }
  let ou = i / r.inner; let in = i % r.inner;
  var s = 0.0;
  for (var k = 0u; k < r.red; k++) { s += x[r.xo + (ou * r.red + k) * r.inner + in]; }
  o[r.oo + i] = s * r.scale;
}
