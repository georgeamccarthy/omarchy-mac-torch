struct S { R: u32, D: u32, eps: f32, aff: u32 };
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> mean: array<f32>;
@group(0) @binding(3) var<storage, read> rstd: array<f32>;
@group(0) @binding(4) var<storage, read_write> dg: array<f32>;
@group(0) @binding(5) var<storage, read_write> db: array<f32>;
@group(0) @binding(6) var<uniform> s: S;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  if (d >= s.D) { return; }
  var a = 0.0; var b = 0.0;
  for (var r = 0u; r < s.R; r++) {
    let g = dy[r * s.D + d];
    a += g * (x[r * s.D + d] - mean[r]) * rstd[r];
    b += g;
  }
  dg[d] = a; db[d] = b;
}
