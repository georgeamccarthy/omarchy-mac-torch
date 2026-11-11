struct R { c: u32, mom: f32, corr: f32, eps: f32, upd: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> stats: array<f32>;
@group(0) @binding(1) var<storage, read_write> rm: array<f32>;
@group(0) @binding(2) var<storage, read_write> rv: array<f32>;
@group(0) @binding(3) var<storage, read_write> sm: array<f32>;
@group(0) @binding(4) var<storage, read_write> si: array<f32>;
@group(0) @binding(5) var<uniform> r: R;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= r.c) { return; }
  let mean = stats[i]; let v = stats[r.c + i];
  sm[i] = mean; si[i] = inverseSqrt(v + r.eps);
  if (r.upd == 1u) {
    rm[i] = rm[i] + r.mom * (mean - rm[i]);
    rv[i] = rv[i] + r.mom * (v * r.corr - rv[i]);
  }
}
