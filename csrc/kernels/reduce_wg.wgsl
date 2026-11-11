// Sum / mean over the middle dim of a contiguous f32 tensor viewed as (outer, red, inner): one workgroup per output,
// shared-memory tree reduction. Structure from TensorFlow.js's WebGPU reduce kernel (Apache-2.0).
struct R { outer: u32, red: u32, inner: u32, xo: u32, oo: u32, scale: f32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> r: R;
var<workgroup> part: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let out = wg.x + wg.y * nwg.x;
  if (out >= r.outer * r.inner) { return; }
  let ou = out / r.inner; let inn = out % r.inner;
  let base = r.xo + ou * r.red * r.inner + inn;
  var s = 0.0;
  for (var k = t; k < r.red; k += 256u) { s += x[base + k * r.inner]; }
  part[t] = s;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h = h / 2u) {
    if (t < h) { part[t] += part[t + h]; }
    workgroupBarrier();
  }
  if (t == 0u) { o[r.oo + out] = part[0] * r.scale; }
}
