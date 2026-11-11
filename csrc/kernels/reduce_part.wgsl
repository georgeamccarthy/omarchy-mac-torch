// First pass of a split reduction (few outputs, long reductions): workgroup (out, k) sums slice k of output `out`'s
// reduction into part[out * K + k]; reduce_wg then sums the K partials.
struct R { outer: u32, red: u32, inner: u32, xo: u32, chunk: u32, K: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> part: array<f32>;
@group(0) @binding(2) var<uniform> r: R;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let id = wg.x + wg.y * nwg.x;
  if (id >= r.outer * r.inner * r.K) { return; }
  let out = id / r.K; let k = id % r.K;
  let ou = out / r.inner; let inn = out % r.inner;
  let base = r.xo + ou * r.red * r.inner + inn;
  let lo = k * r.chunk; let hi = min(r.red, lo + r.chunk);
  var s = 0.0;
  for (var j = lo + t; j < hi; j += 256u) { s += x[base + j * r.inner]; }
  sh[t] = s;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h = h / 2u) {
    if (t < h) { sh[t] += sh[t + h]; }
    workgroupBarrier();
  }
  if (t == 0u) { part[id] = sh[0]; }
}
