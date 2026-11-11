// softmax / log_softmax along the middle dim of a contiguous tensor viewed as (outer, dim, inner): one workgroup per
// (outer, inner) row, max and sum by shared-memory trees. From TensorFlow.js's WebGPU softmax kernel (Apache-2.0),
// generalised to any dim and given a log mode.
struct S { outer: u32, dim: u32, inner: u32, mode: u32, xo: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<uniform> s: S;
var<workgroup> buf: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let row = wg.x + wg.y * nwg.x;
  if (row >= s.outer * s.inner) { return; }
  let base = (row / s.inner) * s.dim * s.inner + row % s.inner;
  var m = -3.402823e+38;
  for (var k = t; k < s.dim; k += 256u) { m = max(m, x[s.xo + base + k * s.inner]); }
  buf[t] = m;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h = h / 2u) {
    if (t < h) { buf[t] = max(buf[t], buf[t + h]); }
    workgroupBarrier();
  }
  let mx = buf[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var k = t; k < s.dim; k += 256u) { sum += exp(x[s.xo + base + k * s.inner] - mx); }
  buf[t] = sum;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h = h / 2u) {
    if (t < h) { buf[t] += buf[t + h]; }
    workgroupBarrier();
  }
  let lse = log(buf[0]) + mx;
  for (var k = t; k < s.dim; k += 256u) {
    let lp = x[s.xo + base + k * s.inner] - lse;
    o[base + k * s.inner] = select(exp(lp), lp, s.mode == 1u);
  }
}
