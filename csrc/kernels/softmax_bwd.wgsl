// Backward of softmax (mode 0: gx = y * (g - sum(g * y))) and log_softmax (mode 1: gx = g - exp(y) * sum(g)) along
// the middle dim of contiguous (outer, dim, inner) tensors; one workgroup per row.
struct S { outer: u32, dim: u32, inner: u32, mode: u32, p0: u32, p1: u32, p2: u32, p3: u32 };
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
@group(0) @binding(3) var<uniform> s: S;
var<workgroup> buf: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let row = wg.x + wg.y * nwg.x;
  if (row >= s.outer * s.inner) { return; }
  let base = (row / s.inner) * s.dim * s.inner + row % s.inner;
  var acc = 0.0;
  for (var k = t; k < s.dim; k += 256u) {
    let i = base + k * s.inner;
    acc += select(g[i] * y[i], g[i], s.mode == 1u);
  }
  buf[t] = acc;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h = h / 2u) {
    if (t < h) { buf[t] += buf[t + h]; }
    workgroupBarrier();
  }
  let tot = buf[0];
  for (var k = t; k < s.dim; k += 256u) {
    let i = base + k * s.inner;
    o[i] = select(y[i] * (g[i] - tot), g[i] - exp(y[i]) * tot, s.mode == 1u);
  }
}
