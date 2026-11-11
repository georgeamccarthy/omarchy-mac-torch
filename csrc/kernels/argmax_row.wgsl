// argmax / argmin along one dim of a contiguous f32 tensor viewed as (outer, red, inner); int64 output.
// One workgroup per output, tree reduction (ties resolve to the first index, NaN wins, as in PyTorch).
struct S { outer: u32, red: u32, inner: u32, xo: u32, mode: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@group(0) @binding(2) var<uniform> s: S;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
// WGSL lets the compiler assume NaN never occurs, so `x != x` may fold to false: test the bits instead
fn isnan_(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
fn better(v: f32, i: u32, w: f32, j: u32) -> bool {   // is (v, i) better than (w, j)?
  if (isnan_(v)) { return !isnan_(w) || i < j; }
  if (isnan_(w)) { return false; }
  if (s.mode == 1u) { return v < w || (v == w && i < j); }
  return v > w || (v == w && i < j);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let out = wg.x + wg.y * nwg.x;
  if (out >= s.outer * s.inner) { return; }
  let ou = out / s.inner; let inn = out % s.inner;
  let base = s.xo + ou * s.red * s.inner + inn;
  var best = x[base]; var besti = 0u;
  for (var k = t; k < s.red; k += 256u) {
    let v = x[base + k * s.inner];
    if (better(v, k, best, besti)) { best = v; besti = k; }
  }
  bv[t] = best; bi[t] = besti;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h = h / 2u) {
    if (t < h && better(bv[t + h], bi[t + h], bv[t], bi[t])) { bv[t] = bv[t + h]; bi[t] = bi[t + h]; }
    workgroupBarrier();
  }
  if (t == 0u) { o[2u * out] = bi[0]; o[2u * out + 1u] = 0u; }
}
