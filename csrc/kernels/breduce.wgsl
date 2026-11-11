// all / any over the middle dim of a contiguous tensor viewed as (outer, red, inner); bool output, 4 outputs per
// thread (packed bytes). kind: 0 f32 (nonzero; NaN counts as true), 1 i32, 2 i64, 3 bool/uint8. mode 0 all, 1 any.
struct S { outer: u32, red: u32, inner: u32, xo: u32, kind: u32, mode: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@group(0) @binding(2) var<uniform> s: S;
fn truthy(j: u32) -> bool {
  switch (s.kind) {
    case 0u: { return (x[j] & 0x7fffffffu) != 0u; }
    case 1u: { return x[j] != 0u; }
    case 2u: { return x[2u * j] != 0u || x[2u * j + 1u] != 0u; }
    default: { return ((x[j / 4u] >> (8u * (j % 4u))) & 255u) != 0u; }
  }
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let w = gid.x + gid.y * nwg.x * 256u;
  let total = s.outer * s.inner;
  if (w * 4u >= total) { return; }
  var word = 0u;
  for (var q = 0u; q < 4u; q++) {
    let out = w * 4u + q;
    if (out >= total) { break; }
    let base = s.xo + (out / s.inner) * s.red * s.inner + out % s.inner;
    var r = s.mode == 0u;
    for (var k = 0u; k < s.red; k++) {
      let t = truthy(base + k * s.inner);
      if (s.mode == 0u && !t) { r = false; break; }
      if (s.mode == 1u && t) { r = true; break; }
    }
    word = word | (select(0u, 1u, r) << (8u * q));
  }
  o[w] = word;
}
