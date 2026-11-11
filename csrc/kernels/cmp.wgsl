// Comparisons with broadcasting: out (bool, 1 byte per element, packed 4 per word) = a <op> b.
// kind: 0 f32, 1 i32, 2 i64, 3 bool/uint8 (byte); strides and offsets in elements.
struct E { n: u32, op: u32, bs: u32, nd: u32,
           kind: u32, slo: u32, shi: u32, oa: u32,
           ob: u32, p0: u32, p1: u32, p2: u32,
           sh: array<vec4<u32>, 2>, sa: array<vec4<u32>, 2>, sb: array<vec4<u32>, 2> };
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> o: array<u32>;
@group(0) @binding(3) var<uniform> e: E;
// WGSL lets the compiler assume NaN never occurs, so `x != x` may fold to false: test the bits instead
fn isnan_(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
fn cmp3(x: i32, y: i32) -> i32 { return select(select(0, 1, x > y), -1, x < y); }
// -1, 0, 1 comparing element ix of buffer `which` (0 = a, 1 = b) with element iy
fn load_lo(which: u32, ix: u32) -> u32 { if (which == 0u) { return a[ix]; } return b[ix]; }
fn order(xi: u32, yi: u32, yscalar: bool) -> i32 {
  switch (e.kind) {
    case 0u: {
      let x = bitcast<f32>(a[xi]); var y = bitcast<f32>(e.slo);
      if (!yscalar) { y = bitcast<f32>(b[yi]); }
      if (isnan_(x) || isnan_(y)) { return 2; }   // NaN: unordered
      return select(select(0, 1, x > y), -1, x < y);
    }
    case 1u: {
      var y = bitcast<i32>(e.slo); if (!yscalar) { y = bitcast<i32>(b[yi]); }
      return cmp3(bitcast<i32>(a[xi]), y);
    }
    case 2u: {
      var ylo = e.slo; var yhi = e.shi;
      if (!yscalar) { ylo = b[2u * yi]; yhi = b[2u * yi + 1u]; }
      let h = cmp3(bitcast<i32>(a[2u * xi + 1u]), bitcast<i32>(yhi));
      if (h != 0) { return h; }
      let xlo = a[2u * xi];
      return select(select(0, 1, xlo > ylo), -1, xlo < ylo);
    }
    default: {
      let x = i32((a[xi / 4u] >> (8u * (xi % 4u))) & 255u);
      var y = i32(e.slo);
      if (!yscalar) { y = i32((b[yi / 4u] >> (8u * (yi % 4u))) & 255u); }
      return cmp3(x, y);
    }
  }
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let w = gid.x + gid.y * nwg.x * 256u;
  if (w * 4u >= e.n) { return; }
  var word = 0u;
  for (var j = 0u; j < 4u; j++) {
    let i = w * 4u + j;
    if (i >= e.n) { break; }
    var r = i; var ia = e.oa; var ib = e.ob;
    for (var d = i32(e.nd) - 1; d >= 0; d--) {
      let q = u32(d) / 4u; let k = u32(d) % 4u;
      let s = e.sh[q][k];
      let c = r % s; r = r / s;
      ia += c * e.sa[q][k]; ib += c * e.sb[q][k];
    }
    var t = false;
    if (e.op >= 6u) {  // unary float classification (kind 0): isnan, isinf, isneginf, isposinf
      let bits = a[ia] & 0x7fffffffu; let neg = (a[ia] >> 31u) == 1u;
      switch (e.op) {
        case 6u: { t = bits > 0x7f800000u; }
        case 7u: { t = bits == 0x7f800000u; }
        case 8u: { t = bits == 0x7f800000u && neg; }
        default: { t = bits == 0x7f800000u && !neg; }
      }
      word = word | (select(0u, 1u, t) << (8u * j));
      continue;
    }
    let c = order(ia, ib, e.bs == 1u);
    switch (e.op) {
      case 0u: { t = c == -1; }            // lt
      case 1u: { t = c == -1 || c == 0; }  // le
      case 2u: { t = c == 1; }             // gt
      case 3u: { t = c == 1 || c == 0; }   // ge
      case 4u: { t = c != 0; }             // ne (NaN != anything)
      default: { t = c == 0; }             // eq
    }
    word = word | (select(0u, 1u, t) << (8u * j));
  }
  o[w] = word;
}
