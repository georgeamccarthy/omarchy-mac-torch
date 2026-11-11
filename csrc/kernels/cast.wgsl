// dtype conversion of a contiguous tensor, 4 elements per thread (so 1-byte outputs are written a word at a time).
// kinds: 0 f32, 1 i32, 2 i64, 3 bool, 4 uint8. Offsets in elements.
struct S { n: u32, sk: u32, dk: u32, so: u32, p0: u32, p1: u32, p2: u32, p3: u32 };
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@group(0) @binding(2) var<uniform> s: S;
fn load(j: u32) -> f32 {
  switch (s.sk) {
    case 0u: { return bitcast<f32>(x[j]); }
    case 1u: { return f32(bitcast<i32>(x[j])); }
    case 2u: {
      let lo = x[2u * j]; let hi = bitcast<i32>(x[2u * j + 1u]);
      if (hi == 0 && lo < 0x80000000u) { return f32(lo); }               // fits in i32: one rounding, like the CPU
      if (hi == -1 && lo >= 0x80000000u) { return f32(bitcast<i32>(lo)); }
      return f32(hi) * 4294967296.0 + f32(lo);
    }
    case 3u: { return select(0.0, 1.0, ((x[j / 4u] >> (8u * (j % 4u))) & 255u) != 0u); }
    default: { return f32((x[j / 4u] >> (8u * (j % 4u))) & 255u); }
  }
}
// integer source as (lo, hi) 32-bit words, for int -> int conversions without going through float (wraps like C)
fn load_int(j: u32) -> vec2<u32> {
  switch (s.sk) {
    case 1u: { let v = x[j]; return vec2<u32>(v, select(0u, 0xffffffffu, bitcast<i32>(v) < 0)); }
    case 2u: { return vec2<u32>(x[2u * j], x[2u * j + 1u]); }
    default: { return vec2<u32>((x[j / 4u] >> (8u * (j % 4u))) & 255u, 0u); }
  }
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let w = gid.x + gid.y * nwg.x * 256u;
  if (w * 4u >= s.n) { return; }
  var word = 0u;
  for (var q = 0u; q < 4u; q++) {
    let i = w * 4u + q;
    if (i >= s.n) { break; }
    if (s.sk != 0u && s.dk != 0u) {  // int / bool -> int / bool
      let iv = load_int(s.so + i);
      switch (s.dk) {
        case 1u: { o[i] = iv.x; }
        case 2u: { o[2u * i] = iv.x; o[2u * i + 1u] = iv.y; }
        case 3u: { word = word | (select(0u, 1u, (iv.x | iv.y) != 0u) << (8u * q)); }
        default: { word = word | ((iv.x & 255u) << (8u * q)); }
      }
      continue;
    }
    let v = load(s.so + i);
    switch (s.dk) {
      case 0u: { o[i] = bitcast<u32>(v); }
      case 1u: { o[i] = bitcast<u32>(i32(v)); }
      case 2u: { let t = i32(v); o[2u * i] = bitcast<u32>(t); o[2u * i + 1u] = select(0u, 0xffffffffu, t < 0); }
      case 3u: { word = word | (select(0u, 1u, v != 0.0) << (8u * q)); }
      default: { word = word | ((u32(i32(v)) & 255u) << (8u * q)); }
    }
  }
  if (s.dk >= 3u) { o[w] = word; }
}
