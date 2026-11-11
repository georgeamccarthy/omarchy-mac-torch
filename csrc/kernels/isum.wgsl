// Sum of an integer or bool tensor (contiguous) into one int64. kind: 1 i32, 2 i64 (low word), 3 bool/uint8 bytes.
struct S { n: u32, xo: u32, kind: u32, p: u32 };
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@group(0) @binding(2) var<uniform> s: S;
var<workgroup> part: array<i32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
  var acc = 0;
  for (var i = t; i < s.n; i += 256u) {
    let j = s.xo + i;
    switch (s.kind) {
      case 1u: { acc += bitcast<i32>(x[j]); }
      case 2u: { acc += bitcast<i32>(x[2u * j]); }
      default: { acc += i32((x[j / 4u] >> (8u * (j % 4u))) & 255u); }
    }
  }
  part[t] = acc;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h = h / 2u) {
    if (t < h) { part[t] += part[t + h]; }
    workgroupBarrier();
  }
  if (t == 0u) { o[0] = bitcast<u32>(part[0]); o[1] = select(0u, 0xffffffffu, part[0] < 0); }
}
