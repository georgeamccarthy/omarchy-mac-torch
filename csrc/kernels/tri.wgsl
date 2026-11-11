// tril / triu over the last two dims (H x W) of a contiguous tensor: out = keep ? x : 0, keep = (col - row <= k) for
// tril (mode 0), (col - row >= k) for triu (mode 1). es = element bytes: 4 (one word each) or 1 (4 per word, packed).
struct S { n: u32, h: u32, w: u32, k: i32, mode: u32, es: u32, xo: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> o: array<u32>;
@group(0) @binding(2) var<uniform> s: S;
fn keep(i: u32) -> bool {
  let d = i32(i % s.w) - i32((i / s.w) % s.h);
  return select(d >= s.k, d <= s.k, s.mode == 0u);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let t = gid.x + gid.y * nwg.x * 256u;
  if (s.es == 4u) {
    if (t >= s.n) { return; }
    o[t] = select(0u, x[s.xo + t], keep(t));
    return;
  }
  if (t * 4u >= s.n) { return; }
  var word = 0u;
  for (var q = 0u; q < 4u; q++) {
    let i = t * 4u + q;
    if (i >= s.n) { break; }
    let j = s.xo + i;
    let b = (x[j / 4u] >> (8u * (j % 4u))) & 255u;
    word = word | (select(0u, b, keep(i)) << (8u * q));
  }
  o[t] = word;
}
