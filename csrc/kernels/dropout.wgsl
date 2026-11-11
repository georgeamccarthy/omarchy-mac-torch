// Dropout: 4 elements per thread so the bool mask (1 byte per element) is written a whole word at a time.
struct S { n: u32, seed: u32, p: f32, scale: f32, xo: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<u32>;
@group(0) @binding(3) var<uniform> s: S;
fn hash(v: u32) -> u32 {
  var x = v * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  return (x >> 22u) ^ x;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let w = gid.x + gid.y * nwg.x * 256u;
  if (w * 4u >= s.n) { return; }
  let hs = hash(s.seed);
  var word = 0u;
  for (var j = 0u; j < 4u; j++) {
    let i = w * 4u + j;
    if (i >= s.n) { break; }
    let u = f32(hash(i ^ hs) >> 8u) * (1.0 / 16777216.0);
    let keep = u >= s.p;
    o[i] = select(0.0, x[s.xo + i] * s.scale, keep);
    word = word | (select(0u, 1u, keep) << (8u * j));
  }
  m[w] = word;
}
