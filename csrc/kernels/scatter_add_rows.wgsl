// embedding backward: out[idx[r], :] += g[r, :] for every row r (int64 indices, rows equal to `pad` skipped).
// Float add by compare-and-swap on the bit pattern (WGSL has no float atomics); summation order is not deterministic.
struct S { n: u32, d: u32, pad: i32, go: u32, io: u32, nw: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> o: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.n * s.d) { return; }
  let r = i / s.d; let c = i % s.d;
  let k = idx[2u * (s.io + r)];
  if (k == s.pad || k < 0 || u32(k) >= s.nw) { return; }
  let v = g[s.go + i];
  let a = u32(k) * s.d + c;
  var old = atomicLoad(&o[a]);
  loop {
    let r2 = atomicCompareExchangeWeak(&o[a], old, bitcast<u32>(bitcast<f32>(old) + v));
    if (r2.exchanged) { break; }
    old = r2.old_value;
  }
}
