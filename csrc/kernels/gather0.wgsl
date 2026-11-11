// out[r, :] = x[idx[r], :] along dim 0. Rows are `row` 32-bit words (int64 rows are twice as long);
// idx is int32 (iw = 1) or int64 (iw = 2, low word); negative indices wrap.
struct S { n: u32, row: u32, xo: u32, io: u32, iw: u32, rows: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> o: array<u32>;
@group(0) @binding(3) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.n * s.row) { return; }
  let r = i / s.row; let c = i % s.row;
  var k = idx[(s.io + r) * s.iw];
  if (k < 0) { k += i32(s.rows); }
  o[i] = x[s.xo + u32(k) * s.row + c];
}
