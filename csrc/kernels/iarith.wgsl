// int32 / int64 add and subtract on contiguous tensors (int64 as little-endian (lo, hi) word pairs, with carry).
struct S { n: u32, op: u32, wide: u32, bs: u32, ao: u32, bo: u32, oo: u32, fa: u32, slo: u32, shi: u32, fb: u32, p: u32 };
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> o: array<u32>;
@group(0) @binding(3) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.n) { return; }
  let w = select(1u, 2u, s.wide == 1u);
  let ia = s.ao + i * w; let ib = s.bo + i * w; let io = s.oo + i * w;
  var alo = 0u; var ahi = 0u; var blo = s.slo; var bhi = s.shi;
  if (s.fa == 1u) { alo = o[ia]; if (w == 2u) { ahi = o[ia + 1u]; } } else { alo = a[ia]; if (w == 2u) { ahi = a[ia + 1u]; } }
  if (s.bs == 0u) {
    if (s.fb == 1u) { blo = o[ib]; if (w == 2u) { bhi = o[ib + 1u]; } } else { blo = b[ib]; if (w == 2u) { bhi = b[ib + 1u]; } }
  }
  var lo = 0u; var hi = 0u;
  if (s.op == 0u) { lo = alo + blo; hi = ahi + bhi + select(0u, 1u, lo < alo); }
  else { lo = alo - blo; hi = ahi - bhi - select(0u, 1u, alo < blo); }
  o[io] = lo;
  if (w == 2u) { o[io + 1u] = hi; }
}
