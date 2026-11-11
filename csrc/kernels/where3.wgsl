// out = cond ? a : b with broadcasting (cond bool bytes; a, b f32 or scalars); strides/offsets in elements.
// fa/fb = 1: that input aliases the output buffer (masked_fill_), read it through `o`.
struct E { n: u32, nd: u32, co: u32, oa: u32,
           ob: u32, as_: u32, bs: u32, av: u32,
           bv: u32, fa: u32, fb: u32, oo: u32,
           sh: array<vec4<u32>, 2>, sc: array<vec4<u32>, 2>, sa: array<vec4<u32>, 2>, sb: array<vec4<u32>, 2>,
           so: array<vec4<u32>, 2> };
@group(0) @binding(0) var<storage, read> c: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read_write> o: array<u32>;
@group(0) @binding(4) var<uniform> e: E;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= e.n) { return; }
  var r = i; var ic = e.co; var ia = e.oa; var ib = e.ob; var io = e.oo;
  for (var d = i32(e.nd) - 1; d >= 0; d--) {
    let q = u32(d) / 4u; let k = u32(d) % 4u;
    let s = e.sh[q][k];
    let cc = r % s; r = r / s;
    ic += cc * e.sc[q][k]; ia += cc * e.sa[q][k]; ib += cc * e.sb[q][k]; io += cc * e.so[q][k];
  }
  let cond = ((c[ic / 4u] >> (8u * (ic % 4u))) & 255u) != 0u;
  var v = 0u;
  if (cond) {
    if (e.as_ == 1u) { v = e.av; } else if (e.fa == 1u) { v = o[ia]; } else { v = a[ia]; }
  } else {
    if (e.bs == 1u) { v = e.bv; } else if (e.fb == 1u) { v = o[ib]; } else { v = b[ib]; }
  }
  o[io] = v;
}
