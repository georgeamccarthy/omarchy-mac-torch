// Max pool 2d, any window / stride / padding / dilation (ceil_mode is in the output size). Indices are int64 flat
// positions in the input plane, as PyTorch returns them.
struct S { nc: u32, ih: u32, iw: u32, oh: u32, ow: u32, kh: u32, kw: u32, sh: u32,
           sw: u32, ph: u32, pw: u32, dh: u32, dw: u32, mode: u32, div: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> idx: array<i32>;
@group(0) @binding(3) var<uniform> s: S;
// WGSL lets the compiler assume NaN never occurs, so `x != x` may fold to false: test the bits instead
fn isnan_(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.nc * s.oh * s.ow) { return; }
  let ox = i % s.ow; let oy = (i / s.ow) % s.oh; let nc = i / (s.ow * s.oh);
  var best = -3.402823e+38; var bi = -1;
  for (var ky = 0u; ky < s.kh; ky++) {
    let y = i32(oy * s.sh + ky * s.dh) - i32(s.ph);
    if (y < 0 || y >= i32(s.ih)) { continue; }
    for (var kx = 0u; kx < s.kw; kx++) {
      let xx = i32(ox * s.sw + kx * s.dw) - i32(s.pw);
      if (xx < 0 || xx >= i32(s.iw)) { continue; }
      let pos = y * i32(s.iw) + xx;
      let v = x[nc * s.ih * s.iw + u32(pos)];
      if (bi < 0 || isnan_(v) || v > best) { best = v; bi = pos; if (isnan_(v)) { break; } }
    }
  }
  out[i] = best; idx[2u * i] = bi; idx[2u * i + 1u] = select(0, -1, bi < 0);
}
