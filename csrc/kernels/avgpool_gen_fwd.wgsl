// Average pool 2d, any window / stride / padding, ceil_mode, count_include_pad (mode bit 0) and divisor_override
// (div > 0), with PyTorch's divisor rules (aten/src/ATen/native/cpu/AvgPoolKernel.cpp; MPS behaves the same).
struct S { nc: u32, ih: u32, iw: u32, oh: u32, ow: u32, kh: u32, kw: u32, sh: u32,
           sw: u32, ph: u32, pw: u32, dh: u32, dw: u32, mode: u32, div: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> s: S;
fn divisor(oy: u32, ox: u32) -> f32 {
  if (s.div > 0u) { return f32(s.div); }
  var hs = i32(oy * s.sh) - i32(s.ph); var ws = i32(ox * s.sw) - i32(s.pw);
  var he = min(hs + i32(s.kh), i32(s.ih + s.ph)); var we = min(ws + i32(s.kw), i32(s.iw + s.pw));
  let pool = (he - hs) * (we - ws);
  if ((s.mode & 1u) == 1u) { return f32(pool); }
  hs = max(hs, 0); ws = max(ws, 0); he = min(he, i32(s.ih)); we = min(we, i32(s.iw));
  return f32((he - hs) * (we - ws));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.nc * s.oh * s.ow) { return; }
  let ox = i % s.ow; let oy = (i / s.ow) % s.oh; let nc = i / (s.ow * s.oh);
  let hs = i32(oy * s.sh) - i32(s.ph); let ws = i32(ox * s.sw) - i32(s.pw);
  var acc = 0.0;
  for (var y = max(hs, 0); y < min(hs + i32(s.kh), i32(s.ih)); y++) {
    for (var xx = max(ws, 0); xx < min(ws + i32(s.kw), i32(s.iw)); xx++) {
      acc += x[(nc * s.ih + u32(y)) * s.iw + u32(xx)];
    }
  }
  out[i] = acc / divisor(oy, ox);
}
