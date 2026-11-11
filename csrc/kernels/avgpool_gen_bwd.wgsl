// Average pool 2d backward (same divisor rules as the forward).
struct S { nc: u32, ih: u32, iw: u32, oh: u32, ow: u32, kh: u32, kw: u32, sh: u32,
           sw: u32, ph: u32, pw: u32, dh: u32, dw: u32, mode: u32, div: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read_write> gx: array<f32>;
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
  if (i >= s.nc * s.ih * s.iw) { return; }
  let xx = i32(i % s.iw); let y = i32((i / s.iw) % s.ih); let nc = i / (s.iw * s.ih);
  let ylo = max(0, (y + i32(s.ph) - i32(s.kh) + i32(s.sh)) / i32(s.sh));
  let yhi = min(i32(s.oh) - 1, (y + i32(s.ph)) / i32(s.sh));
  let xlo = max(0, (xx + i32(s.pw) - i32(s.kw) + i32(s.sw)) / i32(s.sw));
  let xhi = min(i32(s.ow) - 1, (xx + i32(s.pw)) / i32(s.sw));
  var acc = 0.0;
  for (var oy = ylo; oy <= yhi; oy++) {
    for (var ox = xlo; ox <= xhi; ox++) {
      acc += g[(nc * s.oh + u32(oy)) * s.ow + u32(ox)] / divisor(u32(oy), u32(ox));
    }
  }
  gx[i] = acc;
}
