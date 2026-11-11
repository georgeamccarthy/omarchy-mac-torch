// Max pool 2d backward: each input element gathers the gradients of the outputs whose index points at it.
struct S { nc: u32, ih: u32, iw: u32, oh: u32, ow: u32, kh: u32, kw: u32, sh: u32,
           sw: u32, ph: u32, pw: u32, dh: u32, dw: u32, mode: u32, div: u32, p0: u32 };
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> gx: array<f32>;
@group(0) @binding(3) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.nc * s.ih * s.iw) { return; }
  let xx = i32(i % s.iw); let y = i32((i / s.iw) % s.ih); let nc = i / (s.iw * s.ih);
  // outputs whose window can contain (y, xx): oy*sh + ky*dh - ph == y for some ky
  let ylo = max(0, (y + i32(s.ph) - i32((s.kh - 1u) * s.dh) + i32(s.sh) - 1) / i32(s.sh));
  let yhi = min(i32(s.oh) - 1, (y + i32(s.ph)) / i32(s.sh));
  let xlo = max(0, (xx + i32(s.pw) - i32((s.kw - 1u) * s.dw) + i32(s.sw) - 1) / i32(s.sw));
  let xhi = min(i32(s.ow) - 1, (xx + i32(s.pw)) / i32(s.sw));
  let me = y * i32(s.iw) + xx;
  var acc = 0.0;
  for (var oy = ylo; oy <= yhi; oy++) {
    for (var ox = xlo; ox <= xhi; ox++) {
      let o = (nc * s.oh + u32(oy)) * s.ow + u32(ox);
      if (idx[2u * o] == me) { acc += g[o]; }
    }
  }
  gx[i] = acc;
}
