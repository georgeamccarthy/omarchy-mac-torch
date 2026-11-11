// Depthwise conv forward, any kernel size / stride / padding (one thread per output element).
struct D { n: u32, c: u32, ih: u32, iw: u32, oh: u32, ow: u32, ks: u32, stride: u32, pad: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> q: D;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= q.n * q.c * q.oh * q.ow) { return; }
  let ox = i % q.ow; let oy = (i / q.ow) % q.oh; let nc = i / (q.oh * q.ow); let ch = nc % q.c;
  let y0 = i32(oy * q.stride) - i32(q.pad); let x0 = i32(ox * q.stride) - i32(q.pad);
  var s = 0.0;
  for (var ky = 0u; ky < q.ks; ky++) {
    let sy = y0 + i32(ky);
    if (sy < 0 || sy >= i32(q.ih)) { continue; }
    for (var kx = 0u; kx < q.ks; kx++) {
      let sx = x0 + i32(kx);
      if (sx < 0 || sx >= i32(q.iw)) { continue; }
      s += x[(nc * q.ih + u32(sy)) * q.iw + u32(sx)] * wt[(ch * q.ks + ky) * q.ks + kx];
    }
  }
  out[i] = s;
}
