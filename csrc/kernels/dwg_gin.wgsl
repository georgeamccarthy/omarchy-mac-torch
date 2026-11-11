// Depthwise conv input gradient, any kernel size / stride / padding (one thread per input element).
struct D { n: u32, c: u32, ih: u32, iw: u32, oh: u32, ow: u32, ks: u32, stride: u32, pad: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> gx: array<f32>;
@group(0) @binding(3) var<uniform> q: D;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= q.n * q.c * q.ih * q.iw) { return; }
  let xx = i32(i % q.iw); let yy = i32((i / q.iw) % q.ih); let nc = i / (q.ih * q.iw); let ch = nc % q.c;
  let st = i32(q.stride);
  var s = 0.0;
  for (var ky = 0u; ky < q.ks; ky++) {
    let ty = yy + i32(q.pad) - i32(ky);
    if (ty < 0 || ty % st != 0 || ty / st >= i32(q.oh)) { continue; }
    for (var kx = 0u; kx < q.ks; kx++) {
      let tx = xx + i32(q.pad) - i32(kx);
      if (tx < 0 || tx % st != 0 || tx / st >= i32(q.ow)) { continue; }
      s += g[(nc * q.oh + u32(ty / st)) * q.ow + u32(tx / st)] * wt[(ch * q.ks + ky) * q.ks + kx];
    }
  }
  gx[i] = s;
}
