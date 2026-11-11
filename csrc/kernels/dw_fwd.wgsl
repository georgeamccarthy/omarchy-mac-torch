struct Q { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {

  let i = gid.x + gid.y * nwg.x * 256u;
  let hw = q.h * q.w;

  if (i >= q.n * q.c * hw) { return; }
  let xx = i32(i % q.w); let yy = i32((i / q.w) % q.h); let nc = i / hw; let ch = nc % q.c;
  var s = 0.0;
  let ks = i32(q.p0); let pad = ks / 2;
  for (var ky = 0; ky < ks; ky++) {
    let sy = yy + ky - pad;
    if (sy < 0 || sy >= i32(q.h)) { continue; }
    for (var kx = 0; kx < ks; kx++) {
      let sx = xx + kx - pad;
      if (sx < 0 || sx >= i32(q.w)) { continue; }
      s += x[nc * hw + u32(sy) * q.w + u32(sx)] * wt[ch * q.p0 * q.p0 + u32(ky * ks + kx)];
    }
  }
  out[i] = s;
}
