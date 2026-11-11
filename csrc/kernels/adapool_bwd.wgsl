struct S { n: u32, c: u32, ih: u32, iw: u32, oh: u32, ow: u32, k: u32, stride: u32, pad: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read_write> gx: array<f32>;
@group(0) @binding(2) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.n * s.c * s.ih * s.iw) { return; }
  let xx = i % s.iw; let y = (i / s.iw) % s.ih; let nc = i / (s.iw * s.ih);
  var acc = 0.0;
  for (var oy = 0u; oy < s.oh; oy++) {
    let y0 = (oy * s.ih) / s.oh; let y1 = ((oy + 1u) * s.ih + s.oh - 1u) / s.oh;
    if (y < y0 || y >= y1) { continue; }
    for (var ox = 0u; ox < s.ow; ox++) {
      let x0 = (ox * s.iw) / s.ow; let x1 = ((ox + 1u) * s.iw + s.ow - 1u) / s.ow;
      if (xx < x0 || xx >= x1) { continue; }
      acc += g[(nc * s.oh + oy) * s.ow + ox] / f32((y1 - y0) * (x1 - x0));
    }
  }
  gx[i] = acc;
}
