struct S { n: u32, c: u32, ih: u32, iw: u32, oh: u32, ow: u32, k: u32, stride: u32, pad: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> s: S;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= s.n * s.c * s.oh * s.ow) { return; }
  let ox = i % s.ow; let oy = (i / s.ow) % s.oh; let nc = i / (s.ow * s.oh);
  let y0 = (oy * s.ih) / s.oh; let y1 = ((oy + 1u) * s.ih + s.oh - 1u) / s.oh;
  let x0 = (ox * s.iw) / s.ow; let x1 = ((ox + 1u) * s.iw + s.ow - 1u) / s.ow;
  var acc = 0.0;
  for (var y = y0; y < y1; y++) { for (var xx = x0; xx < x1; xx++) { acc += x[(nc * s.ih + y) * s.iw + xx]; } }
  out[i] = acc / f32((y1 - y0) * (x1 - x0));
}
