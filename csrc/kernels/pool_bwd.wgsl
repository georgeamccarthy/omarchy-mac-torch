struct Q { n: u32, c: u32, h: u32, w: u32, oh: u32, ow: u32, p0: u32, p1: u32 };

@group(0) @binding(0) var<storage, read> g: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> gx: array<f32>;
@group(0) @binding(3) var<uniform> q: Q;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {

  let i = gid.x + gid.y * nwg.x * 256u;
  let hw = q.h * q.w;

  if (i >= q.n * q.c * hw) { return; }
  let xx = i % q.w; let yy = (i / q.w) % q.h; let nc = i / hw;
  let oy = yy / 2u; let ox = xx / 2u;
  var v = 0.0;
  if (oy < q.oh && ox < q.ow) {
    let o = (nc * q.oh + oy) * q.ow + ox;
    if (idx[o] == (yy % 2u) * 2u + xx % 2u) { v = g[o]; }
  }
  gx[i] = v;
}
