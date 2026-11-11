struct H { lr: f32, mom: f32, damp: f32, wd: f32, nesterov: u32, first: u32, maximize: u32, p: u32 };
struct A { count: u32, po: u32, go: u32, mo: u32 };
@group(0) @binding(0) var<storage, read_write> p: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<uniform> a: A;
@group(0) @binding(4) var<uniform> h: H;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= a.count) { return; }
  var gr = g[a.go + i];
  if (h.maximize == 1u) { gr = -gr; }
  let pv = p[a.po + i];
  gr = gr + h.wd * pv;
  var d = gr;
  if (h.mom != 0.0) {
    var v = gr;
    if (h.first == 0u) { v = h.mom * m[a.mo + i] + (1.0 - h.damp) * gr; }
    m[a.mo + i] = v;
    d = select(v, gr + h.mom * v, h.nesterov == 1u);
  }
  p[a.po + i] = pv - h.lr * d;
}
