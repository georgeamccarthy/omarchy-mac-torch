// Fused Adam / AdamW for one parameter. The step count is read from the parameter's own (device) step tensor, so
// the bias corrections are computed here and stepping never waits for the GPU.
struct H { lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, decoupled: u32, maximize: u32, p: u32 };
struct A { count: u32, po: u32, go: u32, mo: u32, vo: u32, so: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read_write> p: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<storage, read> step: array<f32>;
@group(0) @binding(5) var<uniform> a: A;
@group(0) @binding(6) var<uniform> h: H;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= a.count) { return; }
  let t = step[a.so];
  let bc1 = 1.0 - pow(h.b1, t);
  let sbc2 = sqrt(1.0 - pow(h.b2, t));
  var gr = g[a.go + i];
  if (h.maximize == 1u) { gr = -gr; }
  var pv = p[a.po + i];
  if (h.decoupled == 1u) { pv = pv * (1.0 - h.lr * h.wd); } else { gr = gr + h.wd * pv; }
  let mi = m[a.mo + i] + (1.0 - h.b1) * (gr - m[a.mo + i]);
  let vi = h.b2 * v[a.vo + i] + (1.0 - h.b2) * gr * gr;
  m[a.mo + i] = mi; v[a.vo + i] = vi;
  p[a.po + i] = pv - (h.lr / bc1) * mi / (sqrt(vi) / sbc2 + h.eps);
}
