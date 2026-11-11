// General strided elementwise kernel: out[io] = op(a[ia], b[ib]) over up to 8 dims (shapes and strides in 32-bit
// words, coalesced on the host). fa/fb = 1: that input aliases the output buffer, read it through `o`.
struct E { n: u32, op: u32, bs: u32, nd: u32,
           alpha: u32, sval: u32, oa: u32, ob: u32,
           oo: u32, fa: u32, fb: u32, p0: u32,
           sh: array<vec4<u32>, 2>, sa: array<vec4<u32>, 2>, sb: array<vec4<u32>, 2>, so: array<vec4<u32>, 2> };
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> o: array<u32>;
@group(0) @binding(3) var<uniform> e: E;

// erf, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)
fn erf_(x: f32) -> f32 {
  let s = sign(x); let z = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
  return s * y;
}
fn powf(x: f32, p: f32) -> f32 {
  if (p == 2.0) { return x * x; }
  if (p == 1.0) { return x; }
  if (p == 0.5) { return sqrt(x); }
  if (p == 0.0) { return 1.0; }
  let r = pow(abs(x), p);
  if (x >= 0.0) { return r; }
  if (p != floor(p)) { return bitcast<f32>(0x7fc00000u); }
  return select(r, -r, (i32(p) & 1) == 1);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = gid.x + gid.y * nwg.x * 256u;
  if (i >= e.n) { return; }
  var r = i; var ia = e.oa; var ib = e.ob; var io = e.oo; var clast = 0u;
  for (var d = i32(e.nd) - 1; d >= 0; d--) {
    let q = u32(d) / 4u; let k = u32(d) % 4u;
    let s = e.sh[q][k];
    let c = r % s; r = r / s;
    if (u32(d) == e.nd - 1u) { clast = c; }
    ia += c * e.sa[q][k]; ib += c * e.sb[q][k]; io += c * e.so[q][k];
  }
  if (e.op == 10u) { o[io] = e.sval; return; }                                   // fill (raw word)
  if (e.op == 31u) { o[io] = select(e.sval, e.alpha, (clast & 1u) == 1u); return; }  // fill 64-bit (lo, hi)
  var xu = 0u;
  if (e.fa == 1u) { xu = o[ia]; } else { xu = a[ia]; }
  if (e.op == 9u) { o[io] = xu; return; }                                        // copy (raw word)
  let x = bitcast<f32>(xu);
  var y = bitcast<f32>(e.sval);
  if (e.bs == 0u) { if (e.fb == 1u) { y = bitcast<f32>(o[ib]); } else { y = bitcast<f32>(b[ib]); } }
  let al = bitcast<f32>(e.alpha);
  var v = 0.0;
  switch (e.op) {
    case 0u: { v = x + al * y; }
    case 1u: { v = x - al * y; }
    case 2u: { v = x * y; }
    case 3u: { v = x / y; }
    case 4u: { v = -x; }
    case 5u: { v = max(x, 0.0); }
    case 6u: { v = exp(x); }
    case 7u: { v = sqrt(x); }
    case 8u: { v = select(0.0, x, y > al); }                       // threshold_backward(grad=x, self=y, threshold)
    case 11u: { v = al * x; }
    case 12u: { v = select(x * log(y), 0.0, x == 0.0); }           // xlogy
    case 13u: { v = 1.0 / (1.0 + exp(-x)); }                       // sigmoid
    case 14u: { v = x / (1.0 + exp(-x)); }                         // silu
    case 15u: { v = x * clamp(x + 3.0, 0.0, 6.0) / 6.0; }          // hardswish
    case 16u: { v = x * y * (1.0 - y); }                           // sigmoid_backward(grad, output)
    case 17u: { let s = 1.0 / (1.0 + exp(-y)); v = x * s * (1.0 + y * (1.0 - s)); }   // silu_backward(grad, self)
    case 18u: { v = select(select(x * (2.0 * y + 3.0) / 6.0, x, y > 3.0), 0.0, y < -3.0); }  // hardswish_backward
    case 19u: { v = clamp(x + 3.0, 0.0, 6.0) / 6.0; }              // hardsigmoid
    case 20u: { v = select(0.0, x / 6.0, y > -3.0 && y < 3.0); }   // hardsigmoid_backward(grad, self)
    case 21u: { v = tanh(x); }
    case 22u: { v = x * (1.0 - y * y); }                           // tanh_backward(grad, output)
    case 23u: { v = min(max(x, al), y); }                          // clamp(x, lo=alpha, hi=sval)
    case 24u: { v = log(x); }
    case 25u: { v = abs(x); }
    case 26u: { v = inverseSqrt(x); }
    case 27u: { v = powf(x, al); }
    case 28u: { v = max(x, y); }
    case 29u: { v = min(x, y); }
    case 30u: { v = 1.0 / x; }
    case 32u: {                                                    // gelu (alpha = 1: tanh approximation)
      if (al == 1.0) { v = 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x * x * x))); }
      else { v = 0.5 * x * (1.0 + erf_(x * 0.7071067812)); }
    }
    case 33u: {                                                    // gelu_backward(grad=x, self=y)
      if (al == 1.0) {
        let k = 0.7978845608 * (y + 0.044715 * y * y * y); let t = tanh(k);
        v = x * (0.5 * (1.0 + t) + 0.5 * y * (1.0 - t * t) * 0.7978845608 * (1.0 + 0.134145 * y * y));
      } else {
        v = x * (0.5 * (1.0 + erf_(y * 0.7071067812)) + y * 0.3989422804 * exp(-0.5 * y * y));
      }
    }
    case 34u: { v = sin(x); }
    case 35u: { v = cos(x); }
    case 36u: { v = x * x; }
    case 37u: { v = x + al * (y - x); }                            // lerp(x, y, weight=alpha)
    case 38u: { v = y - al * x; }                                  // rsub(x, other=y, alpha)
    case 39u: { v = x + al * y * y; }                              // x + alpha * y^2 (addcmul(x, y, y))
    case 40u: { v = inverseSqrt(x + al); }                         // 1 / sqrt(x + eps)
    default: { v = x; }
  }
  o[io] = bitcast<u32>(v);
}
