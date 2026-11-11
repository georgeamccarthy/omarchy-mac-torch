// mm / addmm / bmm through the register-blocked GEMM template (gemm.py, kind 4), with split-K for skinny shapes.
#include "omt.h"

#include <pybind11/functional.h>

namespace omt {

static py::function g_mm_planner;  // (a_k_fast, b_n_fast, tile) -> (kernel, bm, bn)
struct MmKernel { std::string name; int bm, bn; };
static std::map<std::tuple<bool, bool, int>, MmKernel> g_mm_kernels;

static MmKernel mm_kernel(bool a_kfast, bool b_nfast, int tile) {
  auto key = std::make_tuple(a_kfast, b_nfast, tile);
  auto it = g_mm_kernels.find(key);
  if (it != g_mm_kernels.end()) return it->second;
  py::gil_scoped_acquire gil;
  py::tuple r = g_mm_planner(a_kfast, b_nfast, tile);
  MmKernel k{r[0].cast<std::string>(), r[1].cast<int>(), r[2].cast<int>()};
  g_mm_kernels[key] = k;
  return k;
}

// out (contiguous, word offset co) = a @ b (+ bias[n]); a, b any 2-D strides
static void mm_into(const Tensor& a, const Tensor& b, const Tensor* bias, const Tensor& out, int64_t co) {
  int64_t M = a.size(0), K = a.size(1), N = b.size(1);
  const Tensor& bt = bias ? *bias : a;
  int tile = M <= 32 ? 2 : (N <= 32 ? 1 : (M >= 256 && N >= 256 ? 5 : 0));
  MmKernel kn = mm_kernel(a.stride(1) == 1, b.stride(1) == 1, tile);
  int64_t tiles = cdiv(M, kn.bm) * cdiv(N, kn.bn);
  int64_t slices = std::max<int64_t>(1, std::min<int64_t>(192 / tiles, K / 128));
  int64_t kslice = cdiv(cdiv(K, slices), 16) * 16;
  slices = cdiv(K, kslice);
  uint32_t hb = bias ? 1 : 0, bio = bias ? u(bias->storage_offset()) : 0u;
  if (slices == 1) {
    Buf* pr = uniform({u(M), u(N), u(K), hb, u(a.storage_offset()), u(b.storage_offset()), u(co), bio,
                       u(a.stride(0)), u(a.stride(1)), u(b.stride(0)), u(b.stride(1)), u(kslice), 1, 0, 0});
    launch(kn.name.c_str(), {bind(a), bind(b), bind(out), bind_buf(pr), bind(bt)}, u(cdiv(N, kn.bn)), u(cdiv(M, kn.bm)), 1);
    return;
  }
  Tensor part = empty_dev({slices * M * N}, at::kFloat);
  Buf* pr = uniform({u(M), u(N), u(K), hb, u(a.storage_offset()), u(b.storage_offset()), 0, bio,
                     u(a.stride(0)), u(a.stride(1)), u(b.stride(0)), u(b.stride(1)), u(kslice), u(slices), 0, 0});
  launch(kn.name.c_str(), {bind(a), bind(b), bind(part), bind_buf(pr), bind(bt)}, u(cdiv(N, kn.bn)), u(cdiv(M, kn.bm)), u(slices));
  Buf* rr = uniform({u(M), u(N), 0, hb, 0, 0, u(co), bio, 0, 0, 0, 0, 0, u(slices), 0, 0});
  uint32_t gx, gy;
  grid(M * N, 256, gx, gy);
  launch("mm_reduce", {bind(part), bind(out), bind(bt), bind_buf(rr)}, gx, gy);
}

static Tensor mm_(const Tensor& a, const Tensor& b) {
  NEED(f32(a) && f32(b) && a.dim() == 2 && b.dim() == 2, "float32 2-D mm");
  Tensor out = empty_dev({a.size(0), b.size(1)}, at::kFloat);
  if (out.numel() == 0) return out;
  if (a.size(1) == 0) { fill_dev(out, 0.f); return out; }
  mm_into(a, b, nullptr, out, 0);
  return out;
}

static Tensor addmm_(const Tensor& bias, const Tensor& a, const Tensor& b, const Scalar& beta, const Scalar& alpha) {
  NEED(f32(a) && f32(b) && f32(bias) && a.dim() == 2 && b.dim() == 2, "float32 addmm");
  int64_t M = a.size(0), N = b.size(1);
  Tensor out = empty_dev({M, N}, at::kFloat);
  if (out.numel() == 0) return out;
  bool fused = beta.to<double>() == 1 && alpha.to<double>() == 1 && bias.dim() == 1 && bias.size(0) == N &&
               bias.stride(0) == 1 && a.size(1) > 0;
  if (fused) {  // nn.Linear: bias added in the GEMM epilogue
    mm_into(a, b, &bias, out, 0);
    return out;
  }
  if (a.size(1) == 0) fill_dev(out, 0.f);
  else mm_into(a, b, nullptr, out, 0);
  if (alpha.to<double>() != 1) ew_into(ew::MUL, &out, nullptr, out, 1.f, alpha.to<float>());
  if (beta.to<double>() != 0) ew_into(ew::ADD, &out, &bias, out, beta.to<float>());
  return out;
}

static Tensor bmm_(const Tensor& a, const Tensor& b) {
  NEED(f32(a) && f32(b) && a.dim() == 3 && b.dim() == 3, "float32 3-D bmm");
  int64_t B = a.size(0), m = a.size(1), n = b.size(2);
  Tensor out = empty_dev({B, m, n}, at::kFloat);
  if (out.numel() == 0) return out;
  if (a.size(2) == 0) { fill_dev(out, 0.f); return out; }
  for (int64_t i = 0; i < B; i++) mm_into(a.select(0, i), b.select(0, i), nullptr, out, i * m * n);
  return out;
}

void bind_linalg(py::module_& m) {
  m.def("set_mm_planner", [](py::function f) { OMT_LOCK; g_mm_planner = f; });
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  OMT_IMPL(m, mm, mm_);
  OMT_IMPL(m, addmm, addmm_);
  OMT_IMPL(m, bmm, bmm_);
}

}  // namespace omt
