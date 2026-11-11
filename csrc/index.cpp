// cat / stack (any dim, any dtype: strided copies into slices of the output), row gathers along dim 0.
#include "omt.h"

#include <ATen/WrapDimUtils.h>

namespace omt {

static Tensor cat_(const at::ITensorListRef& list, int64_t dim) {
  std::vector<Tensor> ts;
  for (const Tensor& t : list) ts.push_back(t);
  NEED(!ts.empty(), "empty cat");
  // legacy behaviour: 1-D empty tensors are skipped
  std::vector<Tensor> parts;
  for (auto& t : ts) if (!(t.dim() == 1 && t.numel() == 0)) parts.push_back(t);
  if (parts.empty()) return empty_dev({0}, ts[0].scalar_type());
  const Tensor& ref = parts[0];
  for (auto& t : parts) NEED(is_dev(t) && t.scalar_type() == ref.scalar_type() && t.dim() == ref.dim(), "cat of mixed dtypes or devices");
  dim = at::maybe_wrap_dim(dim, ref.dim());
  std::vector<int64_t> shape = ref.sizes().vec();
  shape[dim] = 0;
  for (auto& t : parts) shape[dim] += t.size(dim);
  Tensor out = empty_dev(shape, ref.scalar_type());
  int64_t off = 0;
  for (auto& t : parts) {
    copy_dev(t, out.narrow(dim, off, t.size(dim)));
    off += t.size(dim);
  }
  return out;
}

static Tensor stack_(at::TensorList list, int64_t dim) {
  NEED(!list.empty(), "empty stack");
  dim = at::maybe_wrap_dim(dim, list[0].dim() + 1);
  std::vector<Tensor> ts;
  for (auto& t : list) ts.push_back(t.unsqueeze(dim));
  return cat_(ts, dim);
}

static Tensor gather0(const Tensor& x_, const Tensor& idx_) {
  NEED(is_dev(x_) && (x_.element_size() == 4 || x_.element_size() == 8) && x_.dim() >= 1, "row gather of a 4/8-byte dtype");
  NEED(idx_.scalar_type() == at::kLong || idx_.scalar_type() == at::kInt, "integer index");
  Tensor x = contig(x_);
  Tensor idx = is_dev(idx_) ? contig(idx_) : to_dev(idx_.contiguous());
  int64_t n = idx.numel(), wpe = x.element_size() / 4;
  int64_t row = (x.size(0) ? x.numel() / x.size(0) : 0) * wpe;
  std::vector<int64_t> shape = idx.sizes().vec();
  for (int i = 1; i < x.dim(); i++) shape.push_back(x.size(i));
  Tensor out = empty_dev(shape, x.scalar_type());
  if (!n || !row) return out;
  uint32_t iw = idx.scalar_type() == at::kLong ? 2 : 1;
  Buf* ub = uniform({u(n), u(row), u(off_bytes(x) / 4), u(idx.storage_offset()), iw, u(x.size(0)), 0, 0});
  uint32_t gx, gy;
  grid(n * row, 256, gx, gy);
  launch("gather0", {bind(x), bind(idx), bind(out), bind_buf(ub)}, gx, gy);
  return out;
}

static Tensor index_(const Tensor& x, const c10::List<std::optional<Tensor>>& indices) {
  NEED(indices.size() == 1, "multi-dim advanced indexing");
  std::optional<Tensor> i0 = indices[0];
  NEED(i0 && i0->defined() && (i0->scalar_type() == at::kLong || i0->scalar_type() == at::kInt), "integer index on dim 0");
  return gather0(x, *i0);
}
static Tensor index_select_(const Tensor& x, int64_t dim, const Tensor& idx) {
  NEED(at::maybe_wrap_dim(dim, std::max<int64_t>(x.dim(), 1)) == 0 && idx.dim() <= 1, "index_select along dim 0");
  return gather0(x, idx);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  OMT_IMPL(m, cat, cat_);
  OMT_IMPL(m, stack, stack_);
  OMT_IMPL(m, index_Tensor, index_);
  OMT_IMPL(m, index_select, index_select_);
}

}  // namespace omt
