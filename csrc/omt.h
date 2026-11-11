// omarchy-mac-torch native backend: shared declarations.
//
// Layout of the C++ sources:
//   runtime.cpp     wgpu context, buffer pool, kernel registry, recorder/queue, uploads/downloads, errors, events
//   device.cpp      allocator, device guard, hooks, generator, CPU fallback, foreach registration
//   elementwise.cpp general strided elementwise kernel, in-place ops, comparisons, where, fill, copies
//   reduce.cpp      sum/mean, argmax, softmax/log_softmax, nll_loss
//   linalg.cpp      mm/addmm/bmm
//   conv.cpp        implicit-GEMM convolutions, tile planning and autotuning
//   nn.cpp          batch norm, layer norm, pooling, dropout, random fills
//   optim.cpp       fused SGD / Adam
//   index.cpp       cat/stack/index/gather
//   plans.cpp       fused conv/BN/ReLU/pool stacks
//   module.cpp      Python bindings
//
// Memory model: a device tensor's storage is a pooled wgpu buffer holding exactly the bytes PyTorch would hold on the
// CPU (float32 = 4 bytes, int64 = 8 bytes little endian, bool = 1 byte). Storage-level operations (torch.save, deepcopy,
// .view(dtype), untyped_storage copies) therefore behave as they do for any other device. Kernels address buffers as
// arrays of 32-bit words; int64 values are read through their low word (indices and labels always fit) and written as
// (low, sign-extended high) pairs.
#pragma once

#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/Operators.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <webgpu.h>
#include <wgpu.h>

#include <cstring>
#include <map>
#include <tuple>
#include <mutex>
#include <string>
#include <vector>

namespace omt {

using at::Tensor;
using c10::IntArrayRef;
using c10::Scalar;
constexpr auto kDev = c10::DeviceType::PrivateUse1;

// ---- runtime (runtime.cpp) -----------------------------------------------------------------------------------------

struct Buf {
  WGPUBuffer b = nullptr;
  size_t bytes = 0;
  uint64_t id = 0;       // unique for the life of the process (bind groups are keyed by it, never by address)
  bool uniform = false;  // a cached uniform buffer
  int pins = 0;          // uniform buffers referenced by a prebuilt plan are never evicted
};

struct Binding {
  Buf* buf;
  uint64_t off;   // bytes
  uint64_t size;  // bytes
};

// One lock around everything that touches the recorder, caches or queue. Recursive because ops call ops.
std::recursive_mutex& big_lock();
#define OMT_LOCK std::lock_guard<std::recursive_mutex> omt_lock_(::omt::big_lock())

void init_ctx();
WGPUDevice device();
WGPUQueue queue();
WGPUAdapter adapter();

Buf* alloc_buf(size_t bytes);
void free_buf(void* p);
void empty_cache();
size_t memory_allocated();
size_t memory_reserved();

void register_source(const std::string& name, const std::string& wgsl);
std::vector<std::string> validate_kernels();  // compile every registered kernel; returns "name: error" for failures
Buf* uniform(const std::vector<uint32_t>& words);
void pin(Buf* b);
void unpin(Buf* b);

void launch(const char* kernel, const std::vector<Binding>& b, uint32_t x, uint32_t y = 1, uint32_t z = 1);
void record_copy(Buf* src, uint64_t src_off, Buf* dst, uint64_t dst_off, uint64_t bytes);  // 4-byte aligned
void grid(int64_t total, int wg, uint32_t& gx, uint32_t& gy);
void submit_pending();
void sync_all();
uint64_t last_submission();
bool submission_done(uint64_t idx);
void wait_submission(uint64_t idx);
void check_errors();  // raise any asynchronous wgpu error (validation, device lost) as a Python exception

void upload_bytes(Buf* dst, uint64_t off, const void* src, size_t bytes);  // any alignment
void download_bytes(Buf* src, uint64_t off, void* dst, size_t bytes);    // waits for the GPU

// debug / tuning state shared with module.cpp
struct RuntimeStats {
  std::map<std::string, int64_t> kcount;
  std::map<std::string, double> ktime;
  bool time_each = false, skip = false, debug = false;
};
RuntimeStats& stats();
void set_queue(int chunk, int depth);
void set_target_backlog_ms(double ms);
std::tuple<int, int, int, double> queue_state();
std::string adapter_name();

// ---- tensors ------------------------------------------------------------------------------------------------------

inline Buf* bufof(const Tensor& t) { return (Buf*)t.storage().data_ptr().get_context(); }
inline uint64_t off_bytes(const Tensor& t) { return (uint64_t)t.storage_offset() * t.element_size(); }
// offset in 32-bit words of a 4-byte-element tensor (float32 / int32); int64 tensors use 2 words per element
inline uint64_t off_words(const Tensor& t) { return off_bytes(t) / 4; }
inline bool is_dev(const Tensor& t) { return t.defined() && t.device().type() == kDev; }
inline Binding bind_buf(Buf* b) { return {b, 0, b->bytes}; }
inline Binding bind(const Tensor& t) { return bind_buf(bufof(t)); }  // whole buffer; offsets go in uniforms
inline bool f32(const Tensor& t) { return is_dev(t) && t.scalar_type() == at::kFloat; }
inline uint32_t u(int64_t v) { return (uint32_t)v; }
inline uint32_t fbits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  return x;
}
inline int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

Tensor empty_dev(IntArrayRef size, c10::ScalarType dt);
Tensor download(const Tensor& t);            // device -> new CPU tensor
void upload(const Tensor& cpu, Tensor& dst);  // CPU -> existing device tensor (same shape and dtype)
Tensor to_dev(const Tensor& cpu);
inline Tensor to_host(const Tensor& t) { return is_dev(t) ? download(t) : t; }
inline std::optional<Tensor> to_host(const std::optional<Tensor>& t) {
  return (t && t->defined()) ? std::optional<Tensor>(to_host(*t)) : std::nullopt;
}
Tensor contig(const Tensor& t);  // contiguous (any dtype)
Tensor base0(const Tensor& t);   // contiguous with storage offset 0 (kernels that take no offsets)
// whether a fresh tensor owns its whole storage, so a kernel may write the padding up to the next 4-byte word
inline bool owns_storage(const Tensor& t) {
  return t.storage_offset() == 0 && t.is_contiguous() && (size_t)t.storage().nbytes() == (size_t)t.nbytes();
}

// ---- elementwise (elementwise.cpp), used by other ops ---------------------------------------------------------------

namespace ew {
enum Op : uint32_t {
  ADD = 0, SUB = 1, MUL = 2, DIV = 3, NEG = 4, RELU = 5, EXP = 6, SQRT = 7, THRESH_BWD = 8, COPY = 9, FILL = 10,
  SCALE = 11, XLOGY = 12, SIGMOID = 13, SILU = 14, HARDSWISH = 15, SIGMOID_BWD = 16, SILU_BWD = 17,
  HARDSWISH_BWD = 18, HARDSIGMOID = 19, HARDSIGMOID_BWD = 20, TANH = 21, TANH_BWD = 22, CLAMP = 23, LOG = 24,
  ABS = 25, RSQRT = 26, POW = 27, MAX = 28, MIN = 29, RECIP = 30, FILL2 = 31, GELU = 32, GELU_BWD = 33,
  SIN = 34, COS = 35, SQUARE = 36, LERP = 37, RSUB = 38, ADD_SQ = 39, RSQRT_EPS = 40
};
}
// out[i] = op(a[i], b[i]) for float32 tensors of any rank <= 8 and any strides (b may be null; sval replaces it).
// out may alias a or b (in-place); partial overlap is rejected the way PyTorch rejects it on the CPU.
void ew_into(uint32_t op, const Tensor* a, const Tensor* b, const Tensor& out, float alpha = 1.f, float sval = 0.f);
Tensor ew_new(uint32_t op, const Tensor* a, const Tensor* b, float alpha = 1.f, float sval = 0.f);
void copy_dev(const Tensor& src, const Tensor& dst);  // device -> device, same dtype, any strides (broadcast src)
void fill_dev(const Tensor& t, const Scalar& v);


// ---- fallback by construction (device.cpp) -----------------------------------------------------------------------
//
// Every op is registered through Safe<Op, Fn>. An implementation checks what it supports with NEED(cond, why) before
// it launches anything; a failed NEED() throws Unsupported and Safe re-runs the call through PyTorch's boxed CPU
// fallback. No op can fail just because a case isn't implemented on the GPU. OMARCHY_MAC_TORCH_STRICT=1 turns
// Unsupported into an error instead (to find what falls back).

struct Unsupported : std::exception {
  std::string why;
  explicit Unsupported(std::string w) : why(std::move(w)) {}
  const char* what() const noexcept override { return why.c_str(); }
};
#define NEED(cond, why) \
  do {                  \
    if (!(cond)) throw ::omt::Unsupported(why); \
  } while (0)

void cpu_fallback_boxed(const c10::OperatorHandle& op, torch::jit::Stack* stack);
void cpu_fallback_quiet(const c10::OperatorHandle& op, torch::jit::Stack* stack);
void note_fallback(const std::string& op, const std::string& why);
std::map<std::string, int64_t>& fallback_counts();
bool strict_mode();
void debug_begin();
void debug_end(const char* op);

template <class Op, auto Fn, class Sig = typename Op::schema>
struct Safe;
template <class Op, auto Fn, class R, class... A>
struct Safe<Op, Fn, R(A...)> {
  static R call(typename c10::maybe_keep_symint<false, A>::type... a) {
    OMT_LOCK;
    check_errors();
    try {
      if (!stats().debug) return Fn(a...);
      debug_begin();
      if constexpr (std::is_void_v<R>) {
        Fn(a...);
        debug_end(Op::name);
      } else {
        R r = Fn(a...);
        debug_end(Op::name);
        return r;
      }
    } catch (const Unsupported& e) {
      std::string name = std::string(Op::name).substr(6);
      if (Op::overload_name[0]) name += std::string(".") + Op::overload_name;
      TORCH_CHECK(!strict_mode(), "omarchy-mac-torch (strict): ", name, " not supported on the GPU: ", e.why);
      note_fallback(name, e.why);
      return at::native::call_fallback_fn<&cpu_fallback_quiet, Op>::call(a...);
    }
  }
};
// register aten::<op> for PrivateUse1 through the fallback wrapper
#define OMT_IMPL(m, op, fn) ::omt::impl_name<at::_ops::op>(m, &::omt::Safe<at::_ops::op, fn>::call)
template <class Op, class F>
void impl_name(torch::Library& m, F f) {
  std::string name = std::string(Op::name).substr(6);
  if (Op::overload_name[0]) name += std::string(".") + Op::overload_name;
  m.impl(name.c_str(), f);
}

// ---- conv (conv.cpp), used by plans --------------------------------------------------------------------------------

struct Geo {  // input n x ci x ih x iw, kernel ks x ks, output n x co x oh x ow
  int64_t n, ci, co, ih, iw, oh, ow, ks, stride, pad;
};
inline Geo same_geo(int64_t n, int64_t ci, int64_t co, int64_t h, int64_t w, int64_t ks) {
  return {n, ci, co, h, w, h, w, ks, 1, ks / 2};
}
struct ConvPlan {
  std::string kernel;
  int bm = 64, bn = 64;
  int tile = -1;
  double mult = 1.0;  // split-K multiplier for the weight gradient
};
void mnk(int kind, const Geo& g, int64_t& M, int64_t& N, int64_t& K);
ConvPlan conv_plan(int kind, const Geo& g);
Buf* pbuf_geo(const Geo& g, int64_t M, int64_t N, int64_t K, int64_t kslice, int64_t slices, uint32_t epilogue = 0);  // epilogue bits: 1 bias, 2 ReLU
Geo conv_geo(int64_t n, int64_t ci, int64_t co, int64_t ih, int64_t iw, int64_t ks, int64_t stride, int64_t pad);
void gemm_forward(const Tensor& x, const Tensor& w, const Tensor* bias, const Tensor& out, const Geo& g, bool relu = false);
Buf* pbuf_reduce(int64_t M, int64_t N, int64_t slices);
Buf* dummy_buf();  // bound where a kernel declares an input it won't read
extern int g_fold;

}  // namespace omt
