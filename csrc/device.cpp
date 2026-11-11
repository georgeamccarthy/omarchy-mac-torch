// Allocator, device guard (with events), hooks and generator, CPU fallback, and the memory/copy ops.
#include "omt.h"

#include <ATen/CPUGeneratorImpl.h>
#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/Resize.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include <cstdlib>

namespace omt {

// ------------------------------------------------------------------------------------------------
// allocator
// ------------------------------------------------------------------------------------------------

struct Allocator final : c10::Allocator {
  c10::DataPtr allocate(size_t n) override {
    Buf* b = alloc_buf(std::max<size_t>(n, 4));
    return {(void*)b, (void*)b, &free_buf, c10::Device(kDev, 0)};
  }
  // storage-to-storage copies (storage clone / deepcopy): data pointers are Buf* (+ byte offset)
  void copy_data(void* dst, const void* src, std::size_t count) const override {
    TORCH_CHECK(false, "omarchy-mac-torch: raw storage copy_data is not supported; copy tensors instead");
  }
  c10::DeleterFnPtr raw_deleter() const override { return &free_buf; }
};
static Allocator g_alloc;
REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &g_alloc);

Tensor empty_dev(IntArrayRef size, c10::ScalarType dt) {
  c10::DispatchKeySet ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &g_alloc, ks, dt, std::nullopt);
}

// ------------------------------------------------------------------------------------------------
// transfers between host and device
// ------------------------------------------------------------------------------------------------

void upload(const Tensor& cpu, Tensor& dst) {
  TORCH_INTERNAL_ASSERT(dst.is_contiguous() && cpu.scalar_type() == dst.scalar_type());
  Tensor c = cpu.contiguous();
  upload_bytes(bufof(dst), off_bytes(dst), c.data_ptr(), c.nbytes());
}

Tensor to_dev(const Tensor& cpu) {
  Tensor out = empty_dev(cpu.sizes(), cpu.scalar_type());
  upload(cpu, out);
  return out;
}

Tensor contig(const Tensor& t) {
  if (t.is_contiguous()) return t;
  Tensor out = empty_dev(t.sizes(), t.scalar_type());
  copy_dev(t, out);
  return out;
}

Tensor base0(const Tensor& t) {
  Tensor c = contig(t);
  if (c.storage_offset() == 0) return c;
  Tensor o = empty_dev(c.sizes(), c.scalar_type());
  copy_dev(c, o);
  return o;
}

// ------------------------------------------------------------------------------------------------
// memory / copy ops
// ------------------------------------------------------------------------------------------------

static Tensor empty_memory_format_impl(IntArrayRef size, std::optional<at::ScalarType> dtype, std::optional<at::Layout>,
                                  std::optional<at::Device>, std::optional<bool>, std::optional<at::MemoryFormat> mf) {
  c10::DispatchKeySet ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_generic(size, &g_alloc, ks, dtype.value_or(at::kFloat), mf);
}
static Tensor empty_strided_impl(IntArrayRef size, IntArrayRef stride, std::optional<at::ScalarType> dtype,
                            std::optional<at::Layout>, std::optional<at::Device>, std::optional<bool>) {
  c10::DispatchKeySet ks(c10::DispatchKey::PrivateUse1);
  return at::detail::empty_strided_generic(size, stride, &g_alloc, ks, dtype.value_or(at::kFloat));
}

static Tensor copy_from_impl(const Tensor& self, const Tensor& dst, bool) {  // copy self -> dst
  if (!is_dev(self) && is_dev(dst)) {  // host -> device
    Tensor src = self.to(dst.scalar_type()).expand(dst.sizes());
    if (dst.is_contiguous()) {
      Tensor d = dst;
      upload(src, d);
    } else {
      Tensor tmp = to_dev(src.contiguous());
      copy_dev(tmp, dst);
    }
  } else if (is_dev(self) && !is_dev(dst)) {  // device -> host
    const_cast<Tensor&>(dst).copy_(download(self));
  } else if (self.scalar_type() == dst.scalar_type()) {  // device -> device
    copy_dev(self, dst);
  } else {  // dtype conversion on the device: through the host (rare: .to(dtype) of a device tensor)
    Tensor cpu = download(self).to(dst.scalar_type()).expand(dst.sizes()).contiguous();
    Tensor tmp = to_dev(cpu);
    copy_dev(tmp, dst);
  }
  return dst;
}
static Tensor copy_from_and_resize_impl(const Tensor& self, const Tensor& dst) {
  at::native::resize_(dst, self.sizes(), std::nullopt);
  return copy_from_impl(self, dst, false);
}
static Scalar local_scalar(const Tensor& self) { return download(self).item(); }
static Tensor& set_storage(Tensor& self, c10::Storage src, int64_t off, IntArrayRef size, IntArrayRef stride) {
  return at::native::set_storage_cpu_(self, src, off, size, stride);  // metadata only
}

// ------------------------------------------------------------------------------------------------
// device guard with events
// ------------------------------------------------------------------------------------------------
//
// One in-order queue, so streams are trivial. An event is a queue submission index: recorded when the work before
// it has been submitted, complete when that submission has finished on the GPU.

struct Event {
  uint64_t index = 0;
};

struct Guard final : c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override { return kDev; }
  c10::Device exchangeDevice(c10::Device) const override { return {kDev, 0}; }
  c10::Device getDevice() const override { return {kDev, 0}; }
  void setDevice(c10::Device) const override {}
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Stream getStream(c10::Device d) const noexcept override { return c10::Stream(c10::Stream::DEFAULT, d); }
  c10::Stream getDefaultStream(c10::Device d) const override { return c10::Stream(c10::Stream::DEFAULT, d); }
  c10::Stream getNewStream(c10::Device d, int) const override { return c10::Stream(c10::Stream::DEFAULT, d); }
  c10::Stream exchangeStream(c10::Stream s) const noexcept override { return s; }
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }
  void record(void** event, const c10::Stream&, const c10::DeviceIndex, const c10::EventFlag) const override {
    OMT_LOCK;
    if (!*event) *event = new Event();
    submit_pending();
    ((Event*)*event)->index = last_submission();
  }
  void block(void*, const c10::Stream&) const override {}  // a single in-order queue
  bool queryEvent(void* event) const override {
    if (!event) return true;
    OMT_LOCK;
    return submission_done(((Event*)event)->index);
  }
  void destroyEvent(void* event, const c10::DeviceIndex) const noexcept override { delete (Event*)event; }
  void synchronizeEvent(void* event) const override {
    if (!event) return;
    OMT_LOCK;
    wait_submission(((Event*)event)->index);
  }
  bool queryStream(const c10::Stream&) const override {
    OMT_LOCK;
    submit_pending();
    return submission_done(last_submission());
  }
  void synchronizeStream(const c10::Stream&) const override { sync_all(); }
  void synchronizeDevice(const c10::DeviceIndex) const override { sync_all(); }
  void recordDataPtrOnStream(const c10::DataPtr&, const c10::Stream&) const override {}
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, Guard);

// ------------------------------------------------------------------------------------------------
// hooks and generator (random numbers are drawn on the host with this device's own generator, then uploaded)
// ------------------------------------------------------------------------------------------------

struct Generator : at::CPUGeneratorImpl {
  explicit Generator(c10::DeviceIndex i) {
    device_ = c10::Device(kDev, i);
    key_set_ = c10::DispatchKeySet(c10::DispatchKey::PrivateUse1);
  }
};
static const at::Generator& default_generator() {
  static at::Generator g = at::make_generator<Generator>(0);
  return g;
}

struct Hooks final : at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return true; }
  bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
  c10::DeviceIndex deviceCount() const override { return 1; }
  c10::DeviceIndex getCurrentDevice() const override { return 0; }
  void setCurrentDevice(c10::DeviceIndex) const override {}
  c10::DeviceIndex exchangeDevice(c10::DeviceIndex) const override { return 0; }
  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex) const override { return 0; }
  const at::Generator& getDefaultGenerator(c10::DeviceIndex) const override { return default_generator(); }
  at::Generator getNewGenerator(c10::DeviceIndex i) const override { return at::make_generator<Generator>(i); }
};
static struct HooksRegistrar {
  HooksRegistrar() {
    static Hooks hooks;
    at::RegisterPrivateUse1HooksInterface(&hooks);
  }
} g_hooks_registrar;

at::CPUGeneratorImpl* generator(const std::optional<at::Generator>& gen) {
  const at::Generator& g = (gen && gen->defined()) ? *gen : default_generator();
  TORCH_CHECK(g.device().type() == kDev, "omarchy-mac-torch: expected a generator for this device, got one for ", g.device());
  return static_cast<at::CPUGeneratorImpl*>(g.unsafeGetGeneratorImpl());  // Generator derives from CPUGeneratorImpl
}

// ------------------------------------------------------------------------------------------------
// CPU fallback
// ------------------------------------------------------------------------------------------------

std::map<std::string, int64_t>& fallback_counts() {
  static std::map<std::string, int64_t> m;
  return m;
}
void note_fallback(const std::string& op, const std::string& why) { fallback_counts()[op + " (" + why + ")"]++; }
bool strict_mode() {
  static bool s = [] { const char* e = std::getenv("OMARCHY_MAC_TORCH_STRICT"); return e && e[0] == '1'; }();
  return s;
}

void cpu_fallback_boxed(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  const auto& name = op.schema().name();
  // foreach ops (optimizers, clip_grad_norm): their CPU kernels are plain loops over per-tensor ops, so run them on
  // the device tensors directly; each iteration dispatches to the device kernels
  if (name.rfind("aten::_foreach_", 0) == 0) {
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);
    return;
  }
  std::string key = name.substr(6) + (op.schema().overload_name().empty() ? "" : "." + op.schema().overload_name());
  TORCH_CHECK(!strict_mode(), "omarchy-mac-torch (strict): ", key, " has no GPU kernel");
  {
    OMT_LOCK;
    fallback_counts()[key]++;
  }
  at::native::cpu_fallback(op, stack);
}

// used by Safe<> after it has recorded the reason itself
void cpu_fallback_quiet(const c10::OperatorHandle& op, torch::jit::Stack* stack) { at::native::cpu_fallback(op, stack); }

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) { m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback_boxed>()); }

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  OMT_IMPL(m, empty_memory_format, empty_memory_format_impl);
  OMT_IMPL(m, empty_strided, empty_strided_impl);
  OMT_IMPL(m, _copy_from, copy_from_impl);
  OMT_IMPL(m, _copy_from_and_resize, copy_from_and_resize_impl);
  OMT_IMPL(m, _local_scalar_dense, local_scalar);
  OMT_IMPL(m, set__source_Storage_storage_offset, set_storage);
  m.impl("view", &at::native::view);
  m.impl("_unsafe_view", &at::native::_unsafe_view);
  m.impl("as_strided", &at::native::as_strided_tensorimpl);
  m.impl("_reshape_alias", &at::native::_reshape_alias);
  m.impl("resize_", &at::native::resize_);
  m.impl("unfold", &at::native::unfold);  // views can't fall back to the CPU, so every view op needs an impl
  m.impl("view_as_real", &at::native::view_as_real);
  m.impl("view_as_complex", &at::native::view_as_complex);
  m.impl("set_.source_Storage", &at::native::set_);  // storage copies (torch.save, deepcopy) attach storages to tensors
}

}  // namespace omt
