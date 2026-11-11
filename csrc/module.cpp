// Python bindings (omarchy_mac_torch._native).
#include "omt.h"

#include <ATen/CPUGeneratorImpl.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

namespace omt {

void bind_linalg(py::module_& m);
void bind_conv(py::module_& m);
void bind_plans(py::module_& m);
at::CPUGeneratorImpl* generator(const std::optional<at::Generator>& gen);

PYBIND11_MODULE(_native, m) {
  m.def("register_source", [](const std::string& n, const std::string& s) { OMT_LOCK; register_source(n, s); });
  m.def("validate_kernels", &validate_kernels, "compile every registered kernel; returns the failures");
  m.def("init", &init_ctx);
  m.def("sync", &sync_all);
  m.def("flush", []() { OMT_LOCK; submit_pending(); });
  m.def("check_errors", []() { OMT_LOCK; check_errors(); });
  m.def("set_queue", &set_queue);
  m.def("set_target_backlog_ms", &set_target_backlog_ms);
  m.def("queue_state", &queue_state);
  m.def("empty_cache", &empty_cache);
  m.def("memory_allocated", []() { return (int64_t)memory_allocated(); });
  m.def("memory_reserved", []() { return (int64_t)memory_reserved(); });
  m.def("adapter_name", &adapter_name);
  m.def("set_debug", [](bool on) { OMT_LOCK; stats().debug = on; });
  m.def("debug_time_kernels", [](bool b) { OMT_LOCK; stats().time_each = b; });
  m.def("debug_skip_submit", [](bool b) { OMT_LOCK; stats().skip = b; });
  m.def("kernel_times", [](bool reset) { OMT_LOCK; auto r = stats().ktime; if (reset) stats().ktime.clear(); return r; });
  m.def("kernel_counts", [](bool reset) { OMT_LOCK; auto r = stats().kcount; if (reset) stats().kcount.clear(); return r; });
  m.def("fallback_counts", []() { OMT_LOCK; return fallback_counts(); });
  m.def("reset_fallbacks", []() { OMT_LOCK; fallback_counts().clear(); });
  m.def("manual_seed", [](uint64_t s) {
    auto* g = generator(std::nullopt);
    std::lock_guard<std::mutex> lock(g->mutex_);
    g->set_current_seed(s);
  });
  m.def("seed", []() {
    auto* g = generator(std::nullopt);
    std::lock_guard<std::mutex> lock(g->mutex_);
    uint64_t s = c10::detail::getNonDeterministicRandom();
    g->set_current_seed(s);
    return s;
  });
  m.def("initial_seed", []() { return generator(std::nullopt)->current_seed(); });
  m.def("get_rng_state", []() {
    auto* g = generator(std::nullopt);
    std::lock_guard<std::mutex> lock(g->mutex_);
    return Tensor(g->get_state());
  });
  m.def("set_rng_state", [](const Tensor& state) {
    auto* g = generator(std::nullopt);
    std::lock_guard<std::mutex> lock(g->mutex_);
    g->set_state(*state.unsafeGetTensorImpl());
  });
  m.attr("built_with_torch") = TORCH_VERSION;
  bind_linalg(m);
  bind_conv(m);
  bind_plans(m);
}

}  // namespace omt
