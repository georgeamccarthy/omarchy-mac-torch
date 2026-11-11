// Fused SGD and Adam / AdamW (torch.optim's fused=True path): one pass over each parameter.
#include "omt.h"

namespace omt {

static void ew_launch(const char* k, const std::vector<Binding>& b, int64_t total) {
  uint32_t gx, gy;
  grid(total, 256, gx, gy);
  launch(k, b, gx, gy);
}

static void fused_sgd_(at::TensorList params, at::TensorList grads, at::TensorList momentum_bufs, double weight_decay,
                       double momentum, double lr, double dampening, bool nesterov, bool maximize, bool is_first_step,
                       const std::optional<Tensor>& grad_scale, const std::optional<Tensor>& found_inf) {
  NEED(!grad_scale && !found_inf, "fused SGD with grad scaling");
  for (size_t i = 0; i < params.size(); i++)
    NEED(f32(params[i]) && f32(grads[i]) && params[i].is_contiguous() && grads[i].is_contiguous(), "contiguous float32 params");
  Buf* h = uniform({fbits((float)lr), fbits((float)momentum), fbits((float)dampening), fbits((float)weight_decay),
                    nesterov ? 1u : 0u, is_first_step ? 1u : 0u, maximize ? 1u : 0u, 0});
  for (size_t i = 0; i < params.size(); i++) {
    bool has_m = momentum != 0 && momentum_bufs.size() > i && momentum_bufs[i].defined();
    const Tensor& mbuf = has_m ? momentum_bufs[i] : params[i];
    Buf* a = uniform({u(params[i].numel()), u(params[i].storage_offset()), u(grads[i].storage_offset()), u(mbuf.storage_offset())});
    ew_launch("sgd", {bind(params[i]), bind(grads[i]), bind(mbuf), bind_buf(a), bind_buf(h)}, params[i].numel());
  }
}

static void adam_impl(at::TensorList params, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                      at::TensorList state_steps, double lr, double beta1, double beta2, double weight_decay, double eps,
                      bool amsgrad, bool maximize, bool decoupled, const std::optional<Tensor>& grad_scale,
                      const std::optional<Tensor>& found_inf) {
  NEED(!amsgrad && !grad_scale && !found_inf, "fused Adam with amsgrad or grad scaling");
  for (size_t i = 0; i < params.size(); i++) {
    NEED(f32(params[i]) && f32(grads[i]) && f32(exp_avgs[i]) && f32(exp_avg_sqs[i]) && f32(state_steps[i]),
         "float32 params and device step tensors");
    NEED(params[i].is_contiguous() && grads[i].is_contiguous() && exp_avgs[i].is_contiguous() && exp_avg_sqs[i].is_contiguous(),
         "contiguous params");
  }
  Buf* h = uniform({fbits((float)lr), fbits((float)beta1), fbits((float)beta2), fbits((float)eps), fbits((float)weight_decay),
                    decoupled ? 1u : 0u, maximize ? 1u : 0u, 0});
  for (size_t i = 0; i < params.size(); i++) {
    Buf* a = uniform({u(params[i].numel()), u(params[i].storage_offset()), u(grads[i].storage_offset()),
                      u(exp_avgs[i].storage_offset()), u(exp_avg_sqs[i].storage_offset()), u(state_steps[i].storage_offset()), 0, 0});
    ew_launch("adam", {bind(params[i]), bind(grads[i]), bind(exp_avgs[i]), bind(exp_avg_sqs[i]), bind(state_steps[i]),
                       bind_buf(a), bind_buf(h)}, params[i].numel());
  }
}
static void fused_adam_(at::TensorList params, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                        at::TensorList, at::TensorList state_steps, double lr, double beta1, double beta2,
                        double weight_decay, double eps, bool amsgrad, bool maximize,
                        const std::optional<Tensor>& grad_scale, const std::optional<Tensor>& found_inf) {
  adam_impl(params, grads, exp_avgs, exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize,
            false, grad_scale, found_inf);
}
static void fused_adamw_(at::TensorList params, at::TensorList grads, at::TensorList exp_avgs, at::TensorList exp_avg_sqs,
                         at::TensorList, at::TensorList state_steps, double lr, double beta1, double beta2,
                         double weight_decay, double eps, bool amsgrad, bool maximize,
                         const std::optional<Tensor>& grad_scale, const std::optional<Tensor>& found_inf) {
  adam_impl(params, grads, exp_avgs, exp_avg_sqs, state_steps, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize,
            true, grad_scale, found_inf);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  OMT_IMPL(m, _fused_sgd_, fused_sgd_);
  OMT_IMPL(m, _fused_adam_, fused_adam_);
  OMT_IMPL(m, _fused_adamw_, fused_adamw_);
}

}  // namespace omt
