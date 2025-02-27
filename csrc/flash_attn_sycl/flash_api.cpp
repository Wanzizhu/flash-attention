#include "sytla.hpp"

#include "params.h"
#include <c10/xpu/XPUStream.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

// TODO: add dispatch code
void run_mha_fwd(Flash_fwd_params &params, XPUStream stream) {
  using ScalarT = sycl::half;
  using Policy = sytla::flash::FlashForwardPolicy<ScalarT, 8, 64, 8, 64, 32, 3>;
  using Kernel = sytla::flash::FlashForward<Policy, kIsCausal>;
  typename Kernel::Arguments args{static_cast<ScalarT *>(params.query),
                                  static_cast<ScalarT *>(params.key),
                                  static_cast<ScalarT *>(params.value),
                                  static_cast<ScalarT *>(params.output),
                                  params.softmax_scale,
                                  params.bs,
                                  params.nheads,
                                  params.head_dim,
                                  params.q_seqlen,
                                  params.kv_seqlen};
  Kernel kernel(args);
  auto nd_range = kernel.get_nd_range();
  stream.submit(
      [&](sycl::handler &cgh) { cgh.parallel_for(nd_range, kernel); });
}

/**
 * @brief Performs the forward pass of multi-head attention (MHA).
 *
 * This function computes the forward pass for multi-head attention by
 * performing attention on the query (q), key (k), and value (v) tensors. It
 * also supports optional output and alibi slopes tensors, and various
 * customization parameters such as dropout, softmax scaling, and window size
 * for causality.
 *
 * @param q Input tensor representing queries. Shape: [batch_size, seqlen_q,
 * num_heads, round_multiple(head_size, 8)].
 * @param k Input tensor representing keys. Shape: [batch_size, seqlen_k,
 * num_heads_k, round_multiple(head_size, 8)].
 * @param v Input tensor representing values. Shape: [batch_size, seqlen_k,
 * num_heads_k, round_multiple(head_size, 8)].
 * @param out_ Optional output tensor to store the result of the MHA. Shape:
 * [batch_size, seqlen_q, num_heads, round_multiple(head_size, 8)].
 * @param alibi_slopes_ Optional tensor containing alibi slopes. Shape:
 * [num_heads] or [batch_size, num_heads].
 * @param p_dropout Dropout probability to be applied during the MHA
 * computation.
 * @param softmax_scale Scaling factor to be applied to the softmax function.
 * @param is_causal If true, causal masking is applied to prevent attending to
 * future tokens.
 * @param window_size_left Size of the left window for local attention (causal).
 * @param window_size_right Size of the right window for local attention.
 * @param softcap Soft maximum cap applied during the softmax computation.
 * @param return_softmax If true, the softmax output will be returned alongside
 * the final output.
 * @param gen_ Optional random number generator to be used for dropout.
 *
 * @return A vector of tensors containing the result of the multi-head attention
 * forward pass.
 */

std::vector<at::Tensor>
mha_fwd(at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
        std::optional<at::Tensor> &out_,
        std::optional<at::Tensor> &alibi_slopes_, const float p_dropout,
        const float softmax_scale, bool is_causal, int window_size_left,
        int window_size_right, const float softcap, const bool return_softmax,
        std::optional<at::Generator> gen_) {

  auto q_dtype = q.dtype();
  TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
              "FlashAttention only support fp16 and bf16 data type");
  TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
  TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");

  // TODO: remove below limitations in the future
  TORCH_CHECK(p_dropout == 0.f && !gen_.has_value(),
              "dropout does not support for now");
  TORCH_CHECK(softcap == 0.f, "softcap does not support for now");
  TORCH_CHECK(!return_softmax, "return_softmax does not support for now");
  TORCH_CHECK(!alibi_slopes_.has_value(),
              "alibi_slopes_ does not support for now");
  TORCH_CHECK(num_heads == num_heads_k,
              "Number of heads in query/key must be the same for now");
  const auto sizes = q.sizes();

  const int batch_size = sizes[0];
  int seqlen_q = sizes[1];
  int num_heads = sizes[2];
  const int head_size = sizes[3];
  const int seqlen_k = k.size(1);
  const int num_heads_k = k.size(2);
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(head_size * 2 % 64 == 0,
              "q/k/v must have leading dimension that is a multiple of 64");

  if (window_size_left >= seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k) {
    window_size_right = -1;
  }
  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1 && !alibi_slopes_.has_value()) {
    is_causal = false;
  }
  if (is_causal) {
    window_size_right = 0;
  }

  CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
  CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size);
  CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size);

  at::Tensor out;
  if (out_.has_value()) {
    out = out_.value();
    TORCH_CHECK(out.dtype() == q_dtype,
                "Output must have the same dtype as inputs");
    CHECK_DEVICE(out);
    TORCH_CHECK(out.is_contiguous(), "Output tensor must be contiguous");
    CHECK_SHAPE(out, batch_size, sizes[1], sizes[2], head_size);
  } else {
    out = torch::empty_like(q);
  }
  auto softmax_lse =
      torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
  at::Tensor p = torch::empty({0}, q.options());
  auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));

  Flash_fwd_params params(q.data_ptr(), k.data_ptr(), v.data_ptr(),
                          out.data_ptr(), batch_size, num_heads, seqlen_q,
                          seqlen_k, head_size, softmax_scale);
  //   TODO: get current xpu device queue
  auto stream = at::xpu::getCurrentXPUStream();
  run_mha_fwd(params, stream);

  return {out, softmax_scale, p, rng_state};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "FlashAttention";
  m.def("fwd", &mha_fwd, "Forward pass");
  //         m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable
  //         length)"); m.def("bwd", &mha_bwd, "Backward pass");
  //         m.def("varlen_bwd", &mha_varlen_bwd, "Backward pass (variable
  //         length)"); m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass,
  //         with KV-cache");
}
