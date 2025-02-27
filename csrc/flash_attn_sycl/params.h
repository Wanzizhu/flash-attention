#include <string>

struct Flash_fwd_params {
  // The QKV matrices.
  void *__restrict__ query;
  void *__restrict__ key;
  void *__restrict__ value;
  void *__restrict__ output;

  int bs;
  int nheads;
  int q_seqlen;
  int kv_seqlen;
  int head_dim;
  float softmax_scale;
  bool is_causal;
  std::string data_type;
};