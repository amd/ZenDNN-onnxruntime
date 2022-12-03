// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {
enum MaskIndexType {
  kMaskIndexEnd = 0,      // [batch_size]
  kMaskIndexEndAndStart,  // [2 * batch_size]
  kMaskRaw,               // [batch_size, total_sequence_length]
  kMask3D,                // [batch_size, sequence_length, total_sequence_length]
  kMaskDummy,             // Dummy mask with shape [1, 1] or [batch_size, 1]
  kMask4D                 // Megatron causal mask with shape [batch_size, 1, max_sequence_length, max_sequence_length]
};

static void RunAttentionTest(
    const std::vector<float>& input_data,         // input:      [batch_size, sequence_length, hidden_size]
    const std::vector<float>& weights_data,       // weights:    [hidden_size, 3 * hidden_size]
    bool is_weights_constant,                     // weights is constant
    const std::vector<float>& bias_data,          // bias:       [3 * hidden_size]
    const std::vector<int32_t>& mask_index_data,  // mask_index: see MaskIndexType for supported shape
    const std::vector<float>& output_data,        // output:     [batch_size, sequence_length, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    bool use_float16 = false,
    bool is_unidirectional = false,
    bool use_past_state = false,
    int past_sequence_length = 0,
    const std::vector<float>* past_data = nullptr,
    const std::vector<float>* present_data = nullptr,
    MaskIndexType mask_index_type = kMaskIndexEnd,
    int input_hidden_size = 0,
    int max_sequence_length = 0,
    const bool disable_cpu = false,
    const bool disable_cuda = false,
    const bool disable_rocm = false,
    std::vector<int32_t> qkv_sizes = {},
    const std::vector<float>& extra_add_data = {},
    int kv_sequence_length = 0,
    const std::vector<float>* key_data = nullptr,
    const std::vector<float>* value_data = nullptr) {
  input_hidden_size = (input_hidden_size == 0 ? hidden_size : input_hidden_size);  // By default, no pruning.
  kv_sequence_length = (kv_sequence_length == 0 ? sequence_length : kv_sequence_length);

  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture) && !is_weights_constant && !disable_cuda;
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get()) && !is_weights_constant && !disable_rocm;
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get()) && !use_float16 && !disable_cpu;

  int head_size = hidden_size / number_of_heads;
  if (enable_cpu || enable_cuda || enable_rocm) {
    OpTester tester("Attention", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
    tester.AddAttribute<int64_t>("unidirectional", static_cast<int64_t>(is_unidirectional ? 1 : 0));

    int32_t qkv_hidden_size_sum;
    int32_t v_hidden_size;
    if (qkv_sizes.size() != 0) {
      qkv_hidden_size_sum = qkv_sizes[0] + qkv_sizes[1] + qkv_sizes[2];
      std::vector<int64_t> sizes_attribute{qkv_sizes[0], qkv_sizes[1], qkv_sizes[2]};
      tester.AddAttribute<std::vector<int64_t>>("qkv_hidden_sizes", sizes_attribute);
      v_hidden_size = qkv_sizes[2];
    } else {
      qkv_hidden_size_sum = 3 * hidden_size;
      v_hidden_size = hidden_size;
    }

    int64_t total_sequence_length = past_sequence_length + kv_sequence_length;

    std::vector<int64_t> input_dims = {batch_size, sequence_length, input_hidden_size};
    std::vector<int64_t> weights_dims = {input_hidden_size, qkv_hidden_size_sum};
    std::vector<int64_t> bias_dims = {qkv_hidden_size_sum};
    std::vector<int64_t> output_dims = {batch_size, sequence_length, v_hidden_size};
    if (use_float16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));

      if (weights_data.empty()) {
        tester.AddOptionalInputEdge<MLFloat16>();
      } else {
        tester.AddInput<MLFloat16>("weight", weights_dims, ToFloat16(weights_data), is_weights_constant);
      }
      tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("input", input_dims, input_data);
      if (weights_data.empty()) {
        tester.AddOptionalInputEdge<float>();
      } else {
        tester.AddInput<float>("weight", weights_dims, weights_data, is_weights_constant);
      }
      tester.AddInput<float>("bias", bias_dims, bias_data);
      tester.AddOutput<float>("output", output_dims, output_data);
    }

    std::vector<int64_t> mask_index_dims_1 = {batch_size};
    std::vector<int64_t> mask_index_dims_2 = {2 * batch_size};
    std::vector<int64_t> mask_index_dims_3 = {batch_size, total_sequence_length};
    std::vector<int64_t> mask_index_dims_4 = {batch_size, 1};
    std::vector<int64_t> mask_index_dims_5 = {batch_size, sequence_length, total_sequence_length};
    std::vector<int64_t> mask_index_dims_6 = {batch_size, 1, max_sequence_length, max_sequence_length};
    std::vector<int64_t> mask_index_dims;
    switch (mask_index_type) {
      case kMaskIndexEnd:
        mask_index_dims = mask_index_dims_1;
        break;
      case kMaskIndexEndAndStart:
        mask_index_dims = mask_index_dims_2;
        break;
      case kMaskRaw:
        mask_index_dims = mask_index_dims_3;
        break;
      case kMaskDummy:
        mask_index_dims = mask_index_dims_4;
        break;
      case kMask3D:
        mask_index_dims = mask_index_dims_5;
        break;
      case kMask4D:
        mask_index_dims = mask_index_dims_6;
        break;
      default:
        assert(0);  // shall not reach here.
        break;
    }
    if (mask_index_data.size() > 0) {  // mask index is optional.
      tester.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);
    } else {
      tester.AddOptionalInputEdge<int32_t>();
    }

    std::vector<int64_t> past_dims = {2, batch_size, number_of_heads, past_sequence_length, head_size};
    std::vector<int64_t> present_dims = {2, batch_size, number_of_heads, total_sequence_length, head_size};
    if (use_past_state) {
      if (use_float16) {
        if (past_sequence_length > 0) {
          tester.AddInput<MLFloat16>("past", past_dims, ToFloat16(*past_data));
        }
        tester.AddOutput<MLFloat16>("present", present_dims, ToFloat16(*present_data));
      } else {
        if (past_sequence_length > 0) {
          tester.AddInput<float>("past", past_dims, *past_data);
        }
        tester.AddOutput<float>("present", present_dims, *present_data);
      }
    } else {
      if (use_float16) {
        tester.AddOptionalInputEdge<MLFloat16>();
      } else {
        tester.AddOptionalInputEdge<float>();
      }
    }

    std::vector<int64_t> extra_add_data_dims = {batch_size, number_of_heads, sequence_length, sequence_length};
    if (extra_add_data.size() > 0) {
      if (use_float16) {
        tester.AddInput<MLFloat16>("extra_add_qk", extra_add_data_dims, ToFloat16(extra_add_data));
      } else {
        tester.AddInput<float>("extra_add_qk", extra_add_data_dims, extra_add_data);
      }
    } else {
      if (use_float16) {
        tester.AddOptionalInputEdge<MLFloat16>();
      } else {
        tester.AddOptionalInputEdge<float>();
      }
    }

    std::vector<int64_t> key_dims = {batch_size, kv_sequence_length, hidden_size};
    std::vector<int64_t> value_dims = {batch_size, kv_sequence_length, v_hidden_size};
    if (use_float16) {
      if (key_data != nullptr) {
        tester.AddInput<MLFloat16>("key", key_dims, ToFloat16(*key_data));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }
      if (value_data != nullptr) {
        tester.AddInput<MLFloat16>("value", value_dims, ToFloat16(*value_data));
      } else {
        tester.AddOptionalInputEdge<MLFloat16>();
      }
    } else {
      if (key_data != nullptr) {
        tester.AddInput<float>("key", key_dims, *key_data);
      } else {
        tester.AddOptionalInputEdge<float>();
      }
      if (value_data != nullptr) {
        tester.AddInput<float>("value", value_dims, *value_data);
      } else {
        tester.AddOptionalInputEdge<float>();
      }
    }

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_rocm) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultRocmExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

static void RunAttentionTest(
    const std::vector<float>& input_data,         // input:      [batch_size, sequence_length, hidden_size]
    const std::vector<float>& weights_data,       // weights:    [hidden_size, 3 * hidden_size]
    const std::vector<float>& bias_data,          // bias:       [3 * hidden_size]
    const std::vector<int32_t>& mask_index_data,  // mask_index
    const std::vector<float>& output_data,        // output:     [batch_size, sequence_length, hidden_size]
    int batch_size,
    int sequence_length,
    int hidden_size,
    int number_of_heads,
    bool use_float16 = false,
    bool is_unidirectional = false,
    bool use_past_state = false,
    int past_sequence_length = 0,
    const std::vector<float>* past_data = nullptr,
    const std::vector<float>* present_data = nullptr,
    MaskIndexType mask_index_type = kMaskIndexEnd,
    int input_hidden_size = 0,
    int max_sequence_length = 0,
    const bool disable_cpu = false,
    const bool disable_cuda = false,
    const bool disable_rocm = false,
    const std::vector<int32_t> qkv_sizes = {},
    const std::vector<float>& extra_add_data = {},
    int kv_sequence_length = 0,
    const std::vector<float>* key_data = nullptr,
    const std::vector<float>* value_data = nullptr) {
  RunAttentionTest(input_data, weights_data, false, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length,
                   past_data, present_data, mask_index_type, input_hidden_size, max_sequence_length,
                   disable_cpu, disable_cuda, disable_rocm, qkv_sizes, extra_add_data,
                   kv_sequence_length, key_data, value_data);
  RunAttentionTest(input_data, weights_data, true, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length,
                   past_data, present_data, mask_index_type, input_hidden_size, max_sequence_length,
                   disable_cpu, disable_cuda, disable_rocm, qkv_sizes, extra_add_data,
                   kv_sequence_length, key_data, value_data);
}

TEST(AttentionTest, AttentionBatch1) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionBatch1WithQKVAttr1) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<int32_t> qkv_sizes = {
      6, 6, 4};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,

      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f,

      0.3f, 0.2f, 4.0f, 2.2f, 2.4f, 3.3f, 2.1f, 4.2f, 0.5f, 0.1f, 0.4f, 1.6f,
      0.4f, 0.8f, 0.9f, 0.1f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f,
      0.5f, 0.7f, 0.2f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      3.1967618465423584f, 0.51903456449508667f, 0.63051539659500122f, 2.9394614696502686f,
      0.65332180261611938f, 1.000949501991272f, 0.74175024032592773f, 2.8231701850891113f};

  constexpr bool disable_rocm = true;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   false, false, false, 0, nullptr, nullptr, kMaskIndexEnd, 0,
                   0, false, false, disable_rocm, qkv_sizes);
}

TEST(AttentionTest, AttentionBatch1WithQKVAttr2) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      -0.031707365f, 0.053643607f, 0.057394292f, -0.019800574f, 0.075466447f, -0.0034214978f, 0.012995008f, -0.019587509f};

  std::vector<int32_t> qkv_sizes = {
      6, 6, 2};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,

      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f,

      0.3f, 0.2f, 4.0f, 2.2f, 2.4f, 3.3f, 2.1f, 4.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f,
      0.5f, 0.7f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      0.64932525157928467f, 0.79390722513198853f, 0.64932847023010254f, 0.79375863075256348f};

  constexpr bool disable_rocm = true;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   false, false, false, 0, nullptr, nullptr, kMaskIndexEnd, 0,
                   0, false, false, disable_rocm, qkv_sizes);
}

TEST(AttentionTest, AttentionBatch1ExtraAdd) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<int32_t> qkv_sizes = {};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> extra_add_qk = {
      0.2f, -0.1f, 0.4f, 2.5f, 1.6f, -1.1f, 0.4f, -2.5f};

  std::vector<float> output_data = {
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f};

  constexpr bool disable_cpu = false;
  constexpr bool disable_cuda = false;
  constexpr bool disable_rocm = false;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   false, false, false, 0, nullptr, nullptr, kMaskIndexEnd, 0,
                   0, disable_cpu, disable_cuda, disable_rocm, qkv_sizes, extra_add_qk);
}

TEST(AttentionTest, AttentionBatch2ExtraAdd) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<int32_t> qkv_sizes = {};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f,
      0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L, 2L};

  std::vector<float> extra_add_qk = {
      0.2f, -0.1f, 0.4f, 2.5f, 1.6f, -1.1f, 0.4f, -2.5f,
      0.2f, -0.1f, 0.4f, 2.5f, 1.6f, -1.1f, 0.4f, -2.5f};

  std::vector<float> output_data = {
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f,
      4.066014289855957f, 0.068997815251350403f, 4.25f, 5.6499996185302734f,
      -1.8799558877944946f, 0.32488855719566345f, 4.25f, 5.6499996185302734f};

  constexpr bool disable_cpu = false;
  constexpr bool disable_cuda = false;
  constexpr bool disable_rocm = false;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   false, false, false, 0, nullptr, nullptr, kMaskIndexEnd, 0,
                   0, disable_cpu, disable_cuda, disable_rocm, qkv_sizes, extra_add_qk);
}

TEST(AttentionTest, AttentionBatch1_Float16) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L};

  std::vector<float> output_data = {
      3.154296875, 0.1082763671875, 4.25, 5.6484375,
      3.970703125, 0.072998046875, 4.25, 5.6484375};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, true);
}

TEST(AttentionTest, AttentionBatch2) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2L, 2L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionMaskPartialSequence) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index < sequence_length
  std::vector<int32_t> mask_index_data = {1L};

  std::vector<float> output_data = {
      8.6899995803833008f, -0.13000002503395081f, 4.25f, 5.6499996185302734f,
      8.6899995803833008f, -0.13000002503395081f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionMaskExceedSequence) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index > sequence_length
  std::vector<int32_t> mask_index_data = {3L};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionNoMaskIndex) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads);
}

TEST(AttentionTest, AttentionUnidirectional) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.091099896f, -0.018294459f, -0.36594841f, 0.28410032f,
      -0.12125026f, -0.0066160089f, 0.38809127f, -0.22455512f};

  std::vector<float> weight_data = {
      -0.2659236192703247f,
      0.02789675071835518f,
      0.07280516624450684f,
      0.050951678305864334f,
      0.020417947322130203f,
      -0.04751841351389885f,
      0.043815530836582184f,
      0.006015353370457888f,
      -0.11496957391500473f,
      -0.1773347705602646f,
      0.30928605794906616f,
      0.005648412741720676f,

      0.08960387855768204f,
      -0.27270448207855225f,
      0.14847396314144135f,
      -0.17960812151432037f,
      0.01788954995572567f,
      0.09993876516819f,
      0.03943513706326485f,
      -0.02484400011599064f,
      -0.12958766520023346f,
      0.220433309674263f,
      0.1720484346151352f,
      0.22024005651474f,

      0.059368450194597244f,
      0.1710093915462494f,
      -0.3967452347278595f,
      -0.1591450721025467f,
      0.1446179747581482f,
      -0.20505407452583313f,
      0.12749597430229187f,
      0.32139700651168823f,
      0.139958456158638f,
      -0.10619817674160004f,
      0.04528557509183884f,
      0.045598603785037994f,

      -0.007152545265853405f,
      0.109454445540905f,
      -0.1582530289888382f,
      -0.2646341919898987f,
      0.0920850858092308f,
      0.0701494812965393f,
      -0.19062495231628418f,
      -0.24360455572605133f,
      -0.09368397295475006f,
      0.07878211885690689f,
      0.2973634898662567f,
      0.11210034042596817f};

  std::vector<float> bias_data = {
      -0.0540979839861393f,
      -0.06444740295410156f,
      0.03112877532839775f,
      -0.08288222551345825f,
      0.07840359210968018f,
      0.039143580943346024f,
      -0.45591455698013306f,
      -0.11876055598258972f,
      0.3670335114002228f,
      0.028461361303925514f,
      -0.08913630992174149f,
      0.28048714995384216f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.28109729f, 0.069518551f, 0.0038009658f, 0.29213354f, 0.3692801f, 0.029495837f, -0.084964074f, 0.28169215f};

  bool is_unidirectional = true;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional);
}

TEST(AttentionTest, AttentionEmptyPastState) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.091099896f, -0.018294459f, -0.36594841f, 0.28410032f,
      -0.12125026f, -0.0066160089f, 0.38809127f, -0.22455512f};

  std::vector<float> weight_data = {
      -0.2659236192703247f,
      0.02789675071835518f,
      0.07280516624450684f,
      0.050951678305864334f,
      0.020417947322130203f,
      -0.04751841351389885f,
      0.043815530836582184f,
      0.006015353370457888f,
      -0.11496957391500473f,
      -0.1773347705602646f,
      0.30928605794906616f,
      0.005648412741720676f,

      0.08960387855768204f,
      -0.27270448207855225f,
      0.14847396314144135f,
      -0.17960812151432037f,
      0.01788954995572567f,
      0.09993876516819f,
      0.03943513706326485f,
      -0.02484400011599064f,
      -0.12958766520023346f,
      0.220433309674263f,
      0.1720484346151352f,
      0.22024005651474f,

      0.059368450194597244f,
      0.1710093915462494f,
      -0.3967452347278595f,
      -0.1591450721025467f,
      0.1446179747581482f,
      -0.20505407452583313f,
      0.12749597430229187f,
      0.32139700651168823f,
      0.139958456158638f,
      -0.10619817674160004f,
      0.04528557509183884f,
      0.045598603785037994f,

      -0.007152545265853405f,
      0.109454445540905f,
      -0.1582530289888382f,
      -0.2646341919898987f,
      0.0920850858092308f,
      0.0701494812965393f,
      -0.19062495231628418f,
      -0.24360455572605133f,
      -0.09368397295475006f,
      0.07878211885690689f,
      0.2973634898662567f,
      0.11210034042596817f};

  std::vector<float> bias_data = {
      -0.0540979839861393f,
      -0.06444740295410156f,
      0.03112877532839775f,
      -0.08288222551345825f,
      0.07840359210968018f,
      0.039143580943346024f,
      -0.45591455698013306f,
      -0.11876055598258972f,
      0.3670335114002228f,
      0.028461361303925514f,
      -0.08913630992174149f,
      0.28048714995384216f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.28109729f, 0.069518551f, 0.0038009658f, 0.29213354f, 0.3692801f, 0.029495837f, -0.084964074f, 0.28169215f};

  std::vector<float> past_data = {};

  std::vector<float> present_data = {
      0.053175069391727448f, 0.12795503437519073f, 0.11125634610652924f, -0.0510881207883358f, -0.55345797538757324f, -0.3045809268951416f, -0.36920222640037537f, 0.060108467936515808f, 0.28109729290008545f, 0.069518551230430603f, 0.45718482136726379f, -0.010400654748082161f, 0.0038009658455848694f, 0.29213353991508484f, -0.17697516083717346f, 0.27086889743804932f};

  bool is_unidirectional = true;
  bool use_past_state = true;
  int past_sequence_length = 0;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                   use_past_state, past_sequence_length, &past_data, &present_data);
}

TEST(AttentionTest, AttentionPastStateBatch1) {
  int batch_size = 1;
  int sequence_length = 1;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      -0.019333266f, -0.21813886f, 0.16212955f, -0.015626367f};

  std::vector<float> weight_data = {
      -0.4738484025001526f,
      -0.2613658607006073f,
      -0.0978037416934967f,
      -0.34988933801651f,
      0.2243240624666214f,
      -0.0429205559194088f,
      0.418695330619812f,
      0.17441125214099884f,
      -0.18825532495975494f,
      0.18357256054878235f,
      -0.5806483626365662f,
      -0.02251487597823143f,

      0.08742205798625946f,
      0.14734269678592682f,
      0.2387014478445053f,
      0.2884027063846588f,
      0.6490834355354309f,
      0.16965825855731964f,
      -0.06346885114908218f,
      0.4073973298072815f,
      -0.03070945478975773f,
      0.4110257923603058f,
      0.07896808534860611f,
      0.16783113777637482f,

      0.0038893644232302904f,
      0.06946629285812378f,
      0.36680519580841064f,
      -0.07261059433221817f,
      -0.14960581064224243f,
      0.020944256335496902f,
      -0.09378612786531448f,
      -0.1336742341518402f,
      0.06061394885182381f,
      0.2205914407968521f,
      -0.03519909828901291f,
      -0.18405692279338837f,

      0.22149960696697235f,
      -0.1884360909461975f,
      -0.014074507169425488f,
      0.4252440333366394f,
      0.24987126886844635f,
      -0.31396418809890747f,
      0.14036843180656433f,
      0.2854192554950714f,
      0.09709841012954712f,
      0.09935075044631958f,
      -0.012154420837759972f,
      0.2575816512107849f};

  std::vector<float> bias_data = {
      0.4803391396999359f,
      -0.5254325866699219f,
      -0.42926454544067383f,
      -0.2059524953365326f,
      -0.12773379683494568f,
      -0.09542735666036606f,
      -0.35286077857017517f,
      -0.07646317780017853f,
      -0.04590314254164696f,
      -0.03752850368618965f,
      -0.013764488510787487f,
      -0.18478283286094666f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.20141591f, 0.43005896f, 0.35745093f, 0.19957167f};

  std::vector<float> past_data = {
      0.55445826f, 0.10127074f, 0.71770734f, 0.15915526f, 0.13913247f, 0.77447522f, 0.66044068f, 0.27559045f, 0.35731629f, 0.62033528f, 0.24354559f, 0.22859341f,
      0.45075402f, 0.85365993f, 0.097346395f, 0.28859729f, 0.26926181f, 0.65922296f, 0.8177433f, 0.4212271f, 0.34352475f, 0.059609573f, 0.46556228f, 0.7226882f};

  std::vector<float> present_data = {
      0.55445826f, 0.10127074f, 0.71770734f, 0.15915526f, 0.13913247f, 0.77447522f, -0.30182117f, -0.12330482f, 0.66044068f, 0.27559045f, 0.35731629f, 0.62033528f, 0.24354559f, 0.22859341f, -0.36450946f, -0.19483691f,
      0.45075402f, 0.85365993f, 0.097346395f, 0.28859729f, 0.26926181f, 0.65922296f, -0.027254611f, -0.096526355f, 0.8177433f, 0.4212271f, 0.34352475f, 0.059609573f, 0.46556228f, 0.7226882f, -0.025281552f, -0.25482416f};

  bool is_unidirectional = true;
  bool use_past_state = true;
  int past_sequence_length = 3;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                   use_past_state, past_sequence_length, &past_data, &present_data);
}

TEST(AttentionTest, AttentionPastStateBatch2) {
  int batch_size = 2;
  int sequence_length = 1;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      -0.10902753f, 0.0041178204f, 0.1871525f, -0.20399982f,
      0.027207348f, -0.25321805f, 0.12869114f, 0.023136809f};

  std::vector<float> weight_data = {
      -0.4738484025001526f,
      -0.2613658607006073f,
      -0.0978037416934967f,
      -0.34988933801651f,
      0.2243240624666214f,
      -0.0429205559194088f,
      0.418695330619812f,
      0.17441125214099884f,
      -0.18825532495975494f,
      0.18357256054878235f,
      -0.5806483626365662f,
      -0.02251487597823143f,

      0.08742205798625946f,
      0.14734269678592682f,
      0.2387014478445053f,
      0.2884027063846588f,
      0.6490834355354309f,
      0.16965825855731964f,
      -0.06346885114908218f,
      0.4073973298072815f,
      -0.03070945478975773f,
      0.4110257923603058f,
      0.07896808534860611f,
      0.16783113777637482f,

      0.0038893644232302904f,
      0.06946629285812378f,
      0.36680519580841064f,
      -0.07261059433221817f,
      -0.14960581064224243f,
      0.020944256335496902f,
      -0.09378612786531448f,
      -0.1336742341518402f,
      0.06061394885182381f,
      0.2205914407968521f,
      -0.03519909828901291f,
      -0.18405692279338837f,

      0.22149960696697235f,
      -0.1884360909461975f,
      -0.014074507169425488f,
      0.4252440333366394f,
      0.24987126886844635f,
      -0.31396418809890747f,
      0.14036843180656433f,
      0.2854192554950714f,
      0.09709841012954712f,
      0.09935075044631958f,
      -0.012154420837759972f,
      0.2575816512107849f};

  std::vector<float> bias_data = {
      0.4803391396999359f,
      -0.5254325866699219f,
      -0.42926454544067383f,
      -0.2059524953365326f,
      -0.12773379683494568f,
      -0.09542735666036606f,
      -0.35286077857017517f,
      -0.07646317780017853f,
      -0.04590314254164696f,
      -0.03752850368618965f,
      -0.013764488510787487f,
      -0.18478283286094666f};

  // No mask_index
  std::vector<int32_t> mask_index_data = {};

  std::vector<float> output_data = {
      0.14902574f, 0.62273371f, 0.43022552f, 0.12759127f,
      0.26993567f, 0.23553593f, 0.43190649f, 0.086044826f};

  std::vector<float> past_data = {
      0.42028648f, 0.55855948f, 0.044569403f, 0.76525789f, 0.13962431f, 0.40977913f, 0.36911047f, 0.83399564f, 0.36905321f, 0.91414654f, 0.17300875f, 0.78793788f,
      0.10279467f, 0.80501258f, 0.089550517f, 0.85371113f, 0.61801594f, 0.91222942f, 0.88626182f, 0.069776468f, 0.10591964f, 0.84836882f, 0.83520192f, 0.0098680854f,
      0.3113814f, 0.63999802f, 0.28603253f, 0.98899829f, 0.044405211f, 0.95105386f, 0.81278932f, 0.63969064f, 0.14494057f, 0.11349615f, 0.87086016f, 0.20983537f,
      0.35107401f, 0.90144604f, 0.68950737f, 0.18928574f, 0.18029204f, 0.074517399f, 0.70763874f, 0.48440042f, 0.58114725f, 0.1048766f, 0.73694098f, 0.17766342f};

  std::vector<float> present_data = {
      0.42028648f, 0.55855948f, 0.044569403f, 0.76525789f, 0.13962431f, 0.40977913f, -0.22849128f, -0.022080801f, 0.36911047f, 0.83399564f, 0.36905321f, 0.91414654f, 0.17300875f, 0.78793788f, -0.4449589f, -0.17704415f, 0.10279467f, 0.80501258f, 0.089550517f, 0.85371113f, 0.61801594f, 0.91222942f, -0.2994619f, -0.14412443f, 0.88626182f, 0.069776468f, 0.10591964f, 0.84836882f, 0.83520192f, 0.0098680854f, -0.33421949f, -0.18547727f,
      0.3113814f, 0.63999802f, 0.28603253f, 0.98899829f, 0.044405211f, 0.95105386f, -0.033968594f, -0.034833729f, 0.81278932f, 0.63969064f, 0.14494057f, 0.11349615f, 0.87086016f, 0.20983537f, 0.045759238f, -0.26863033f, 0.35107401f, 0.90144604f, 0.68950737f, 0.18928574f, 0.18029204f, 0.074517399f, -0.033201858f, -0.10592631f, 0.70763874f, 0.48440042f, 0.58114725f, 0.1048766f, 0.73694098f, 0.17766342f, -0.054369561f, -0.24562015f};

  bool is_unidirectional = true;
  bool use_past_state = true;
  int past_sequence_length = 3;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                   use_past_state, past_sequence_length, &past_data, &present_data);
}

TEST(AttentionTest, AttentionPastStateBatch2WithPadding) {
  int batch_size = 2;
  int sequence_length = 1;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      -0.10902753f, 0.0041178204f, 0.1871525f, -0.20399982f,
      0.027207348f, -0.25321805f, 0.12869114f, 0.023136809f};

  std::vector<float> weight_data = {
      -0.4738484025001526f,
      -0.2613658607006073f,
      -0.0978037416934967f,
      -0.34988933801651f,
      0.2243240624666214f,
      -0.0429205559194088f,
      0.418695330619812f,
      0.17441125214099884f,
      -0.18825532495975494f,
      0.18357256054878235f,
      -0.5806483626365662f,
      -0.02251487597823143f,

      0.08742205798625946f,
      0.14734269678592682f,
      0.2387014478445053f,
      0.2884027063846588f,
      0.6490834355354309f,
      0.16965825855731964f,
      -0.06346885114908218f,
      0.4073973298072815f,
      -0.03070945478975773f,
      0.4110257923603058f,
      0.07896808534860611f,
      0.16783113777637482f,

      0.0038893644232302904f,
      0.06946629285812378f,
      0.36680519580841064f,
      -0.07261059433221817f,
      -0.14960581064224243f,
      0.020944256335496902f,
      -0.09378612786531448f,
      -0.1336742341518402f,
      0.06061394885182381f,
      0.2205914407968521f,
      -0.03519909828901291f,
      -0.18405692279338837f,

      0.22149960696697235f,
      -0.1884360909461975f,
      -0.014074507169425488f,
      0.4252440333366394f,
      0.24987126886844635f,
      -0.31396418809890747f,
      0.14036843180656433f,
      0.2854192554950714f,
      0.09709841012954712f,
      0.09935075044631958f,
      -0.012154420837759972f,
      0.2575816512107849f};

  std::vector<float> bias_data = {
      0.4803391396999359f,
      -0.5254325866699219f,
      -0.42926454544067383f,
      -0.2059524953365326f,
      -0.12773379683494568f,
      -0.09542735666036606f,
      -0.35286077857017517f,
      -0.07646317780017853f,
      -0.04590314254164696f,
      -0.03752850368618965f,
      -0.013764488510787487f,
      -0.18478283286094666f};

  // One sequence has both left padding and right padding
  std::vector<int32_t> mask_index_data = {4, 3, 0, 2};

  std::vector<float> output_data = {
      0.14902574f, 0.62273371f, 0.43022552f, 0.12759127f,
      0.18029204f, 0.07451740f, 0.73694098f, 0.17766341f};

  std::vector<float> past_data = {
      0.42028648f, 0.55855948f, 0.044569403f, 0.76525789f, 0.13962431f, 0.40977913f, 0.36911047f, 0.83399564f, 0.36905321f, 0.91414654f, 0.17300875f, 0.78793788f,
      0.10279467f, 0.80501258f, 0.089550517f, 0.85371113f, 0.61801594f, 0.91222942f, 0.88626182f, 0.069776468f, 0.10591964f, 0.84836882f, 0.83520192f, 0.0098680854f,
      0.3113814f, 0.63999802f, 0.28603253f, 0.98899829f, 0.044405211f, 0.95105386f, 0.81278932f, 0.63969064f, 0.14494057f, 0.11349615f, 0.87086016f, 0.20983537f,
      0.35107401f, 0.90144604f, 0.68950737f, 0.18928574f, 0.18029204f, 0.074517399f, 0.70763874f, 0.48440042f, 0.58114725f, 0.1048766f, 0.73694098f, 0.17766342f};

  std::vector<float> present_data = {
      0.42028648f, 0.55855948f, 0.044569403f, 0.76525789f, 0.13962431f, 0.40977913f, -0.22849128f, -0.022080801f, 0.36911047f, 0.83399564f, 0.36905321f, 0.91414654f, 0.17300875f, 0.78793788f, -0.4449589f, -0.17704415f, 0.10279467f, 0.80501258f, 0.089550517f, 0.85371113f, 0.61801594f, 0.91222942f, -0.2994619f, -0.14412443f, 0.88626182f, 0.069776468f, 0.10591964f, 0.84836882f, 0.83520192f, 0.0098680854f, -0.33421949f, -0.18547727f,
      0.3113814f, 0.63999802f, 0.28603253f, 0.98899829f, 0.044405211f, 0.95105386f, -0.033968594f, -0.034833729f, 0.81278932f, 0.63969064f, 0.14494057f, 0.11349615f, 0.87086016f, 0.20983537f, 0.045759238f, -0.26863033f, 0.35107401f, 0.90144604f, 0.68950737f, 0.18928574f, 0.18029204f, 0.074517399f, -0.033201858f, -0.10592631f, 0.70763874f, 0.48440042f, 0.58114725f, 0.1048766f, 0.73694098f, 0.17766342f, -0.054369561f, -0.24562015f};

  bool is_unidirectional = true;
  bool use_past_state = true;
  int past_sequence_length = 3;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads, false, is_unidirectional,
                   use_past_state, past_sequence_length, &past_data, &present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionBatch2MaskIndex2) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {2, 2, 0, 0};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionRightPaddingMaskIndex2) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask_index < sequence_length
  std::vector<int32_t> mask_index_data = {1, 0};

  std::vector<float> output_data = {
      8.6899995803833008f, -0.13000002503395081f, 4.25f, 5.6499996185302734f,
      8.6899995803833008f, -0.13000002503395081f, 4.2499995231628418f, 5.6499991416931152f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionLeftPaddingMaskIndex2) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {2, 1};

  std::vector<float> output_data = {
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionBatch2LeftPaddingMaskIndex2) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {2, 2, 1, 0};

  std::vector<float> output_data = {
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.14959716796875f, 0.10843672603368759f, 4.25f, 5.65f,
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, Attention3DMask) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test 3D mask BxSxS*
  std::vector<int32_t> mask_index_data = {
      0, 1,
      0, 1,
      1, 1,
      1, 1};

  std::vector<float> output_data = {
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.14959716796875f, 0.10843672603368759f, 4.25f, 5.65f,
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMask3D);
}

TEST(AttentionTest, AttentionBatch2AttentionMask) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {0, 1, 1, 1};

  std::vector<float> output_data = {
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.14959716796875f, 0.10843672603368759f, 4.25f, 5.65f,
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskRaw);
}


TEST(AttentionTest, FlashAttention_NoPadding) {
  int batch_size = 2;
  int sequence_length = 4;
  int hidden_size = 16;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      1.33886719f, 0.20520020f, -1.68750000f, 0.51025391f, -0.34570312f, 0.64550781f, 1.57324219f, 0.35180664f,
      1.12988281f, 0.00979614f, -0.31494141f, 0.48095703f, -1.23046875f, 1.08984375f, 1.05664062f, -0.97460938f,
      -0.32153320f, -1.61035156f, 1.93164062f, -0.54931641f, -1.98242188f, 0.26684570f, 0.90380859f, 0.16040039f,
      -0.09008789f, -0.07421875f, 0.05511475f, 0.61914062f, 0.71630859f, 0.46484375f, 0.79199219f, -2.19335938f,
      0.27563477f, -1.07617188f, -0.87255859f, 1.81835938f, -0.44458008f, 0.25073242f, 0.48217773f, -0.30615234f,
      -2.73437500f, -0.53417969f, -0.62353516f, -1.28613281f, -1.33886719f, -0.67187500f, 1.21386719f, -1.01953125f,
      -0.10882568f, 1.56542969f, 0.98486328f, -1.22558594f, -1.17773438f, 0.42822266f, 0.59667969f, 0.04766846f,
      1.56152344f, 1.13183594f, -0.10437012f, -0.55859375f, 1.17675781f, 0.17395020f, 1.21093750f, 1.43945312f,
      1.76855469f, -1.68261719f, 0.17419434f, 0.99853516f, 1.53320312f, -0.17883301f, -0.34741211f, -1.96777344f,
      1.10644531f, -0.74902344f, 2.35937500f, 0.06817627f, -2.93750000f, -1.37011719f, -0.50781250f, 0.72509766f,
      0.35009766f, 0.10119629f, 1.51562500f, -1.16113281f, 0.50292969f, 0.43579102f, -0.78955078f, 0.91455078f,
      -0.33007812f, 0.02938843f, 0.30541992f, 1.55078125f, -2.02148438f, 0.35522461f, -0.83105469f, -0.66650391f,
      -1.88183594f, -0.09906006f, -0.43896484f, 0.12017822f, 1.59082031f, 0.16809082f, 0.80322266f, 0.04754639f,
      0.99560547f, 2.17968750f, -0.45141602f, 0.90087891f, -1.08593750f, 0.22705078f, 0.80175781f, 0.90429688f,
      -1.94433594f, -1.42187500f, 0.80126953f, 0.91308594f, -0.58496094f, 0.45898438f, -0.14245605f, 0.16491699f,
      0.01141357f, 0.84716797f, -0.41674805f, -0.53222656f, -0.86279297f, 0.30029297f, 0.24365234f, 1.33007812f};

  std::vector<float> weight_data = {
  0.06982422f, -0.17614746f, 0.16064453f, -0.01165009f, 0.07336426f, 0.03616333f, -0.18896484f, -0.13427734f,
  0.06732178f, 0.23168945f, 0.03628540f, -0.05310059f, -0.00230598f, -0.23022461f, -0.15026855f, -0.21069336f,
  0.15808105f, 0.23498535f, 0.21118164f, -0.20397949f, 0.01976013f, -0.19128418f, -0.08508301f, 0.01477051f,
  -0.23901367f, 0.02339172f, -0.10015869f, -0.02317810f, -0.08056641f, -0.24853516f, -0.20263672f, 0.13952637f,
  0.11090088f, 0.01690674f, -0.18505859f, -0.18884277f, -0.12359619f, 0.09295654f, -0.02415466f, -0.03570557f,
  -0.04028320f, 0.09210205f, 0.16503906f, -0.04162598f, -0.18164062f, 0.01148224f, 0.16638184f, 0.15686035f,
  -0.21691895f, 0.15661621f, -0.19824219f, -0.03713989f, -0.15759277f, 0.22692871f, -0.04885864f, 0.10296631f,
  -0.04849243f, -0.05786133f, 0.13391113f, 0.10900879f, -0.01884460f, 0.01785278f, 0.19262695f, -0.14306641f,
  -0.08142090f, -0.07049561f, 0.23205566f, -0.08129883f, -0.00161171f, -0.07818604f, 0.20385742f, -0.12707520f,
  -0.15502930f, -0.12658691f, 0.14562988f, 0.03576660f, 0.15466309f, -0.09234619f, 0.01782227f, 0.23742676f,
  -0.09429932f, 0.02610779f, -0.24597168f, 0.10870361f, 0.08081055f, 0.24609375f, 0.15930176f, 0.02798462f,
   0.12512207f, -0.18054199f, -0.06204224f, -0.08282471f, -0.11669922f, 0.14013672f, 0.09875488f, -0.14208984f,
  0.00887299f, 0.03210449f, -0.07067871f, 0.23278809f, 0.12139893f, -0.10424805f, 0.11737061f, -0.06878662f,
  -0.10021973f, 0.10046387f, -0.07537842f, -0.24084473f, 0.00635910f, 0.09246826f, 0.10687256f, 0.08917236f,
  -0.09613037f, -0.13793945f, -0.02572632f, 0.09625244f, 0.24853516f, 0.22143555f, -0.22448730f, 0.24536133f,
   0.01716614f, 0.16650391f, 0.07385254f, -0.06188965f, -0.15576172f, 0.00595093f, 0.08038330f, -0.05541992f,
  0.15136719f, 0.19323730f, -0.20214844f, -0.17150879f, 0.09582520f, -0.08459473f, 0.02511597f, -0.24877930f,
  -0.21618652f, -0.15844727f, -0.19152832f, 0.12213135f, -0.10748291f, 0.20935059f, -0.03182983f, -0.09326172f,
  -0.13879395f, 0.07647705f, 0.21081543f, -0.02172852f, -0.02629089f, -0.01597595f, -0.18310547f, 0.14611816f,
  -0.08795166f, -0.06188965f, -0.10211182f, 0.21362305f, 0.18395996f, -0.14807129f, -0.07745361f, 0.12927246f,
  -0.19848633f, 0.06884766f, 0.18835449f, -0.10620117f, 0.01678467f, -0.15246582f, -0.13500977f, -0.18994141f,
  0.13098145f, 0.24145508f, 0.21276855f, -0.00587845f, 0.08172607f, -0.13610840f, 0.21484375f, -0.08837891f,
  -0.08508301f, -0.00902557f, -0.08868408f, -0.13891602f, 0.15368652f, 0.16210938f, 0.13452148f, -0.17468262f,
   0.18286133f, -0.06088257f, -0.09069824f, 0.19299316f, 0.00621796f, 0.10363770f, -0.08990479f, -0.19482422f,
  0.03909302f, 0.09295654f, -0.15637207f, -0.19079590f, -0.10717773f, -0.07946777f, -0.04238892f, -0.00128174f,
  -0.01683044f, -0.00362206f, -0.18286133f, -0.20349121f, -0.00975800f, 0.22802734f, 0.21630859f, -0.09545898f,
  0.21411133f, -0.03573608f, 0.23864746f, 0.03732300f, 0.20751953f, 0.02011108f, -0.06719971f, -0.17028809f,
  0.01774597f, 0.09790039f, -0.19177246f, 0.22961426f, 0.17089844f, -0.06030273f, -0.15405273f, 0.05746460f,
  0.08483887f, 0.18469238f, -0.04577637f, 0.05151367f, 0.19128418f, -0.15344238f, 0.08569336f, 0.23010254f,
  -0.14770508f, -0.17248535f, -0.00046682f, 0.05953979f, 0.06250000f, 0.07922363f, -0.12243652f, 0.17993164f,
  0.18566895f, 0.12152100f, -0.22973633f, -0.14575195f, 0.04092407f, 0.22290039f, -0.18811035f, -0.20861816f,
  -0.02673340f, -0.16235352f, 0.00653076f, -0.18957520f, -0.15026855f, -0.16662598f, 0.17211914f, -0.22143555f,
  -0.09912109f, 0.12060547f, -0.11578369f, 0.23925781f, -0.01065063f, 0.21777344f, 0.13000488f, 0.10327148f,
  0.10449219f, 0.16516113f, -0.16149902f, 0.07958984f, -0.17382812f, 0.11828613f, -0.04739380f, 0.17041016f,
  -0.13220215f, -0.04071045f, -0.03994751f, 0.23937988f, 0.08105469f, -0.15209961f, -0.20288086f, -0.03347778f,
  -0.07153320f, 0.14538574f, 0.03262329f, -0.07299805f, -0.18029785f, 0.23376465f, -0.20422363f, -0.06701660f,
  0.00343513f, -0.03097534f, 0.03335571f, -0.10906982f, 0.08288574f, -0.02381897f, 0.13708496f, 0.03414917f,
  -0.22900391f, 0.14929199f, 0.11181641f, -0.06481934f, -0.02731323f, 0.02947998f, 0.04864502f, 0.08038330f,
  0.05374146f, 0.17858887f, 0.07659912f, -0.23986816f, -0.22351074f, -0.12060547f, -0.16076660f, 0.16650391f,
   -0.12658691f, 0.09918213f, 0.14819336f, -0.23144531f, 0.09466553f, 0.22399902f, 0.07177734f, 0.09228516f,
  0.03479004f, 0.00276947f, -0.07177734f, 0.09051514f, 0.17309570f, 0.12524414f, -0.18725586f, -0.00495529f,
  0.06176758f, 0.06982422f, -0.03591919f, 0.19091797f, -0.17163086f, -0.10894775f, 0.03900146f, -0.10882568f,
  0.22937012f, -0.20507812f, 0.11370850f, -0.11016846f, -0.21520996f, 0.11730957f, 0.18322754f, 0.23107910f,
  -0.02180481f, 0.19030762f, -0.10058594f, -0.08660889f, 0.24645996f, -0.12237549f, -0.01232147f, -0.24047852f,
  -0.13916016f, -0.06237793f, 0.05062866f, 0.24938965f, 0.22534180f, 0.03454590f, -0.24829102f, -0.00137138f,
  0.11779785f, -0.17712402f, 0.19226074f, 0.24707031f, -0.18212891f, 0.12829590f, -0.13818359f, -0.15026855f,
  0.19274902f, 0.19372559f, 0.11511230f, -0.20373535f, -0.12274170f, -0.05865479f, 0.20910645f, -0.08435059f,
  -0.21911621f, 0.13549805f, -0.17639160f, -0.13281250f, 0.18774414f, -0.11175537f, 0.05133057f, -0.10791016f,
  -0.03219604f, -0.00830841f, -0.04135132f, 0.02661133f, 0.24060059f, -0.23498535f, 0.04092407f, -0.06604004f,
   0.24902344f, 0.16052246f, 0.09350586f, -0.24194336f, 0.17687988f, -0.12634277f, -0.13232422f, -0.00743866f,
  0.06671143f, 0.24060059f, -0.23840332f, -0.07409668f, 0.14343262f, -0.21606445f, 0.22875977f, -0.20446777f,
  -0.09484863f, 0.06762695f, -0.20031738f, -0.05169678f, 0.16931152f, 0.01776123f, 0.03906250f, 0.20581055f,
  -0.11962891f, -0.01544189f, -0.18872070f, -0.01165009f, -0.04556274f, 0.21020508f, 0.13159180f, -0.17297363f,
  -0.11566162f, 0.22009277f, 0.04339600f, 0.07080078f, 0.07958984f, -0.21838379f, 0.01718140f, 0.07873535f,
  0.15515137f, 0.01173401f, 0.09552002f, 0.03533936f, -0.04516602f, 0.05374146f, 0.10864258f, -0.18908691f,
  -0.10174561f, 0.19580078f, -0.01185608f, 0.00821686f, 0.03158569f, 0.21948242f, -0.10321045f, 0.03631592f,
  0.18066406f, -0.08312988f, 0.13293457f, 0.09045410f, -0.04382324f, -0.22412109f, -0.03988647f, 0.19433594f,
  -0.07141113f, 0.15625000f, 0.08032227f, -0.09069824f, 0.01383209f, -0.11352539f, 0.10455322f, -0.07006836f,
  0.12597656f, 0.23400879f, 0.10986328f, -0.14562988f, 0.24243164f, -0.04183960f, -0.21069336f, 0.14062500f,
  -0.18750000f, -0.24658203f, -0.14880371f, 0.22753906f, -0.05587769f, 0.18518066f, 0.10058594f, 0.09075928f,
  -0.17089844f, -0.23022461f, 0.09191895f, -0.22167969f, 0.10119629f, 0.09344482f, -0.24548340f, 0.16906738f,
  -0.05865479f, 0.15393066f, -0.13220215f, 0.19848633f, -0.13232422f, -0.24816895f, -0.15539551f, 0.23889160f,
  -0.18725586f, -0.08575439f, -0.18029785f, 0.04040527f, 0.10455322f, 0.14221191f, 0.01489258f, 0.14916992f,
  0.10552979f, 0.04141235f, 0.05511475f, 0.12988281f, -0.03799438f, 0.03308105f, -0.01216888f, -0.20483398f,
  -0.21643066f, -0.21948242f, -0.17614746f, 0.02754211f, 0.04318237f, -0.18298340f, 0.22619629f, 0.13684082f,
  0.22070312f, 0.14147949f, -0.02151489f, 0.02787781f, 0.16552734f, -0.15356445f, -0.12915039f, -0.24450684f,
  0.12561035f, 0.21069336f, 0.06628418f, 0.13293457f, 0.07897949f, 0.02349854f, 0.11749268f, -0.05865479f,
   0.15393066f, 0.16052246f, -0.09234619f, 0.06994629f, -0.07055664f, -0.16259766f, -0.22338867f, -0.22741699f,
  0.19506836f, -0.05749512f, 0.21154785f, -0.19665527f, 0.19152832f, -0.23779297f, -0.04934692f, 0.13879395f,
  0.11804199f, -0.10546875f, 0.07269287f, 0.19165039f, 0.09234619f, 0.14355469f, 0.00853729f, 0.13378906f,
  0.06787109f, -0.03176880f, -0.09411621f, 0.19543457f, 0.02069092f, -0.06439209f, -0.13037109f, 0.00778580f,
  0.11029053f, 0.17687988f, 0.21069336f, 0.22399902f, -0.15856934f, -0.18371582f, 0.12408447f, 0.14990234f,
  -0.23034668f, 0.11895752f, -0.01561737f, 0.13598633f, -0.07043457f, -0.09393311f, 0.19873047f, 0.04968262f,
  -0.08831787f, 0.02133179f, -0.21801758f, -0.09692383f, -0.12377930f, -0.07916260f, -0.12585449f, -0.14709473f,
  -0.14404297f, -0.04705811f, -0.11993408f, -0.24572754f, 0.15856934f, -0.12634277f, 0.01828003f, 0.21691895f,
  -0.09252930f, 0.04376221f, 0.07391357f, 0.13537598f, 0.21215820f, -0.07031250f, 0.20642090f, -0.02078247f,
  -0.10632324f, 0.22583008f, 0.08245850f, -0.15856934f, -0.01436615f, -0.06854248f, -0.10760498f, 0.11743164f,
  0.00469589f, 0.06945801f, 0.23193359f, -0.05236816f, 0.00703049f, -0.16247559f, 0.17626953f, 0.12353516f,
  -0.08862305f, 0.07128906f, 0.04345703f, -0.03350830f, 0.24365234f, 0.06146240f, -0.00362396f, -0.10803223f,
  -0.14062500f, -0.05798340f, 0.00224876f, 0.14953613f, -0.20837402f, 0.18249512f, 0.23950195f, -0.24804688f,
  -0.03479004f, -0.12915039f, 0.24914551f, -0.17700195f, 0.11389160f, 0.19213867f, -0.02809143f, 0.00966644f,
  0.00834656f, -0.10986328f, 0.20947266f, 0.21582031f, -0.23706055f, -0.09619141f, -0.05532837f, -0.07122803f,
  0.20251465f, 0.02970886f, -0.16455078f, 0.02488708f, 0.08026123f, -0.07238770f, 0.14916992f, -0.05032349f,
  0.06079102f, -0.21643066f, -0.01262665f, 0.02890015f, 0.23791504f, -0.10205078f, 0.15087891f, -0.23022461f,
  -0.21398926f, -0.14855957f, 0.20532227f, 0.03152466f, 0.21582031f, -0.10388184f, 0.21728516f, 0.14257812f,
  0.18444824f, 0.19384766f, 0.00561142f, -0.03024292f, -0.24255371f, 0.19335938f, 0.01824951f, -0.24560547f,
  -0.19104004f, 0.24890137f, -0.07897949f, -0.16394043f, -0.23583984f, 0.10034180f, -0.17211914f, -0.06549072f,
  0.11877441f, 0.14672852f, 0.10705566f, -0.07067871f, -0.09246826f, -0.05908203f, -0.17114258f, -0.11340332f,
  0.07373047f, 0.02824402f, 0.13134766f, 0.13342285f, 0.11706543f, -0.20178223f, -0.14050293f, 0.15832520f,
  0.19128418f, -0.11834717f, -0.10559082f, -0.04580688f, 0.19189453f, 0.10974121f, -0.00661469f, 0.22131348f,
  0.17114258f, -0.18798828f, 0.06372070f, -0.10510254f, 0.17480469f, 0.08953857f, -0.02033997f, -0.15673828f,
  0.01210785f, 0.15429688f, 0.24682617f, 0.06027222f, -0.04794312f, 0.18579102f, 0.11181641f, -0.05419922f,
  0.11230469f, -0.11291504f, 0.20166016f, -0.23229980f, -0.21984863f, 0.20751953f, -0.19738770f, 0.02136230f,
  0.18151855f, 0.10235596f, -0.10797119f, 0.17407227f, 0.21801758f, -0.12213135f, 0.20056152f, 0.17541504f,
  -0.00532913f, 0.21691895f, -0.02801514f, -0.06280518f, 0.17211914f, -0.06323242f, 0.21435547f, -0.24328613f,
  0.06085205f, -0.24487305f, -0.03384399f, -0.14697266f, 0.09222412f, -0.01487732f, -0.21813965f, 0.07672119f};

  std::vector<float> bias_data = {
  -0.16381836f, -0.22399902f, 0.13256836f, -0.24621582f, -0.07043457f, -0.00012034f, -0.04183960f, 0.17565918f,
  0.07720947f, -0.19482422f, -0.00689697f, 0.18908691f, -0.15856934f, -0.15246582f, -0.09948730f, 0.05960083f,
  0.08398438f, 0.04818726f, -0.03555298f, 0.15966797f, -0.09649658f, -0.02667236f, 0.13818359f, 0.06063843f,
  0.21057129f, 0.12042236f, -0.07482910f, -0.17614746f, 0.23242188f, 0.08190918f, -0.16467285f, 0.24218750f,
  0.23278809f, -0.05047607f, 0.07141113f, 0.09313965f, -0.23498535f, 0.22863770f, 0.09918213f, -0.04803467f,
  -0.19775391f, 0.03637695f, 0.15600586f, 0.10614014f, -0.21740723f, -0.08190918f, 0.15600586f, -0.21704102f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {4, 4};

  std::vector<float> output_data = {
      0.39868164f, -0.1973877f, 0.12426758f, 0.23986816f, -0.19348145f, 0.3034668f, -0.50927734f, -0.10803223f, 0.085205078f, 0.23901367f, -0.076660156f, 0.28344727f, -0.36694336f, 0.21472168f, 0.31811523f, -0.40112305f,
      0.43383789f, -0.28295898f, 0.15490723f, 0.24951172f, -0.25732422f, 0.26831055f, -0.58056641f, -0.089599609f, 0.085388184f, 0.26171875f, -0.019485474f, 0.27685547f, -0.3449707f, 0.087219238f, 0.38476562f, -0.35058594f,
      0.38427734f, -0.22607422f, 0.27416992f, 0.2019043f, -0.18334961f, 0.24353027f, -0.52880859f, -0.065246582f, -0.02784729f, 0.24609375f, 0.022644043f, 0.25268555f, -0.32055664f, -0.067932129f, 0.49731445f, -0.27490234f,
      0.42211914f, -0.19921875f, 0.14770508f, 0.20568848f, -0.23962402f, 0.2388916f, -0.57666016f, -0.12408447f, 0.0055160522f, 0.36645508f, -0.05090332f, 0.3684082f, -0.39379883f, 0.14404297f, 0.40039062f, -0.43554688f,

      0.55126953f, 0.0050125122f, -0.067016602f, 0.38427734f, 0.29541016f, -0.13244629f, 0.21716309f, -0.25317383f, -0.3684082f, -0.11907959f, -0.22937012f, 0.6640625f, -0.17224121f, 0.16564941f, -0.61230469f, -0.14086914f,
      0.35717773f, -0.21606445f, -0.18676758f, 0.34912109f, 0.26220703f, -0.081542969f, 0.33154297f, -0.29858398f, -0.39477539f, -0.21240234f, -0.32055664f, 0.61914062f, -0.096008301f, 0.20910645f, -0.61914062f, -0.21533203f,
      0.33398438f, -0.26074219f, -0.25195312f, 0.39257812f, 0.27441406f, -0.061798096f, 0.34082031f, -0.23571777f, -0.48828125f, -0.34423828f, -0.41455078f, 0.61279297f, -0.041748047f, 0.24462891f, -0.51904297f, -0.23754883f,
      0.47143555f, -0.1439209f, -0.19091797f, 0.38647461f, 0.23181152f, -0.10321045f, 0.28881836f, -0.25097656f, -0.52685547f, -0.27075195f, -0.35522461f, 0.62353516f, -0.11895752f, 0.22094727f, -0.44702148f, -0.2088623f};

  bool use_float16 = true;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  int input_hidden_size = 0;
  int max_sequence_length = 0;
  bool disable_cpu = true;
  bool disable_cuda = false;
  bool disable_rocm = true;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data,
                   kMaskIndexEnd, input_hidden_size, max_sequence_length, disable_cpu, disable_cuda, disable_rocm);
}


TEST(AttentionTest, FlashAttention_Padding) {
  int batch_size = 2;
  int sequence_length = 4;
  int hidden_size = 16;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      1.33886719f, 0.20520020f, -1.68750000f, 0.51025391f, -0.34570312f, 0.64550781f, 1.57324219f, 0.35180664f,
      1.12988281f, 0.00979614f, -0.31494141f, 0.48095703f, -1.23046875f, 1.08984375f, 1.05664062f, -0.97460938f,
      -0.32153320f, -1.61035156f, 1.93164062f, -0.54931641f, -1.98242188f, 0.26684570f, 0.90380859f, 0.16040039f,
      -0.09008789f, -0.07421875f, 0.05511475f, 0.61914062f, 0.71630859f, 0.46484375f, 0.79199219f, -2.19335938f,
      0.27563477f, -1.07617188f, -0.87255859f, 1.81835938f, -0.44458008f, 0.25073242f, 0.48217773f, -0.30615234f,
      -2.73437500f, -0.53417969f, -0.62353516f, -1.28613281f, -1.33886719f, -0.67187500f, 1.21386719f, -1.01953125f,
      -0.10882568f, 1.56542969f, 0.98486328f, -1.22558594f, -1.17773438f, 0.42822266f, 0.59667969f, 0.04766846f,
      1.56152344f, 1.13183594f, -0.10437012f, -0.55859375f, 1.17675781f, 0.17395020f, 1.21093750f, 1.43945312f,

      1.76855469f, -1.68261719f, 0.17419434f, 0.99853516f, 1.53320312f, -0.17883301f, -0.34741211f, -1.96777344f,
      1.10644531f, -0.74902344f, 2.35937500f, 0.06817627f, -2.93750000f, -1.37011719f, -0.50781250f, 0.72509766f,
      0.35009766f, 0.10119629f, 1.51562500f, -1.16113281f, 0.50292969f, 0.43579102f, -0.78955078f, 0.91455078f,
      -0.33007812f, 0.02938843f, 0.30541992f, 1.55078125f, -2.02148438f, 0.35522461f, -0.83105469f, -0.66650391f,
      -1.88183594f, -0.09906006f, -0.43896484f, 0.12017822f, 1.59082031f, 0.16809082f, 0.80322266f, 0.04754639f,
      0.99560547f, 2.17968750f, -0.45141602f, 0.90087891f, -1.08593750f, 0.22705078f, 0.80175781f, 0.90429688f,
      -1.94433594f, -1.42187500f, 0.80126953f, 0.91308594f, -0.58496094f, 0.45898438f, -0.14245605f, 0.16491699f,
      0.01141357f, 0.84716797f, -0.41674805f, -0.53222656f, -0.86279297f, 0.30029297f, 0.24365234f, 1.33007812f};

  std::vector<float> weight_data = {
      0.06982422f, -0.17614746f, 0.16064453f, -0.01165009f, 0.07336426f, 0.03616333f, -0.18896484f, -0.13427734f,
      0.06732178f, 0.23168945f, 0.03628540f, -0.05310059f, -0.00230598f, -0.23022461f, -0.15026855f, -0.21069336f,
      0.15808105f, 0.23498535f, 0.21118164f, -0.20397949f, 0.01976013f, -0.19128418f, -0.08508301f, 0.01477051f,
      -0.23901367f, 0.02339172f, -0.10015869f, -0.02317810f, -0.08056641f, -0.24853516f, -0.20263672f, 0.13952637f,
      0.11090088f, 0.01690674f, -0.18505859f, -0.18884277f, -0.12359619f, 0.09295654f, -0.02415466f, -0.03570557f,
      -0.04028320f, 0.09210205f, 0.16503906f, -0.04162598f, -0.18164062f, 0.01148224f, 0.16638184f, 0.15686035f,
      -0.21691895f, 0.15661621f, -0.19824219f, -0.03713989f, -0.15759277f, 0.22692871f, -0.04885864f, 0.10296631f,
      -0.04849243f, -0.05786133f, 0.13391113f, 0.10900879f, -0.01884460f, 0.01785278f, 0.19262695f, -0.14306641f,
      -0.08142090f, -0.07049561f, 0.23205566f, -0.08129883f, -0.00161171f, -0.07818604f, 0.20385742f, -0.12707520f,
      -0.15502930f, -0.12658691f, 0.14562988f, 0.03576660f, 0.15466309f, -0.09234619f, 0.01782227f, 0.23742676f,
      -0.09429932f, 0.02610779f, -0.24597168f, 0.10870361f, 0.08081055f, 0.24609375f, 0.15930176f, 0.02798462f,
      0.12512207f, -0.18054199f, -0.06204224f, -0.08282471f, -0.11669922f, 0.14013672f, 0.09875488f, -0.14208984f,
      0.00887299f, 0.03210449f, -0.07067871f, 0.23278809f, 0.12139893f, -0.10424805f, 0.11737061f, -0.06878662f,
      -0.10021973f, 0.10046387f, -0.07537842f, -0.24084473f, 0.00635910f, 0.09246826f, 0.10687256f, 0.08917236f,
      -0.09613037f, -0.13793945f, -0.02572632f, 0.09625244f, 0.24853516f, 0.22143555f, -0.22448730f, 0.24536133f,
      0.01716614f, 0.16650391f, 0.07385254f, -0.06188965f, -0.15576172f, 0.00595093f, 0.08038330f, -0.05541992f,
      0.15136719f, 0.19323730f, -0.20214844f, -0.17150879f, 0.09582520f, -0.08459473f, 0.02511597f, -0.24877930f,
      -0.21618652f, -0.15844727f, -0.19152832f, 0.12213135f, -0.10748291f, 0.20935059f, -0.03182983f, -0.09326172f,
      -0.13879395f, 0.07647705f, 0.21081543f, -0.02172852f, -0.02629089f, -0.01597595f, -0.18310547f, 0.14611816f,
      -0.08795166f, -0.06188965f, -0.10211182f, 0.21362305f, 0.18395996f, -0.14807129f, -0.07745361f, 0.12927246f,
      -0.19848633f, 0.06884766f, 0.18835449f, -0.10620117f, 0.01678467f, -0.15246582f, -0.13500977f, -0.18994141f,
      0.13098145f, 0.24145508f, 0.21276855f, -0.00587845f, 0.08172607f, -0.13610840f, 0.21484375f, -0.08837891f,
      -0.08508301f, -0.00902557f, -0.08868408f, -0.13891602f, 0.15368652f, 0.16210938f, 0.13452148f, -0.17468262f,
      0.18286133f, -0.06088257f, -0.09069824f, 0.19299316f, 0.00621796f, 0.10363770f, -0.08990479f, -0.19482422f,
      0.03909302f, 0.09295654f, -0.15637207f, -0.19079590f, -0.10717773f, -0.07946777f, -0.04238892f, -0.00128174f,
      -0.01683044f, -0.00362206f, -0.18286133f, -0.20349121f, -0.00975800f, 0.22802734f, 0.21630859f, -0.09545898f,
      0.21411133f, -0.03573608f, 0.23864746f, 0.03732300f, 0.20751953f, 0.02011108f, -0.06719971f, -0.17028809f,
      0.01774597f, 0.09790039f, -0.19177246f, 0.22961426f, 0.17089844f, -0.06030273f, -0.15405273f, 0.05746460f,
      0.08483887f, 0.18469238f, -0.04577637f, 0.05151367f, 0.19128418f, -0.15344238f, 0.08569336f, 0.23010254f,
      -0.14770508f, -0.17248535f, -0.00046682f, 0.05953979f, 0.06250000f, 0.07922363f, -0.12243652f, 0.17993164f,
      0.18566895f, 0.12152100f, -0.22973633f, -0.14575195f, 0.04092407f, 0.22290039f, -0.18811035f, -0.20861816f,
      -0.02673340f, -0.16235352f, 0.00653076f, -0.18957520f, -0.15026855f, -0.16662598f, 0.17211914f, -0.22143555f,
      -0.09912109f, 0.12060547f, -0.11578369f, 0.23925781f, -0.01065063f, 0.21777344f, 0.13000488f, 0.10327148f,
      0.10449219f, 0.16516113f, -0.16149902f, 0.07958984f, -0.17382812f, 0.11828613f, -0.04739380f, 0.17041016f,
      -0.13220215f, -0.04071045f, -0.03994751f, 0.23937988f, 0.08105469f, -0.15209961f, -0.20288086f, -0.03347778f,
      -0.07153320f, 0.14538574f, 0.03262329f, -0.07299805f, -0.18029785f, 0.23376465f, -0.20422363f, -0.06701660f,
      0.00343513f, -0.03097534f, 0.03335571f, -0.10906982f, 0.08288574f, -0.02381897f, 0.13708496f, 0.03414917f,
      -0.22900391f, 0.14929199f, 0.11181641f, -0.06481934f, -0.02731323f, 0.02947998f, 0.04864502f, 0.08038330f,
      0.05374146f, 0.17858887f, 0.07659912f, -0.23986816f, -0.22351074f, -0.12060547f, -0.16076660f, 0.16650391f,
      -0.12658691f, 0.09918213f, 0.14819336f, -0.23144531f, 0.09466553f, 0.22399902f, 0.07177734f, 0.09228516f,
      0.03479004f, 0.00276947f, -0.07177734f, 0.09051514f, 0.17309570f, 0.12524414f, -0.18725586f, -0.00495529f,
      0.06176758f, 0.06982422f, -0.03591919f, 0.19091797f, -0.17163086f, -0.10894775f, 0.03900146f, -0.10882568f,
      0.22937012f, -0.20507812f, 0.11370850f, -0.11016846f, -0.21520996f, 0.11730957f, 0.18322754f, 0.23107910f,
      -0.02180481f, 0.19030762f, -0.10058594f, -0.08660889f, 0.24645996f, -0.12237549f, -0.01232147f, -0.24047852f,
      -0.13916016f, -0.06237793f, 0.05062866f, 0.24938965f, 0.22534180f, 0.03454590f, -0.24829102f, -0.00137138f,
      0.11779785f, -0.17712402f, 0.19226074f, 0.24707031f, -0.18212891f, 0.12829590f, -0.13818359f, -0.15026855f,
      0.19274902f, 0.19372559f, 0.11511230f, -0.20373535f, -0.12274170f, -0.05865479f, 0.20910645f, -0.08435059f,
      -0.21911621f, 0.13549805f, -0.17639160f, -0.13281250f, 0.18774414f, -0.11175537f, 0.05133057f, -0.10791016f,
      -0.03219604f, -0.00830841f, -0.04135132f, 0.02661133f, 0.24060059f, -0.23498535f, 0.04092407f, -0.06604004f,
      0.24902344f, 0.16052246f, 0.09350586f, -0.24194336f, 0.17687988f, -0.12634277f, -0.13232422f, -0.00743866f,
      0.06671143f, 0.24060059f, -0.23840332f, -0.07409668f, 0.14343262f, -0.21606445f, 0.22875977f, -0.20446777f,
      -0.09484863f, 0.06762695f, -0.20031738f, -0.05169678f, 0.16931152f, 0.01776123f, 0.03906250f, 0.20581055f,
      -0.11962891f, -0.01544189f, -0.18872070f, -0.01165009f, -0.04556274f, 0.21020508f, 0.13159180f, -0.17297363f,
      -0.11566162f, 0.22009277f, 0.04339600f, 0.07080078f, 0.07958984f, -0.21838379f, 0.01718140f, 0.07873535f,
      0.15515137f, 0.01173401f, 0.09552002f, 0.03533936f, -0.04516602f, 0.05374146f, 0.10864258f, -0.18908691f,
      -0.10174561f, 0.19580078f, -0.01185608f, 0.00821686f, 0.03158569f, 0.21948242f, -0.10321045f, 0.03631592f,
      0.18066406f, -0.08312988f, 0.13293457f, 0.09045410f, -0.04382324f, -0.22412109f, -0.03988647f, 0.19433594f,
      -0.07141113f, 0.15625000f, 0.08032227f, -0.09069824f, 0.01383209f, -0.11352539f, 0.10455322f, -0.07006836f,
      0.12597656f, 0.23400879f, 0.10986328f, -0.14562988f, 0.24243164f, -0.04183960f, -0.21069336f, 0.14062500f,
      -0.18750000f, -0.24658203f, -0.14880371f, 0.22753906f, -0.05587769f, 0.18518066f, 0.10058594f, 0.09075928f,
      -0.17089844f, -0.23022461f, 0.09191895f, -0.22167969f, 0.10119629f, 0.09344482f, -0.24548340f, 0.16906738f,
      -0.05865479f, 0.15393066f, -0.13220215f, 0.19848633f, -0.13232422f, -0.24816895f, -0.15539551f, 0.23889160f,
      -0.18725586f, -0.08575439f, -0.18029785f, 0.04040527f, 0.10455322f, 0.14221191f, 0.01489258f, 0.14916992f,
      0.10552979f, 0.04141235f, 0.05511475f, 0.12988281f, -0.03799438f, 0.03308105f, -0.01216888f, -0.20483398f,
      -0.21643066f, -0.21948242f, -0.17614746f, 0.02754211f, 0.04318237f, -0.18298340f, 0.22619629f, 0.13684082f,
      0.22070312f, 0.14147949f, -0.02151489f, 0.02787781f, 0.16552734f, -0.15356445f, -0.12915039f, -0.24450684f,
      0.12561035f, 0.21069336f, 0.06628418f, 0.13293457f, 0.07897949f, 0.02349854f, 0.11749268f, -0.05865479f,
      0.15393066f, 0.16052246f, -0.09234619f, 0.06994629f, -0.07055664f, -0.16259766f, -0.22338867f, -0.22741699f,
      0.19506836f, -0.05749512f, 0.21154785f, -0.19665527f, 0.19152832f, -0.23779297f, -0.04934692f, 0.13879395f,
      0.11804199f, -0.10546875f, 0.07269287f, 0.19165039f, 0.09234619f, 0.14355469f, 0.00853729f, 0.13378906f,
      0.06787109f, -0.03176880f, -0.09411621f, 0.19543457f, 0.02069092f, -0.06439209f, -0.13037109f, 0.00778580f,
      0.11029053f, 0.17687988f, 0.21069336f, 0.22399902f, -0.15856934f, -0.18371582f, 0.12408447f, 0.14990234f,
      -0.23034668f, 0.11895752f, -0.01561737f, 0.13598633f, -0.07043457f, -0.09393311f, 0.19873047f, 0.04968262f,
      -0.08831787f, 0.02133179f, -0.21801758f, -0.09692383f, -0.12377930f, -0.07916260f, -0.12585449f, -0.14709473f,
      -0.14404297f, -0.04705811f, -0.11993408f, -0.24572754f, 0.15856934f, -0.12634277f, 0.01828003f, 0.21691895f,
      -0.09252930f, 0.04376221f, 0.07391357f, 0.13537598f, 0.21215820f, -0.07031250f, 0.20642090f, -0.02078247f,
      -0.10632324f, 0.22583008f, 0.08245850f, -0.15856934f, -0.01436615f, -0.06854248f, -0.10760498f, 0.11743164f,
      0.00469589f, 0.06945801f, 0.23193359f, -0.05236816f, 0.00703049f, -0.16247559f, 0.17626953f, 0.12353516f,
      -0.08862305f, 0.07128906f, 0.04345703f, -0.03350830f, 0.24365234f, 0.06146240f, -0.00362396f, -0.10803223f,
      -0.14062500f, -0.05798340f, 0.00224876f, 0.14953613f, -0.20837402f, 0.18249512f, 0.23950195f, -0.24804688f,
      -0.03479004f, -0.12915039f, 0.24914551f, -0.17700195f, 0.11389160f, 0.19213867f, -0.02809143f, 0.00966644f,
      0.00834656f, -0.10986328f, 0.20947266f, 0.21582031f, -0.23706055f, -0.09619141f, -0.05532837f, -0.07122803f,
      0.20251465f, 0.02970886f, -0.16455078f, 0.02488708f, 0.08026123f, -0.07238770f, 0.14916992f, -0.05032349f,
      0.06079102f, -0.21643066f, -0.01262665f, 0.02890015f, 0.23791504f, -0.10205078f, 0.15087891f, -0.23022461f,
      -0.21398926f, -0.14855957f, 0.20532227f, 0.03152466f, 0.21582031f, -0.10388184f, 0.21728516f, 0.14257812f,
      0.18444824f, 0.19384766f, 0.00561142f, -0.03024292f, -0.24255371f, 0.19335938f, 0.01824951f, -0.24560547f,
      -0.19104004f, 0.24890137f, -0.07897949f, -0.16394043f, -0.23583984f, 0.10034180f, -0.17211914f, -0.06549072f,
      0.11877441f, 0.14672852f, 0.10705566f, -0.07067871f, -0.09246826f, -0.05908203f, -0.17114258f, -0.11340332f,
      0.07373047f, 0.02824402f, 0.13134766f, 0.13342285f, 0.11706543f, -0.20178223f, -0.14050293f, 0.15832520f,
      0.19128418f, -0.11834717f, -0.10559082f, -0.04580688f, 0.19189453f, 0.10974121f, -0.00661469f, 0.22131348f,
      0.17114258f, -0.18798828f, 0.06372070f, -0.10510254f, 0.17480469f, 0.08953857f, -0.02033997f, -0.15673828f,
      0.01210785f, 0.15429688f, 0.24682617f, 0.06027222f, -0.04794312f, 0.18579102f, 0.11181641f, -0.05419922f,
      0.11230469f, -0.11291504f, 0.20166016f, -0.23229980f, -0.21984863f, 0.20751953f, -0.19738770f, 0.02136230f,
      0.18151855f, 0.10235596f, -0.10797119f, 0.17407227f, 0.21801758f, -0.12213135f, 0.20056152f, 0.17541504f,
      -0.00532913f, 0.21691895f, -0.02801514f, -0.06280518f, 0.17211914f, -0.06323242f, 0.21435547f, -0.24328613f,
      0.06085205f, -0.24487305f, -0.03384399f, -0.14697266f, 0.09222412f, -0.01487732f, -0.21813965f, 0.07672119f};

  std::vector<float> bias_data = {
      -0.16381836f, -0.22399902f, 0.13256836f, -0.24621582f, -0.07043457f, -0.00012034f, -0.04183960f, 0.17565918f,
      0.07720947f, -0.19482422f, -0.00689697f, 0.18908691f, -0.15856934f, -0.15246582f, -0.09948730f, 0.05960083f,
      0.08398438f, 0.04818726f, -0.03555298f, 0.15966797f, -0.09649658f, -0.02667236f, 0.13818359f, 0.06063843f,
      0.21057129f, 0.12042236f, -0.07482910f, -0.17614746f, 0.23242188f, 0.08190918f, -0.16467285f, 0.24218750f,
      0.23278809f, -0.05047607f, 0.07141113f, 0.09313965f, -0.23498535f, 0.22863770f, 0.09918213f, -0.04803467f,
      -0.19775391f, 0.03637695f, 0.15600586f, 0.10614014f, -0.21740723f, -0.08190918f, 0.15600586f, -0.21704102f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {1, 3};

  // std::vector<float> output_data = {
  //     0.48193359f, 0.40234375f, -0.26196289f, 0.33251953f, -0.13049316f, -0.08978271f, -0.59619141f, 0.53125000f,
  //     -0.49365234f, 0.81054688f, 0.72070312f, -0.14941406f, -0.71533203f, 0.25659180f, -0.30517578f, -0.44165039f,

  //     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
  //     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,

  //     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
  //     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,

  //     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
  //     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,

  //     0.30541992f, 0.25610352f, 0.27929688f, 1.02343750f, -0.38159180f, 0.13659668f, 0.17712402f, 0.52099609f,
  //     1.00000000f, -0.59277344f, 1.01464844f, 0.10467529f, -0.51464844f, 0.63378906f, -0.00745773f, -0.50292969f,

  //     0.29077148f, 0.25390625f, 0.31347656f, 0.92041016f, -0.33642578f, 0.20593262f, 0.19995117f, 0.56298828f,
  //     0.84619141f, -0.57080078f, 1.06738281f, 0.05352783f, -0.49584961f, 0.64013672f, -0.06304932f, -0.45312500f,

  //     0.01612854f, 0.28637695f, 0.48901367f, 0.84667969f, -0.46655273f, 0.04162598f, 0.17150879f, 0.58691406f,
  //     0.96484375f, -0.51904297f, 0.96972656f, 0.01826477f, -0.50634766f, 0.71582031f, -0.10363770f, -0.40478516f,

  //     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f,
  //     0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f, 0.00000000f};

  std::vector<float> output_data = {
      0.47241211f, -0.75878906f, -0.33007812f, 0.78369141f, -0.24353027f, 0.87792969f, -0.30566406f, 0.073608398f, 0.31152344f, 0.74902344f, 0.32177734f, 0.48364258f, -0.31738281f, -0.49121094f, 0.67089844f, -0.29785156f,
      0.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f,
      0.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f,
      0.41967773f, -0.10870361f, 0.33251953f, 0.053863525f, -0.26245117f, 0.045013428f, -0.68115234f, -0.14941406f, -0.014549255f, 0.046447754f, -0.17016602f, 0.1854248f, -0.35717773f, 0.3425293f, 0.25830078f, -0.3737793f,
      0.36547852f, -0.11206055f, 0.40380859f, 0.077270508f, -0.17053223f, 0.10760498f, -0.57666016f, -0.095031738f, -0.20397949f, -0.014648438f, -0.13256836f, 0.1328125f, -0.32202148f, 0.15161133f, 0.4074707f, -0.26293945f,
      0.40795898f, -0.043487549f, 0.28076172f, 0.044708252f, -0.23852539f, 0.060943604f, -0.65185547f, -0.17907715f, -0.047363281f, 0.30078125f, -0.1149292f, 0.34863281f, -0.40698242f, 0.25317383f, 0.35400391f, -0.45922852f,
      0.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f};

      // 0.47241211f, -0.75878906f, -0.33007812f, 0.78369141f, -0.24353027f, 0.87792969f, -0.30566406f, 0.073608398f, 0.31152344f, 0.74902344f, 0.32177734f, 0.48364258f, -0.31738281f, -0.49121094f, 0.67089844f, -0.29785156f,
      // 0.47241211f, -0.75878906f, -0.33007812f, 0.78369141f, -0.24353027f, 0.87792969f, -0.30566406f, 0.073608398f, 0.31152344f, 0.74902344f, 0.32177734f, 0.48364258f, -0.31738281f, -0.49121094f, 0.67089844f, -0.29785156f,
      // 0.47241211f, -0.75878906f, -0.33007812f, 0.78369141f, -0.24353027f, 0.87792969f, -0.30566406f, 0.073608398f, 0.31152344f, 0.74902344f, 0.32177734f, 0.48364258f, -0.31738281f, -0.49121094f, 0.67089844f, -0.29785156f,
      // 0.47241211f, -0.75878906f, -0.33007812f, 0.78369141f, -0.24353027f, 0.87792969f, -0.30566406f, 0.073608398f, 0.31152344f, 0.74902344f, 0.32177734f, 0.48364258f, -0.31738281f, -0.49121094f, 0.67089844f, -0.29785156f,

      // 0.58398438f, -0.091064453f, -0.27270508f, 0.51220703f, 0.24987793f, -0.10003662f, 0.2310791f, -0.074951172f, -0.3503418f, -0.0049171448f, -0.11749268f, 0.71923828f, -0.27246094f, 0.11236572f, -0.58447266f, -0.04876709f,
      // 0.3347168f, -0.37988281f, -0.43432617f, 0.47167969f, 0.20532227f, -0.033172607f, 0.37915039f, -0.12780762f, -0.36791992f, -0.01235199f, -0.12445068f, 0.71533203f, -0.27368164f, 0.11578369f, -0.56396484f, -0.053924561f,
      // 0.31567383f, -0.37207031f, -0.42016602f, 0.4777832f, 0.24084473f, -0.027664185f, 0.37280273f, -0.11627197f, -0.49975586f, -0.16137695f, -0.21838379f, 0.72753906f, -0.23413086f, 0.14697266f, -0.40405273f, -0.049560547f,
      // 0.47924805f, -0.24829102f, -0.37280273f, 0.48388672f, 0.18383789f, -0.07244873f, 0.31469727f, -0.11602783f, -0.54541016f, -0.13586426f, -0.21630859f, 0.69824219f, -0.26074219f, 0.15344238f, -0.35449219f, -0.084106445f};

  bool use_float16 = true;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  int input_hidden_size = 0;
  int max_sequence_length = 0;
  bool disable_cpu = true;
  bool disable_cuda = false;
  bool disable_rocm = true;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data,
                   kMaskIndexEnd, input_hidden_size, max_sequence_length, disable_cpu, disable_cuda, disable_rocm);
}

TEST(AttentionTest, AttentionUnidirectional3DMask) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test 3D mask BxSxS*
  std::vector<int32_t> mask_index_data = {
      0, 1,
      0, 1,
      1, 1,
      1, 1};

  std::vector<float> output_data = {
      3.0146f, 0.1142f, 3.9834f, 5.3394f,
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.96967912f, 0.07314367f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = true;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMask3D);
}

TEST(AttentionTest, AttentionUnidirectionalAttentionMask) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test mask start position > 0.
  std::vector<int32_t> mask_index_data = {0, 1, 1, 1};

  std::vector<float> output_data = {
      3.0146f, 0.1142f, 3.9834f, 5.3394f,
      8.69f, -0.13f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f,
      3.96967912f, 0.07314367f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = true;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskRaw);
}

TEST(AttentionTest, AttentionMask1DEndNoWord) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEnd);
}

TEST(AttentionTest, AttentionMask1DNoWord) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0, 2, 2};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

TEST(AttentionTest, AttentionMask2DNoWord) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0, 0, 0};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskRaw);
}

TEST(AttentionTest, AttentionMask3DNoWord) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMask3D);
}

TEST(AttentionTest, AttentionDummyMask2D) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_index_data = {1, 1};

  std::vector<float> output_data = {
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.65f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.65f,
      3.9696791172027588f, 0.073143675923347473f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskDummy);
}

TEST(AttentionTest, Attention4DMask) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test 4D mask Bx1xmax_Sxmax_S
  std::vector<int32_t> mask_index_data = {
      0, 0, 0, 0,
      0, 1, 0, 0,
      0, 1, 1, 0,
      0, 1, 1, 1};

  std::vector<float> output_data = {
      3.97f, 0.073f, 4.25f, 5.65f,
      8.69f, -0.13f, 4.25f, 5.65f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  int input_hidden_size = 0;
  int max_sequence_length = 4;
  bool disable_cpu = true;  // 4D mask not support in CPU kernel
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length,
                   past_data, present_data, kMask4D, input_hidden_size, max_sequence_length,
                   disable_cpu);
}

TEST(AttentionTest, AttentionMaskIndexOutOfRange) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test end_position > sequence length, or start_position < 0
  std::vector<int32_t> mask_index_data = {3, 2, 0, -1};

  std::vector<float> output_data = {
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f,
      3.1495983600616455f, 0.10843668878078461f, 4.25f, 5.6499996185302734f,
      3.9696791172027588f, 0.073143675923347473f, 4.2499995231628418f, 5.6499991416931152f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskIndexEndAndStart);
}

#if !defined(__wasm__)
// TODO: fix in web assembly
TEST(AttentionTest, AttentionPastState_dynamic) {
  // create rand inputs
  RandomValueGenerator random{};

  std::vector<int64_t> input_dims{2, 5, 768};
  std::vector<float> input_data = random.Gaussian<float>(input_dims, 0.0f, 0.3f);

  std::vector<int64_t> weight_dims{768, 2304};
  std::vector<float> weight_data = random.Gaussian<float>(weight_dims, 0.0f, 0.3f);

  std::vector<int64_t> bias_dims{2304};
  std::vector<float> bias_data = random.Gaussian<float>(bias_dims, 0.0f, 0.3f);

  std::vector<int64_t> past_dims{2, 2, 12, 15, 64};
  std::vector<float> past_data = random.Gaussian<float>(past_dims, 0.0f, 0.3f);

  OpTester test("Attention", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("num_heads", 12);
  test.AddAttribute<int64_t>("unidirectional", 1);
  test.AddInput<float>("input", input_dims, input_data);
  test.AddInput<float>("weight", weight_dims, weight_data);
  test.AddInput<float>("bias", bias_dims, bias_data);
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<float>("past", past_dims, past_data);

  test.AddReferenceOutputs("testdata/attention_past_state.onnx", 0.005f);
  test.Run();
}
#endif  //! defined(__wasm__)

TEST(AttentionTest, AttentionPrunedModel) {
  int batch_size = 2;
  int sequence_length = 2;
  // test input_hidden_size > hidden_size
  int input_hidden_size = 6;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f, 0.0f, 1.0f,
      0.8f, -0.5f, 0.0f, 1.f, 2.0f, 3.0f,
      0.8f, -0.5f, 0.0f, 1.f, 4.0f, 5.0f,
      0.5f, 0.2f, 0.3f, -0.6f, 6.0f, 7.0f};

  std::vector<float> weight_data = {
      0.1f,
      -0.2f,
      0.3f,
      1.0f,
      1.1f,
      0.3f,
      0.5f,
      0.2f,
      0.3f,
      -0.6f,
      1.5f,
      2.0f,
      0.5f,
      0.1f,
      0.4f,
      1.6f,
      1.0f,
      2.0f,
      0.4f,
      0.8f,
      0.9f,
      0.1f,
      -1.3f,
      0.7f,
      0.3f,
      0.2f,
      4.0f,
      2.2f,
      1.6f,
      1.1f,
      0.7f,
      0.2f,
      0.4f,
      1.0f,
      1.2f,
      0.5f,
      0.2f,
      0.1f,
      0.4f,
      1.6f,
      2.4f,
      3.3f,
      2.1f,
      4.2f,
      8.4f,
      0.0f,
      2.1f,
      3.2f,
      0.1f,
      0.2f,
      0.3f,
      0.4f,
      0.5f,
      0.6f,
      0.7f,
      0.8f,
      0.9f,
      1.0f,
      1.1f,
      1.2f,
      1.2f,
      1.1f,
      1.0f,
      0.9f,
      0.8f,
      0.7f,
      0.6f,
      0.5f,
      0.4f,
      0.3f,
      0.2f,
      0.1f,
  };

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  std::vector<int32_t> mask_data = {1, 1, 1, 1};

  std::vector<float> output_data = {
      11.689527f, 2.769937f, 7.05f, 8.350000f,
      11.690000f, 2.770000f, 7.05f, 8.350000f,
      14.276558f, 5.374159f, 9.650001f, 10.95f,
      14.289073f, 5.370287f, 9.650001f, 10.95f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  RunAttentionTest(input_data, weight_data, bias_data, mask_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length, past_data, present_data, kMaskRaw, input_hidden_size);
}

static void RunModelWithRandomInput(
    int batch_size,
    int sequence_length,
    std::vector<int64_t>& mask_index_dims,
    std::vector<int32_t>& mask_index_data,
    std::string& onnx_model,
    bool is_float16) {
  RandomValueGenerator random{234};

  constexpr int hidden_size = 768;
  constexpr int num_heads = 12;

  std::vector<int64_t> batch_input_dims{1, sequence_length, hidden_size};
  std::vector<float> batch_input_data = random.Uniform<float>(batch_input_dims, -1.0f, 1.0f);

  std::vector<int64_t> input_dims{batch_size, sequence_length, hidden_size};
  std::vector<float> input_data;
  for (int i = 0; i < batch_size; i++) {
    input_data.insert(input_data.end(), batch_input_data.begin(), batch_input_data.end());
  }

  std::vector<int64_t> weight_dims{hidden_size, 3 * hidden_size};
  std::vector<float> weight_data = random.Uniform<float>(weight_dims, -1.0f, 1.0f);

  std::vector<int64_t> bias_dims{3 * hidden_size};
  std::vector<float> bias_data = random.Uniform<float>(bias_dims, -1.0f, 1.0f);

  float gpu_threshold = is_float16 ? static_cast<float>(sequence_length) / 32.0f : 0.005f;
  constexpr float cpu_threshold = 0.002f;
  bool enable_cuda = HasCudaEnvironment(is_float16 ? 530 : 0);
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get());
  bool enable_cpu = (nullptr != DefaultCpuExecutionProvider().get() && !is_float16);
  if (enable_cuda || enable_rocm) {
    OpTester test("Attention", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("num_heads", num_heads);
    if (is_float16) {
      test.AddInput<MLFloat16>("input", input_dims, ToFloat16(input_data));
      test.AddInput<MLFloat16>("weight", weight_dims, ToFloat16(weight_data));
      test.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias_data));
    } else {
      test.AddInput<float>("input", input_dims, input_data);
      test.AddInput<float>("weight", weight_dims, weight_data);
      test.AddInput<float>("bias", bias_dims, bias_data);
    }
    test.AddInput<int>("mask_index", mask_index_dims, mask_index_data);
    test.AddReferenceOutputs(onnx_model, gpu_threshold);
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    if (enable_cuda) {
      execution_providers.push_back(DefaultCudaExecutionProvider());
    } else {
      execution_providers.push_back(DefaultRocmExecutionProvider());
    }
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }

  if (enable_cpu) {
    OpTester test("Attention", 1, onnxruntime::kMSDomain);
    test.AddAttribute<int64_t>("num_heads", num_heads);
    test.AddInput<float>("input", input_dims, input_data);
    test.AddInput<float>("weight", weight_dims, weight_data);
    test.AddInput<float>("bias", bias_dims, bias_data);
    test.AddInput<int>("mask_index", mask_index_dims, mask_index_data);
    test.AddReferenceOutputs(onnx_model, cpu_threshold);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(AttentionTest, Attention_Mask2D_Fp32_B2_S32) {
  constexpr int batch_size = 2;
  constexpr int sequence_length = 32;

  std::vector<int64_t> mask_index_dims{batch_size, sequence_length};
  std::vector<int32_t> mask_index_data;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < sequence_length; j++) {
      mask_index_data.push_back((i == 0 || j < sequence_length / 2) ? 1 : 0);
    }
  }

  std::string onnx_model = "testdata/attention_mask2d_fp32.onnx";
  RunModelWithRandomInput(
      batch_size,
      sequence_length,
      mask_index_dims,
      mask_index_data,
      onnx_model,
      false);
}

TEST(AttentionTest, Attention_Mask1D_Fp32_B2_S64) {
  constexpr int batch_size = 2;
  constexpr int sequence_length = 64;

  std::vector<int64_t> mask_index_dims{batch_size};
  std::vector<int32_t> mask_index_data;
  for (int i = 0; i < batch_size; i++) {
    mask_index_data.push_back(i == 0 ? sequence_length : (sequence_length / 2));
  }

  std::string onnx_model = "testdata/attention_mask1d_fp32.onnx";
  RunModelWithRandomInput(
      batch_size,
      sequence_length,
      mask_index_dims,
      mask_index_data,
      onnx_model,
      false);
}

TEST(AttentionTest, Attention_Mask1D_Fp16_B2_FusedNoPadding) {
  constexpr int batch_size = 2;

  // Sequence lengths used in TRT fused attention fp16 v2 kernels.
  std::vector<int> sequence_lengths{64, 128, 192, 256, 384, 512};

  for (const auto& sequence_length : sequence_lengths) {
    std::vector<int64_t> mask_index_dims{batch_size};
    std::vector<int32_t> mask_index_data;
    for (int i = 0; i < batch_size; i++) {
      mask_index_data.push_back(sequence_length);
    }

    std::string onnx_model = "testdata/attention_mask1d_fp16.onnx";

    RunModelWithRandomInput(
        batch_size,
        sequence_length,
        mask_index_dims,
        mask_index_data,
        onnx_model,
        true);
  }
}

TEST(AttentionTest, AttentionBatch1_No_Weights) {
  int batch_size = 1;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;
  int kv_sequence_length = 3;
  int v_hidden_size = 2;

  // query: (batch_size, sequence_length, hidden_size)
  std::vector<float> input_data = {
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {};

  // (hidden_size + hidden_size + v_hidden_size)
  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f};

  std::vector<int32_t> mask_index_data = {2L};

  // (batch_size, kv_sequence_length, hidden_size)
  std::vector<float> key_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};

  // (batch_size, kv_sequence_length, v_hidden_size)
  std::vector<float> value_data = {0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f};

  // (batch_size, sequence_length, v_hidden_size)
  std::vector<float> output_data = {0.99434918f, 0.0f, 0.9887343f, 0.74572039f};

  bool use_float16 = false;
  bool is_unidirectional = false;
  bool use_past_state = false;
  int past_sequence_length = 0;
  const std::vector<float>* past_data = nullptr;
  const std::vector<float>* present_data = nullptr;
  MaskIndexType mask_index_type = kMaskIndexEnd;
  int input_hidden_size = 0;
  int max_sequence_length = 0;
  constexpr bool disable_cpu = true;  // not supported in cpu right now.
  constexpr bool disable_cuda = false;
  constexpr bool disable_rocm = true;  // not supported in rocm right now.
  const std::vector<int32_t> qkv_sizes = {hidden_size, hidden_size, v_hidden_size};
  const std::vector<float>& extra_add_data = {};

  RunAttentionTest(input_data, weight_data, bias_data, mask_index_data, output_data,
                   batch_size, sequence_length, hidden_size, number_of_heads,
                   use_float16, is_unidirectional, use_past_state, past_sequence_length,
                   past_data, present_data, mask_index_type, input_hidden_size, max_sequence_length,
                   disable_cpu, disable_cuda, disable_rocm, qkv_sizes, extra_add_data,
                   kv_sequence_length, &key_data, &value_data);
}

#ifndef ENABLE_TRAINING  // Prepacking is enabled only on non-training builds
TEST(AttentionTest, SharedPrepackedWeights) {
  int batch_size = 2;
  int sequence_length = 2;
  int hidden_size = 4;
  int number_of_heads = 2;

  std::vector<float> input_data = {
      0.5f, 0.2f, 0.3f, -0.6f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.8f, -0.5f, 0.0f, 1.f,
      0.5f, 0.2f, 0.3f, -0.6f};

  std::vector<float> weight_data = {
      0.1f, -0.2f, 0.3f, 1.0f, 1.1f, 0.3f, 0.5f, 0.2f, 0.3f, -0.6f, 1.5f, 2.0f,
      0.5f, 0.1f, 0.4f, 1.6f, 1.0f, 2.0f, 0.4f, 0.8f, 0.9f, 0.1f, -1.3f, 0.7f,
      0.3f, 0.2f, 4.0f, 2.2f, 1.6f, 1.1f, 0.7f, 0.2f, 0.4f, 1.0f, 1.2f, 0.5f,
      0.2f, 0.1f, 0.4f, 1.6f, 2.4f, 3.3f, 2.1f, 4.2f, 8.4f, 0.0f, 2.1f, 3.2f};

  std::vector<float> bias_data = {
      -0.5f, 0.6f, 1.2f, 2.1f, 0.5f, 0.7f, 0.2f, 1.2f, 0.5f, 0.4f, 0.3f, 1.2f};

  // Test that all attention masks are zero.
  std::vector<int32_t> mask_index_data = {0, 0, 2, 2};

  std::vector<float> output_data = {
      3.96724534f, 0.07324841f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.14984703f, 0.10842596f, 4.25f, 5.65f,
      3.96724534f, 0.07324841f, 4.25f, 5.65f};

  OpTester tester("Attention", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
  tester.AddAttribute<int64_t>("unidirectional", static_cast<int64_t>(0));

  std::vector<int64_t> input_dims = {batch_size, sequence_length, hidden_size};
  std::vector<int64_t> weights_dims = {hidden_size, 3 * hidden_size};
  std::vector<int64_t> bias_dims = {3 * hidden_size};

  std::vector<int64_t> mask_index_dims = {2 * batch_size};
  std::vector<int64_t> output_dims = {batch_size, sequence_length, hidden_size};

  tester.AddInput<float>("input", input_dims, input_data);
  tester.AddInput<float>("weight", weights_dims, weight_data, true);  // Trigger pre-packing
  tester.AddInput<float>("bias", bias_dims, bias_data);
  tester.AddOutput<float>("output", output_dims, output_data);
  tester.AddInput<int32_t>("mask_index", mask_index_dims, mask_index_data);

  OrtValue weight;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape(weights_dims),
                       weight_data.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), weight);

  SessionOptions so;

  // Set up weight as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("weight", &weight), Status::OK());

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  tester.EnableSharingOfPrePackedWeightsAcrossSessions();

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  size_t number_of_pre_packed_weights_counter_session_1 = 0;
  size_t number_of_shared_pre_packed_weights_counter = 0;

  // Session 1
  {
    auto ep_vec = cpu_ep();
    tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
               &ep_vec, {}, &number_of_pre_packed_weights_counter_session_1, &number_of_shared_pre_packed_weights_counter);
    // Assert that no pre-packed weights have been shared thus far
    ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
  }

  auto number_of_elements_in_shared_prepacked_buffers_container =
      tester.GetNumPrePackedWeightsShared();
  // Assert that the number of elements in the shared container
  // is the same as the number of weights that have been pre-packed
  ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_elements_in_shared_prepacked_buffers_container);

  // On some platforms/architectures MLAS may choose to not do any pre-packing and the number of elements
  // that have been pre-packed will be zero in which case we do not continue with the testing
  // of "sharing" of pre-packed weights as there are no pre-packed weights to be shared at all.
  if (number_of_pre_packed_weights_counter_session_1 == 0)
    return;

  // Session 2
  {
    size_t number_of_pre_packed_weights_counter_session_2 = 0;
    auto ep_vec = cpu_ep();
    tester.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
               &ep_vec, {}, &number_of_pre_packed_weights_counter_session_2, &number_of_shared_pre_packed_weights_counter);

    // Assert that the same number of weights were pre-packed in both sessions
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_pre_packed_weights_counter_session_2);

    // Assert that the number of pre-packed weights that were shared equals
    // the number of pre-packed weights in the second session
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_2,
              static_cast<size_t>(number_of_shared_pre_packed_weights_counter));
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime
