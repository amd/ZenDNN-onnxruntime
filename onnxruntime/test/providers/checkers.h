// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <string_view>

#include "gtest/gtest.h"

#include "core/framework/ort_value.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace test {

struct ValidateOutputParams {
  std::optional<float> absolute_error;
  std::optional<float> relative_error;
  bool sort_output = false;
};

//// Check for Tensor
//void Check(std::string_view name, const ValidateOutputParams& params,
//           const OrtValue& expected, const Tensor& output_tensor, const std::string& provider_type);

// General purpose Check
void Check(std::string_view name, const ValidateOutputParams& params,
           const OrtValue& expected, OrtValue& actual, const std::string& provider_type);

//// Check for non tensor types
//template <typename T>
//void Check(std::string_view name, const ValidateOutputParams& /*params*/,
//           const OrtValue& expected, const T& actual, const std::string& provider_type) {
//  EXPECT_EQ(expected.Get<T>(), actual) << "name: " << name << " provider_type : " << provider_type;
//}
//
//// Check for sequence of tensors
//template <>
//void Check<TensorSeq>(std::string_view name, const ValidateOutputParams& params,
//                      const OrtValue& expected, const TensorSeq& actual, const std::string& provider_type);

inline void ConvertFloatToMLFloat16(const float* f_datat, MLFloat16* h_data, int input_size) {
  auto in_vector = ConstEigenVectorMap<float>(f_datat, input_size);
  auto output_vector = EigenVectorMap<Eigen::half>(static_cast<Eigen::half*>(static_cast<void*>(h_data)), input_size);
  output_vector = in_vector.template cast<Eigen::half>();
}

inline void ConvertMLFloat16ToFloat(const MLFloat16* h_data, float* f_data, int input_size) {
  auto in_vector =
      ConstEigenVectorMap<Eigen::half>(static_cast<const Eigen::half*>(static_cast<const void*>(h_data)), input_size);
  auto output_vector = EigenVectorMap<float>(f_data, input_size);
  output_vector = in_vector.template cast<float>();
}

inline std::vector<MLFloat16> FloatsToMLFloat16s(const std::vector<float>& f) {
  std::vector<MLFloat16> m(f.size());
  ConvertFloatToMLFloat16(f.data(), m.data(), static_cast<int>(f.size()));
  return m;
}

inline std::vector<BFloat16> MakeBFloat16(const std::initializer_list<float>& input) {
  std::vector<BFloat16> output;
  std::transform(input.begin(), input.end(), std::back_inserter(output), [](float f) { return BFloat16(f); });
  return output;
}

inline std::vector<BFloat16> FloatsToBFloat16s(const std::vector<float>& input) {
  std::vector<BFloat16> output;
  std::transform(input.begin(), input.end(), std::back_inserter(output), [](float f) { return BFloat16(f); });
  return output;
}

}  // namespace test
}  // namespace onnxruntime
