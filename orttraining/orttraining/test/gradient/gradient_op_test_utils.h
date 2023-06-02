// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <initializer_list>
#include <functional>

#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

struct TensorInfo {
  TensorInfo(std::initializer_list<int64_t> shape_init,
             bool has_gradient = true,
             std::function<float(float)>* transformer = nullptr,
             MLDataType data_type = DataTypeImpl::GetTensorType<float>(),
             const std::vector<std::string>& dim_params = std::vector<std::string>{})
      : shape(gsl::make_span(shape_init.begin(), shape_init.end())),
        has_gradient(has_gradient),
        transformer(transformer),
        data_type(data_type),
        dim_params(dim_params) {}

  TensorInfo(const TensorShape& shape, bool has_gradient = true, std::function<float(float)>* transformer = nullptr,
             MLDataType data_type = DataTypeImpl::GetTensorType<float>())
      : shape(shape), has_gradient(has_gradient), transformer(transformer), data_type(data_type) {}

  TensorShape shape;
  bool has_gradient;
  std::function<float(float)>* transformer;
  MLDataType data_type;
  std::vector<std::string> dim_params;
};

using TestDataVector = std::tuple<std::vector<std::vector<TensorInfo>>,             // Input data
                                  std::vector<std::vector<TensorInfo>>,             // output data
                                  std::vector<std::vector<onnx::AttributeProto>>>;  // attribute

class GradientOpTester : public OpTester {
 public:
  // ctor for testing where gradIents are set
  explicit GradientOpTester(const char* op,
                            int opset_version = 9,
                            const char* domain = onnxruntime::kOnnxDomain,
                            bool verify_output = true,
                            const std::vector<TensorInfo>* input_infos = nullptr,
                            const std::vector<TensorInfo>* output_infos = nullptr)
      : OpTester(op, opset_version, domain, verify_output),
        input_infos_{input_infos},
        output_infos_{output_infos} {
  }

  // Initial Run
  void Run(ExpectResult expect_result = ExpectResult::kExpectSuccess,
           const std::string& expected_failure_string = "",
           const std::unordered_set<std::string>& excluded_provider_types = {},
           const RunOptions* run_options = nullptr,
           std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr) {
    BaseTester::Run(expect_result, expected_failure_string, excluded_provider_types, run_options, execution_providers);
  }

  // Run when input_infos_ and output_infos_ are provided
  void Run(int output_index_to_use_as_loss,
           int data_index_of_output,
           ExpectResult expect_result = ExpectResult::kExpectSuccess,
           const std::string& expected_failure_string = "",
           const std::unordered_set<std::string>& excluded_provider_types = {},
           const RunOptions* run_options = nullptr,
           std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr);

  const std::vector<Data>& GetInputData() {
    return BaseTester::GetInputData();
  }

  const std::vector<Data>& GetOutputData() {
    return BaseTester::GetOutputData();
  }

  void ClearData() {
    BaseTester::GetInputData().clear();
    BaseTester::GetOutputData().clear();
    BaseTester::GetInitializerIndexes().clear();
  }

 private:
  void FillFeedsAndOutputNames(std::unordered_map<std::string, OrtValue>& feeds,
                               std::vector<std::string>& output_names,
                               int output_index_to_use_as_loss,
                               int data_index_of_output);

  const std::vector<TensorInfo>* input_infos_;
  const std::vector<TensorInfo>* output_infos_;
};
}  // namespace test
}  // namespace onnxruntime
