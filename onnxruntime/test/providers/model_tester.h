// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/path_string.h"
#include "core/graph/model.h"
#include "core/session/environment.h"

#include "test/providers/base_tester.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {
// To use ModelTester:
//  1. Create one with the model path. Specify
//  2. Call AddInput for all the inputs
//  3. Call AddOutput with all expected outputs
//  4. Call Run
//
class ModelTester : public BaseTester {
 public:
  /// <summary>
  /// Create a model tester. Intended usage is a simple model that is primarily testing a specific operator but may
  /// require additional nodes to exercise the intended code path.
  /// </summary>
  /// <param name="test_name">Name of test to use in logs and error messages.</param>
  /// <param name="model_uri">Model to load</param>
  /// <param name="opset_version">Opset version for the model.</param>
  /// <param name="cache">
  /// Can the model be cached and re-used for each EP?
  /// If optimizations may change the model on a per-EP basis consider setting this to false, or limiting the
  /// optimization level when calling BaseTester::Run via SessionOptions.
  /// </param>
  explicit ModelTester(std::string_view test_name, const PathString& model_uri, int opset_version = -1,
                       bool cache = true)
      : BaseTester{test_name, opset_version, onnxruntime::kOnnxDomain},
        model_uri_{model_uri},
        cache_{cache} {
  }

  using ExpectResult = BaseTester::ExpectResult;

 private:
  Model* CreateModelToTest(const ModelOptions& model_options) override {
    if (!model_ || !cache_) {
      auto status = Model::Load(model_uri_, model_, nullptr, DefaultLoggingManager().DefaultLogger(), model_options);
      ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    } else {
      // previous run would have assigned nodes so we need to clear that out to allowing test a different EP
      ClearEpsForAllNodes(model_->MainGraph());
    }

    return model_.get();
  }

  const PathString& model_uri_;
  const bool cache_;
  std::shared_ptr<Model> model_;
};
}  // namespace test
}  // namespace onnxruntime
