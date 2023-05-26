// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/path_string.h"
#include "core/graph/model.h"
#include "core/session/environment.h"

#include "test/util/include/asserts.h"
#include "test/util/include/base_tester.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {
// To use ModelTester:
//  1. Create one with the model path
//  2. Call AddInput for all the inputs
//  3. Call AddOutput with all expected outputs
//  4. Call Run
//
class ModelTester : public BaseTester {
 public:
  explicit ModelTester(const PathString& model_uri)
      : BaseTester() {
    ASSERT_STATUS_OK(Model::Load(model_uri, model_, nullptr, GetEnvironment().GetLoggingManager()->DefaultLogger()));
  }

  using ExpectResult = BaseTester::ExpectResult;

  const Model& GetModel() const { return *model_; }

 private:
  std::shared_ptr<Model> model_;
};
}  // namespace test
}  // namespace onnxruntime
