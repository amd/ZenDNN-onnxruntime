/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*******************************************************************************
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
* LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*******************************************************************************/

#include "gtest/gtest.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/common/zendnn_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// Disable TensorRT on the tests because of SegFault errors in the parser

TEST(TensorOpTest, Unsqueeze_1) {
  OpTester test("Unsqueeze");

  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddOutput<float>("output", {2, 1, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Unsqueeze_1_int32) {
  OpTester test("Unsqueeze");

  test.AddAttribute("axes", std::vector<int64_t>{1});
  test.AddInput<int32_t>("input", {2, 3, 4}, std::vector<int32_t>(2 * 3 * 4, 1));
  test.AddOutput<int32_t>("output", {2, 1, 3, 4}, std::vector<int32_t>(2 * 3 * 4, 1));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(TensorOpTest, Unsqueeze_2) {
  OpTester test("Unsqueeze");

  test.AddAttribute("axes", std::vector<int64_t>{0, 4});
  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddOutput<float>("output", {1, 2, 3, 4, 1}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.Run();
}

TEST(TensorOpTest, Unsqueeze_3) {
  OpTester test("Unsqueeze");

  test.AddAttribute("axes", std::vector<int64_t>{2, 1, 0});
  test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
  test.Run();
}

TEST(TensorOpTest, Unsqueeze_scalar) {
  {
    OpTester test("Unsqueeze");

    test.AddAttribute("axes", std::vector<int64_t>{0});
    test.AddInput<float>("input", {}, std::vector<float>{1.0f});
    test.AddOutput<float>("output", {1}, std::vector<float>{1.0f});
    test.Run();
  }
  {
    OpTester test("Unsqueeze", 11);  // Negative axes added in version 11

    test.AddAttribute("axes", std::vector<int64_t>{-1});
    test.AddInput<float>("input", {}, std::vector<float>{1.0f});
    test.AddOutput<float>("output", {1}, std::vector<float>{1.0f});
    test.Run();
  }

  auto run_test = [](bool axes_is_initializer) {
    {
      OpTester test("Unsqueeze", 13);
      test.AddInput<float>("input", {}, std::vector<float>{1.0f});
      test.AddInput<int64_t>("axes", {1}, std::vector<int64_t>{0}, axes_is_initializer);
      test.AddOutput<float>("output", {1}, std::vector<float>{1.0f});
      test.Run();
    }
    {
      OpTester test("Unsqueeze", 13);
      test.AddInput<float>("input", {}, std::vector<float>{1.0f});
      test.AddInput<int64_t>("axes", {1}, std::vector<int64_t>{-1}, axes_is_initializer);
      test.AddOutput<float>("output", {1}, std::vector<float>{1.0f});
      test.Run();
    }
  };
  run_test(false);
  run_test(true);
}

TEST(TensorOpTest, Unsqueeze_scalar_2) {
  {
    OpTester test("Unsqueeze");

    test.AddAttribute("axes", std::vector<int64_t>{0, 1});
    test.AddInput<float>("input", {}, std::vector<float>{1.0f});
    test.AddOutput<float>("output", {1, 1}, std::vector<float>{1.0f});
    test.Run();
  }
  auto run_test = [](bool axes_is_initializer) {
    OpTester test("Unsqueeze", 13);
    test.AddInput<float>("input", {}, std::vector<float>{1.0f});
    test.AddInput<int64_t>("axes", {2}, std::vector<int64_t>{0, -1}, axes_is_initializer);
    test.AddOutput<float>("output", {1, 1}, std::vector<float>{1.0f});
    test.Run();
  };
  run_test(false);
  run_test(true);
}

TEST(TensorOpTest, Unsqueeze_Duplicate) {
  {
    OpTester test("Unsqueeze", 12);  // opset 1-12 has axes attribute

    test.AddAttribute("axes", std::vector<int64_t>{2, 1, 0, 2});
    test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "[ShapeInferenceError] 'axes' attribute must not contain any duplicates",
             {kTensorrtExecutionProvider});  // TensorRT failed
  }
  {
    OpTester test("Unsqueeze", -1);  // use latest opset with axis input

    test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.AddInput<int64_t>("axes", {4}, std::vector<int64_t>{2, 1, 0, 2}, true);  // set as initializer to enable shape inference
    test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "[ShapeInferenceError] Axis 2 is referred to more than once",
             {kTensorrtExecutionProvider});  // TensorRT failed
  }
}

TEST(TensorOpTest, Unsqueeze_OutOfRange) {
  {
    OpTester test("Unsqueeze", 12);  // opset 1-12 has axes attribute
    test.AddAttribute("axes", std::vector<int64_t>{4});
    test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.AddOutput<float>("output", {2, 1, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "[ShapeInferenceError] values in 'axes' are beyond the bounds of the computed output shape");
  }
  {
    OpTester test("Unsqueeze", -1);  // use latest opset with axis input
    test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.AddInput<int64_t>("axes", {1}, std::vector<int64_t>{4}, true);  // set as initializer to enable shape inference
    test.AddOutput<float>("output", {2, 1, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    // TensorRT does not support negative axis.
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "[ShapeInferenceError] Unexpected axis value",
             {kTensorrtExecutionProvider});  // TensorRT expects 'axes' attribute
  }
}

TEST(TensorOpTest, UnsqueezeNegAxis_3) {
  {
    OpTester test("Unsqueeze", 12);  // opset 1-12 has axes attribute
    test.AddAttribute("axes", std::vector<int64_t>{-4, 1, -6});
    test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    // TensorRT does not support negative axis.
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  }
  auto run_test = [](bool axes_is_initializer) {
    OpTester test("Unsqueeze", 13);  // use latest opset with axis input
    test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.AddInput<int64_t>("axes", {3}, std::vector<int64_t>{-4, 1, -6}, axes_is_initializer);
    test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    // TensorRT does not support negative axis.
    // TODO: TensorRT, OpenVINO dont support "axes" input in opset 13, re-enable after
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
  };
  run_test(false);
  run_test(true);
}

TEST(TensorOpTest, Unsqueeze_1_int32_axes_input) {
  auto run_test = [](bool axes_is_initializer) {
    OpTester test("Unsqueeze", 13);

    test.AddInput<int32_t>("input", {2, 3, 4}, std::vector<int32_t>(2 * 3 * 4, 1));
    test.AddInput<int64_t>("axes", {1}, std::vector<int64_t>{1}, axes_is_initializer);
    test.AddOutput<int32_t>("output", {2, 1, 3, 4}, std::vector<int32_t>(2 * 3 * 4, 1));
    // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
  };
  run_test(false);
  run_test(true);
}

TEST(TensorOpTest, Unsqueeze_3_axes_input) {
  auto run_test = [](bool axes_is_initializer) {
    OpTester test("Unsqueeze", 13);

    test.AddInput<float>("input", {2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    test.AddInput<int64_t>("axes", {3}, std::vector<int64_t>{2, 1, 0}, axes_is_initializer);
    test.AddOutput<float>("output", {1, 1, 1, 2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0f));
    // TODO: TensorRT and OpenVINO dont support "axes" input in opset 13, re-enable after
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
  };
  run_test(false);
  run_test(true);
}

#if defined(USE_DNNL) || defined(USE_ZENDNN)
TEST(TensorOpTest, Unsqueeze_3_axes_input_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#ifdef USE_ZENDNN
  if (!ZendnnHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Unsqueeze", 13);
  test.AddInput<BFloat16>("input", {2, 3, 4}, FloatsToBFloat16s(std::vector<float>(2 * 3 * 4, 1.0f)));
  test.AddInput<int64_t>("axes", {3}, std::vector<int64_t>{2, 1, 0}, true);
  test.AddOutput<BFloat16>("output", {1, 1, 1, 2, 3, 4}, FloatsToBFloat16s(std::vector<float>(2 * 3 * 4, 1.0f)));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
#if defined(USE_ZENDNN)
  execution_providers.push_back(DefaultZendnnExecutionProvider());
#endif  //  USE_ZENDNN
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(TensorOpTest, UnsqueezeNegAxis_3_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
#ifdef USE_ZENDNN
  if (!ZendnnHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("Unsqueeze", 13);
  test.AddInput<BFloat16>("input", {2, 3, 4}, FloatsToBFloat16s(std::vector<float>(2 * 3 * 4, 1.0f)));
  test.AddInput<int64_t>("axes", {3}, std::vector<int64_t>{-4, 1, -6}, true);
  test.AddOutput<BFloat16>("output", {1, 1, 1, 2, 3, 4}, FloatsToBFloat16s(std::vector<float>(2 * 3 * 4, 1.0f)));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
#if defined(USE_ZENDNN)
  execution_providers.push_back(DefaultZendnnExecutionProvider());
#endif  //  USE_ZENDNN
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  //  USE_DNNL USE_ZENDNN

}  // namespace test
}  // namespace onnxruntime
