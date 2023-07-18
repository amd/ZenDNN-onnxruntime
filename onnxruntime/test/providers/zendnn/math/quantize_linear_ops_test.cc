/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#ifdef USE_ZENDNN
// The same as the default provider, but in this case with constant initializers to test optimization

TEST(DequantizeLinearOpTest, ZENDNN_Uint8_ConstantInitializer) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{4};
  test.AddInput<uint8_t>("x", dims, {0, 3, 128, 255});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<uint8_t>("x_zero_point", {}, {128}, true);
  test.AddOutput<float>("y", dims, {-256.0f, -250.0f, 0.0f, 254.0f});
  test.Run();
}

// scalar zero & scale with int8
TEST(DequantizeLinearOpTest, ZENDNN_Int8_ConstantInitializer) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{4};
  test.AddInput<int8_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {}, {2.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {-10}, true);
  test.AddOutput<float>("y", dims, {-40.0f, 14.0f, 220.0f, 274.0f});
  // Disable Tensorrt EP due to error:node1_quantize_scale_node: out of bounds channel axis 1. Number of input dimensions is 1.
  test.Run();
}

// scalar zero & scale with int8
TEST(DequantizeLinearOpTest, ZENDNN_Int32_ConstantInitializer) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{4};
  test.AddInput<int32_t>("x", dims, {-30, -3, 100, 127});
  test.AddInput<float>("x_scale", {}, {2.0f}, true);
  test.AddOutput<float>("y", dims, {-60.f, -6.f, 200.f, 254.f});
  test.Run();
}

// 2d inputs
TEST(DequantizeLinearOpTest, ZENDNN_2D_ConstantInitializer) {
  OpTester test("DequantizeLinear", 10);
  std::vector<int64_t> dims{3, 4};
  test.AddInput<uint8_t>("X", dims,
                         {0, 1, 2, 3,
                          0, 1, 2, 3,
                          0, 10, 20, 30});
  test.AddInput<float>("scale", {}, {1.0f});
  test.AddInput<uint8_t>("zero_point", {}, {0}, true);
  test.AddOutput<float>("Y", dims,
                        {0, 1, 2, 3,
                         0, 1, 2, 3,
                         0, 10, 20, 30});
  test.Run();
}

#endif  // USE_ZENDNN

}  // namespace test
}  // namespace onnxruntime