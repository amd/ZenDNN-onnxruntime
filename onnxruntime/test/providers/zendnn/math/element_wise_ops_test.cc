/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

// Copyright (c)Intel. All rights reserved.
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
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/util/math.h"
#include <algorithm>
#include <math.h>

namespace onnxruntime {
namespace test {

#ifdef USE_ZENDNN
// Many of the "Pow" tests are identical to the CPU element wise ops tests with the
// exception the exponent for the "Pow" operator is setup as an initilizer value. Since
// the ZENDNN execution provider will only accept exponents that are initializers. This matches
// what is seen in many models that use "Pow"
TEST(MathOpTest, ZENDNN_Pow_Broadcast_Scalar1) {
  OpTester test("Pow");

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("Y", {}, {2.0f}, true);
  test.AddOutput<float>("Z", dims, {1.0f, 4.0f, 9.0f});
  test.Run();
}

TEST(MathOpTest, ZENDNN_Pow_Broadcast_Scalar1_12) {
  OpTester test("Pow", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<float>("Y", {1}, {2.0f}, true);
  test.AddOutput<float>("Z", dims, {1.0f, 4.0f, 9.0f});
  test.Run();
}

TEST(MathOpTest, ZENDNN_Pow_Broadcast_Scalar1_float_int32_12) {
  OpTester test("Pow", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<int32_t>("Y", {}, {3}, true);
  test.AddOutput<float>("Z", dims, {1.0f, 8.0f, 27.0f});
  test.Run();
}

TEST(MathOpTest, ZENDNN_Pow_Broadcast_Scalar1_float_int8_12) {
  OpTester test("Pow", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<int8_t>("Y", {}, {3}, true);
  test.AddOutput<float>("Z", dims, {1.0f, 8.0f, 27.0f});
  test.Run();
}

TEST(MathOpTest, ZENDNN_Pow_Broadcast_Scalar1_float_uint8_12) {
  OpTester test("Pow", 12);

  std::vector<int64_t> dims{3};
  test.AddInput<float>("X", dims, {1.0f, 2.0f, 3.0f});
  test.AddInput<uint8_t>("Y", {}, {3}, true);
  test.AddOutput<float>("Z", dims, {1.0f, 8.0f, 27.0f});
  test.Run();
}
#endif  // USE_ZENDNN

}  // namespace test
}  // namespace onnxruntime