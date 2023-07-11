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

#include <algorithm>

#include "core/common/denormal.h"

#include "gtest/gtest.h"

#include "test/util/include/asserts.h"

#include <array>

namespace onnxruntime {
namespace test {

TEST(DenormalTest, DenormalAsZeroTest) {
  auto test_denormal = [&](bool set_denormal_as_zero) {
    constexpr float denormal_float = 1e-38f;
    constexpr double denormal_double = 1e-308;

    volatile float input_float = denormal_float;
    volatile double input_double = denormal_double;

    // When it returns false, disabling denormal as zero isn't supported,
    // so validation will be skipped
    bool set = SetDenormalAsZero(set_denormal_as_zero);
    if (set || !set_denormal_as_zero) {
      EXPECT_EQ(input_float * 2, ((set_denormal_as_zero) ? 0.0f : denormal_float * 2));
      EXPECT_EQ(input_double * 2, ((set_denormal_as_zero) ? 0.0 : denormal_double * 2));
    }
  };

  test_denormal(true);
  test_denormal(false);
}
#ifdef _OPENMP
TEST(DenormalTest, OpenMPDenormalAsZeroTest) {
  auto test_denormal = [&](bool set_denormal_as_zero) {
    const float denormal_float = 1e-38f;
    const double denormal_double = 1e-308;
    const int test_size = 4;

    std::array<float, test_size> input_float;
    std::array<double, test_size> input_double;

    // When it returns false, disabling denormal as zero isn't supported,
    // so validation will be skipped
    bool set = SetDenormalAsZero(set_denormal_as_zero);
    if (set || !set_denormal_as_zero) {
      input_float.fill(denormal_float);
      input_double.fill(denormal_double);

      InitializeWithDenormalAsZero(set_denormal_as_zero);
#pragma omp parallel for
      for (auto i = 0; i < test_size; ++i) {
        input_float[i] *= 2;
        input_double[i] *= 2;
      }

      std::for_each(input_float.begin(), input_float.end(), [&](float f) {
        EXPECT_EQ(f, (set_denormal_as_zero) ? 0.0f : denormal_float * 2);
      });
      std::for_each(input_double.begin(), input_double.end(), [&](double d) {
        EXPECT_EQ(d, (set_denormal_as_zero) ? 0.0 : denormal_double * 2);
      });
    }
  };
  test_denormal(true);
  test_denormal(false);
}
#endif

}  // namespace test
}  // namespace onnxruntime
