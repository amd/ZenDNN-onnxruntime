/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

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

#include "test/common/zendnn_op_test_utils.h"
#include "core/common/cpuid_info.h"
#include "core/platform/env.h"
#include <mutex>

namespace onnxruntime {
namespace test {
bool ZendnnSupportedGpuFound() {
// This assumes that if the code was built using the ZENDNN_GPU_RUNTIME then you have GPU support.
// It is possible this is not true.
// If a compatable GPU is not found at runtime the ZenDNN ep will run the bfloat16 code on the CPU.
// If there is no GPU or CPU support for bfloat16 this assumption may cause unit tests to fail.
// They will fail with a "Could not find an implementation for [operator]" error.
#if defined(ZENDNN_GPU_RUNTIME)
  return true;
#else
  return false;
#endif
}
std::once_flag once_flagZ;

bool ZendnnHasBF16Support() {
  if (ZendnnSupportedGpuFound()) {
    return true;
  }
  // HasAVX512Skylake checks for AVX512BW which can run bfloat16 but
  // is slower than float32 by 3x to 4x.
  static bool use_all_bf16_hardware = false;
  std::call_once(once_flagZ, []() {
    const std::string bf16_env = Env::Default().GetEnvironmentVar("ORT_ZENDNN_USE_ALL_BF16_HW");
    if (!bf16_env.empty()) {
      use_all_bf16_hardware = (std::stoi(bf16_env) == 0 ? false : true);
    }
  });

  // HasAVX512Skylake checks for AVX512BW which can run bfloat16 but
  // is slower than float32 by 3x to 4x.
  // By default the AVX512BW ISA is not used. It is still useful for validation
  // so it can be enabled by setting the environment variable ORT_ZENDNN_USE_ALL_BF16_HW=1
  if (use_all_bf16_hardware && CPUIDInfo::GetCPUIDInfo().HasAVX512Skylake()) {
    return true;
  }

  // If AVX512-BF16 or AMX-BF16 exist then bfloat16 ops are HW accelerated
  if (CPUIDInfo::GetCPUIDInfo().HasAVX512_BF16() ||
      CPUIDInfo::GetCPUIDInfo().HasAMX_BF16()) {
    return true;
  }
  return false;
}
}  // namespace test
}  // namespace onnxruntime
