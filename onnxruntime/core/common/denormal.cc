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

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
#define PLATFORM_X86
#endif

#if defined(PLATFORM_X86)
#if defined(_MSC_VER)
#include <intrin.h>
#define DENORMAL_INTRINC
// intrins headers at gcc 4.8 and older are not usable without compiler flags.
// clang claims gnuc 4.2, but it has a proper intrin header.
#elif defined(__clang__) || (defined(__GNUC__) && ((__GNUC__ >= 5) || ((__GNUC__ == 4) && (__GNUC_MINOR__ > 8))))
#include <cpuid.h>
#include <x86intrin.h>
#define DENORMAL_INTRINC
#endif
#endif

#include "core/common/common.h"
#include "core/common/cpuid_info.h"
#include "core/common/denormal.h"

namespace onnxruntime {

bool SetDenormalAsZero(bool on) {
#ifdef DENORMAL_INTRINC
  if (CPUIDInfo::GetCPUIDInfo().HasSSE3()) {
    if (on) {
      _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
      _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    } else {
      _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
      _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    }
    return true;
  }
#else
  ORT_UNUSED_PARAMETER(on);
#endif
  return false;
}

#ifdef _OPENMP
// To execute an initialization for each openmp thread, use a property of the firstprivate clause:
// "the initialization or construction of the given variable happens as if it were done once per thread,
// prior to the thread's execution of the construct".
class DenormalAsZeroInitializer {
 public:
  explicit DenormalAsZeroInitializer(bool on) : on_(on) {}

  // Working as initializer per each openmp thread.
  DenormalAsZeroInitializer(const DenormalAsZeroInitializer& init) : on_(init.on_) {
    SetDenormalAsZero(on_);
  }

 private:
  bool on_;
};

void InitializeWithDenormalAsZero(bool on) {
  DenormalAsZeroInitializer init(on);
// Each openmp thread calls DenormalAsZeroInitializer's copy constructor by firstprivate.
// Even if loop count is less than the maximum number of openmp threads, all openmp threads are initialized here.
#pragma omp parallel for firstprivate(init)
  for (auto i = 0; i < 1; ++i) {
  }
}
#endif

}  // namespace onnxruntime
