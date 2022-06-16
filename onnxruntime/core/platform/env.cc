/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Portions Copyright (c) Microsoft Corporation

#include "core/platform/env.h"
#include "gsl/gsl"

#include <Windows.h>
#include <avrt.h>

namespace onnxruntime {

const char* const mm_task_name = "Audio";

void threadSetMmCharacteristics(HANDLE& mm_handle) {
  DWORD mmcssTaskIndex = 0;
  mm_handle = ::AvSetMmThreadCharacteristics(mm_task_name, &mmcssTaskIndex);
  if (!mm_handle) {
    auto error_code = ::GetLastError();
    ORT_THROW("AvSetMmThreadCharacteristicsA failed: ", std::system_category().message(error_code));
  }
}

void threadSetMmPriority(HANDLE mm_handle, int priority) {
  BOOL success = ::AvSetMmThreadPriority(mm_handle, static_cast<AVRT_PRIORITY>(priority));
  if (!success) {
    auto error_code = ::GetLastError();
    ORT_THROW("AvSetMmThreadPriority failed: ", std::system_category().message(error_code));
  }
}

void threadRevokeMmCharacteristics(HANDLE mm_handle) {
  BOOL ok = ::AvRevertMmThreadCharacteristics(mm_handle);
  if (!ok) {
    auto error_code = ::GetLastError();
    ORT_THROW("AvSetMmThreadPriority failed: ", std::system_category().message(error_code));
  }
}

Env::Env() = default;

}  // namespace onnxruntime

// This definition is provided to handle GSL failures in CUDA as
// not throwing exception but calling a user-defined handler.
// Otherwise gsl condition checks code does not compile even though
// gsl may not be used in CUDA specific code.
namespace gsl {
gsl_api void fail_fast_assert_handler(
    char const* const expression, char const* const message,
    char const* const file, int line) {
  ORT_ENFORCE(false, expression, file, line, message);
}
} // namespace gsl
