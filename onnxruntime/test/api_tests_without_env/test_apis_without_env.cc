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

#ifndef USE_ONNXRUNTIME_DLL
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <google/protobuf/message_lite.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif

#include "gtest/gtest.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/abi_session_options_impl.h"

TEST(TestSessionOptions, SetIntraOpNumThreadsWithoutEnv) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(48);
  const auto* ort_session_options = (const OrtSessionOptions*)session_options;
#ifdef _OPENMP
  ASSERT_EQ(ort_session_options->value.intra_op_param.thread_pool_size, 0);
#else
  ASSERT_EQ(ort_session_options->value.intra_op_param.thread_pool_size, 48);
#endif
}

int main(int argc, char** argv) {
  int status = 0;
  ORT_TRY {
    ::testing::InitGoogleTest(&argc, argv);
    status = RUN_ALL_TESTS();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      std::cerr << ex.what();
      status = -1;
    });
  }

#ifndef USE_ONNXRUNTIME_DLL
  // make memory leak checker happy
  ::google::protobuf::ShutdownProtobufLibrary();
#endif
  return status;
}
