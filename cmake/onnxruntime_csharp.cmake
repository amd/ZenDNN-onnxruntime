#**************************************************************************************
# Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
#**************************************************************************************

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#**************************************************************************************
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#**************************************************************************************

set (CSHARP_ROOT ${PROJECT_SOURCE_DIR}/../csharp)
set (CSHARP_MASTER_TARGET OnnxRuntime.CSharp)
set (CSHARP_MASTER_PROJECT ${CSHARP_ROOT}/OnnxRuntime.CSharp.proj)
if (onnxruntime_RUN_ONNX_TESTS)
  set (CSHARP_DEPENDS onnxruntime ${test_data_target})
else()
  set (CSHARP_DEPENDS onnxruntime)
endif()

if (onnxruntime_USE_CUDA)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_CUDA;")
endif()

if (onnxruntime_USE_DNNL)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_DNNL;")
endif()

if (onnxruntime_USE_ZENDNN)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_ZENDNN;")
endif()

if (onnxruntime_USE_DML)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_DML;")
endif()

if (onnxruntime_USE_MIGRAPHX)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_MIGRAPHX;")
endif()

if (onnxruntime_USE_NNAPI_BUILTIN)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_NNAPI;")
endif()

if (onnxruntime_USE_TVM)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_TVM,")
endif()

if (onnxruntime_USE_OPENVINO)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_OPENVINO;")
endif()

if (onnxruntime_USE_ROCM)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_ROCM;")
endif()

if (onnxruntime_USE_TENSORRT)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_TENSORRT;")
endif()

if (onnxruntime_USE_XNNPACK)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "USE_XNNPACK;")
endif()

if (onnxruntime_ENABLE_TRAINING_APIS)
  STRING(APPEND CSHARP_PREPROCESSOR_DEFINES "__TRAINING_ENABLED_NATIVE_BUILD__;")
endif()

include(CSharpUtilities)

# generate Directory.Build.props
set(DIRECTORY_BUILD_PROPS_COMMENT "WARNING: This is a generated file, please do not check it in!")
configure_file(${CSHARP_ROOT}/Directory.Build.props.in
               ${CSHARP_ROOT}/Directory.Build.props
               @ONLY)
