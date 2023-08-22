::*******************************************************************************
:: Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
::
:: Permission is hereby granted, free of charge, to any person obtaining
:: a copy of this software and associated documentation files (the
:: "Software"), to deal in the Software without restriction, including
:: without limitation the rights to use, copy, modify, merge, publish,
:: distribute, sublicense, and/or sell copies of the Software, and to
:: permit persons to whom the Software is furnished to do so, subject to
:: the following conditions:
::
:: The above copyright notice and this permission notice shall be
:: included in all copies or substantial portions of the Software.
::
:: THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
:: EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
:: MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
:: NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
:: LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
:: OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
:: WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
::
::*******************************************************************************

::-----------------------------------------------------------------------------
::   zendnn_ONNXRT_env_setup_win.bat
::   Prerequisite: This script needs to run first to setup environment variables
::                 before ONNXRT setup
::
::   This script does following:
::   -sets the environment variables , if they are present they will bre replaced
::   -Sets important environment variables for benchmarking:
::       -OMP_NUM_THREADS, OMP_WAIT_POLICY, OMP_PROC_BIND
::----------------------------------------------------------------------------

@echo off

if not defined ZENDNN_LOG_OPTS set ZENDNN_LOG_OPTS=ALL:0
echo ZENDNN_LOG_OPTS=%ZENDNN_LOG_OPTS%

if not defined OMP_NUM_THREADS set OMP_NUM_THREADS=64
echo OMP_NUM_THREADS=%OMP_NUM_THREADS%

if not defined OMP_WAIT_POLICY set OMP_WAIT_POLICY=ACTIVE
echo OMP_WAIT_POLICY=%OMP_WAIT_POLICY%

if not defined OMP_PROC_BIND set OMP_PROC_BIND=FALSE
echo OMP_PROC_BIND=%OMP_PROC_BIND%

:: If the environment variable OMP_DYNAMIC is set to true, the OpenMP implementation
:: may adjust the number of threads to use for executing parallel regions in order
:: to optimize the use of system resources. ZenDNN depend on a number of threads
:: which should not be modified by runtime, doing so can cause incorrect execution
set OMP_DYNAMIC=FALSE
echo OMP_DYNAMIC=%OMP_DYNAMIC%

::Disable ONNXRT check for training ops and stop execution if any training ops
::found in ONNXRT graph. By default, its enabled
set ZENDNN_INFERENCE_ONLY=1
echo ZENDNN_INFERENCE_ONLY=%ZENDNN_INFERENCE_ONLY%

:: INT8 support  is disabled by default
set ZENDNN_INT8_SUPPORT=0
echo ZENDNN_INT8_SUPPORT=%ZENDNN_INT8_SUPPORT%

:: INT8 Relu6 fusion support is disabled by default
set ZENDNN_RELU_UPPERBOUND=0
echo ZENDNN_RELU_UPPERBOUND=%ZENDNN_RELU_UPPERBOUND%

::ZENDNN_GEMM_ALGO is set to 3 by default
set ZENDNN_GEMM_ALGO=4
echo ZENDNN_GEMM_ALGO=%ZENDNN_GEMM_ALGO%
:: Use local copy of ZenDNN library source code when building ONNXRT
:: Default is build from local source for development and verification.
:: For release, set ZENDNN_ONNXRT_USE_LOCAL_ZENDNN=0
if not defined ZENDNN_ONNXRT_USE_LOCAL_ZENDNN (
    set ZENDNN_ONNXRT_USE_LOCAL_ZENDNN=1
)
echo ZENDNN_ONNXRT_USE_LOCAL_ZENDNN=%ZENDNN_ONNXRT_USE_LOCAL_ZENDNN%

:: Use local copy of BLIS library source code when building ONNXRT
:: Default is download and build from external git for development and verification
:: For verification with early drop pf BLIS, export ZENDNN_ONNXRT_USE_LOCAL_BLIS=1
if not defined ZENDNN_ONNXRT_USE_LOCAL_BLIS (
    set ZENDNN_ONNXRT_USE_LOCAL_BLIS=0
)
echo ZENDNN_ONNXRT_USE_LOCAL_BLIS=%ZENDNN_ONNXRT_USE_LOCAL_BLIS%

set ZENDNN_ONNXRT_VERSION=1.15.1
echo ZENDNN_ONNXRT_VERSION=%ZENDNN_ONNXRT_VERSION%

set ZENDNN_ONNX_VERSION=1.14.0
echo ZENDNN_ONNX_VERSION=%ZENDNN_ONNX_VERSION%

:: Primitive Caching Capacity
set ZENDNN_PRIMITIVE_CACHE_CAPACITY=1024
echo ZENDNN_PRIMITIVE_CACHE_CAPACITY=%ZENDNN_PRIMITIVE_CACHE_CAPACITY%

:: Enable primitive create and primitive execute logs. By default it is disabled
set ZENDNN_PRIMITIVE_LOG_ENABLE=0
echo ZENDNN_PRIMITIVE_LOG_ENABLE=%ZENDNN_PRIMITIVE_LOG_ENABLE%

:: Enable LIBM, By default, its disabled
set ZENDNN_ENABLE_LIBM=0
echo ZENDNN_ENABLE_LIBM=%ZENDNN_ENABLE_LIBM%

:: Flags for optimized execution of ONNXRT model
:: Convolution Direct Algo with Blocked inputs and filter
set ZENDNN_CONV_ALGO=3
echo ZENDNN_CONV_ALGO=%ZENDNN_CONV_ALGO%

set ZENDNN_CONV_ADD_FUSION_ENABLE=0
echo ZENDNN_CONV_ADD_FUSION_ENABLE=%ZENDNN_CONV_ADD_FUSION_ENABLE%

set ZENDNN_RESNET_STRIDES_OPT1_ENABLE=0
echo ZENDNN_RESNET_STRIDES_OPT1_ENABLE=%ZENDNN_RESNET_STRIDES_OPT1_ENABLE%

set ZENDNN_CONV_CLIP_FUSION_ENABLE=0
echo ZENDNN_CONV_CLIP_FUSION_ENABLE=%ZENDNN_CONV_CLIP_FUSION_ENABLE%

set ZENDNN_BN_RELU_FUSION_ENABLE=0
echo ZENDNN_BN_RELU_FUSION_ENABLE=%ZENDNN_BN_RELU_FUSION_ENABLE%

set ZENDNN_CONV_ELU_FUSION_ENABLE=0
echo ZENDNN_CONV_ELU_FUSION_ENABLE=%ZENDNN_CONV_ELU_FUSION_ENABLE%

set ZENDNN_LN_FUSION_ENABLE=0
echo ZENDNN_LN_FUSION_ENABLE%ZENDNN_LN_FUSION_ENABLE%

set ZENDNN_CONV_RELU_FUSION_ENABLE=1
echo ZENDNN_CONV_RELU_FUSION_ENABLE=%ZENDNN_CONV_RELU_FUSION_ENABLE%

set ORT_ZENDNN_ENABLE_INPLACE_CONCAT=0
echo ORT_ZENDNN_ENABLE_INPLACE_CONCAT=%ORT_ZENDNN_ENABLE_INPLACE_CONCAT%

set ZENDNN_ENABLE_MATMUL_BINARY_ELTWISE=1
echo ZENDNN_ENABLE_MATMUL_BINARY_ELTWISE=%ZENDNN_ENABLE_MATMUL_BINARY_ELTWISE%

set ZENDNN_ENABLE_GELU=1
echo ZENDNN_ENABLE_GELU=%ZENDNN_ENABLE_GELU%

set ZENDNN_ENABLE_FAST_GELU=1
echo ZENDNN_ENABLE_FAST_GELU=%ZENDNN_ENABLE_FAST_GELU%

set ZENDNN_REMOVE_MATMUL_INTEGER=1
echo ZENDNN_REMOVE_MATMUL_INTEGER=%ZENDNN_REMOVE_MATMUL_INTEGER%

set ZENDNN_MATMUL_ADD_FUSION_ENABLE=1
echo ZENDNN_MATMUL_ADD_FUSION_ENABLE=%ZENDNN_MATMUL_ADD_FUSION_ENABLE%

set KMP_DUPLICATE_LIB_OK=TRUE
echo KMP_DUPLICATE_LIB_OK=%KMP_DUPLICATE_LIB_OK%

set ZENDNN_ONNXRUNTIME_USE_OPENSOURCE=1
echo ZENDNN_ONNXRUNTIME_USE_OPENSOURCE=%ZENDNN_ONNXRUNTIME_USE_OPENSOURCE%

::Check if below declaration of ONNXRUNTIME_GIT_ROOT is correct
set ONNXRUNTIME_GIT_ROOT=%cd%
if not defined ONNXRUNTIME_GIT_ROOT (
    echo Error: Environment variable ONNXRUNTIME_GIT_ROOT needs to be set
    echo Error: \ONNXRUNTIME_GIT_ROOT points to root of ZENDNN repo
    exit
) else (
    if exist "%ONNXRUNTIME_GIT_ROOT%\" (
        echo Directory ZenDNN exists!
    ) else (
        echo Directory ZenDNN DOES NOT exists!
    )
    echo ONNXRUNTIME_GIT_ROOT=%ONNXRUNTIME_GIT_ROOT%
)

cd ..

::Check if below declaration of ZENDNN_GIT_ROOT is correct
set ZENDNN_GIT_ROOT=%cd%\ZenDNN
if not defined ZENDNN_GIT_ROOT (
    echo Error: Environment variable ZENDNN_GIT_ROOT needs to be set
    echo Error: \ZENDNN_GIT_ROOT points to root of ZENDNN repo
    exit
) else (
    if exist "%ZENDNN_GIT_ROOT%\" (
        echo Directory ZenDNN exists!
    ) else (
        echo Directory ZenDNN DOES NOT exists!
    )
    echo ZENDNN_GIT_ROOT=%ZENDNN_GIT_ROOT%
)

::Change ZENDNN_UTILS_GIT_ROOT as per need in future
set ZENDNN_UTILS_GIT_ROOT=%cd%\ZenDNN_utils
if not defined ZENDNN_UTILS_GIT_ROOT (
    echo Error: Environment variable ZENDNN_UTILS_GIT_ROOT needs to be set
    echo Error: \ZENDNN_UTILS_GIT_ROOT points to root of ZENDNN repo
    pause
    exit
) else (
    if exist "%ZENDNN_UTILS_GIT_ROOT%\" (
        echo Directory ZenDNN_utils exists!
    ) else (
        echo Directory ZenDNN_utils DOES NOT exists!
    )
    echo ZENDNN_UTILS_GIT_ROOT=%ZENDNN_UTILS_GIT_ROOT%
)

::Change ZENDNN_TOOLS_GIT_ROOT as per need in future
set ZENDNN_TOOLS_GIT_ROOT=%cd%\ZenDNN_tools
if not defined ZENDNN_TOOLS_GIT_ROOT (
    echo Error: Environment variable ZENDNN_TOOLS_GIT_ROOT needs to be set
    echo Error: \ZENDNN_TOOLS_GIT_ROOT points to root of ZENDNN repo
    pause
    exit
) else (
    if exist "%ZENDNN_TOOLS_GIT_ROOT%\" (
        echo Directory ZenDNN_tools exists!
    ) else (
        echo Directory ZenDNN_tools DOES NOT exists!
    )
    echo ZENDNN_TOOLS_GIT_ROOT=%ZENDNN_TOOLS_GIT_ROOT%
)

::Change ZENDNN_PARENT_FOLDER as per need in future
::Current assumption, ONNXRT is located parallel to ZenDNN
set ZENDNN_PARENT_FOLDER=%cd%
if defined ZENDNN_PARENT_FOLDER (
    echo ZENDNN_PARENT_FOLDER=%ZENDNN_PARENT_FOLDER%
) else (
    set ZENDNN_PARENT_FOLDER=%cd%
    echo ZENDNN_PARENT_FOLDER=%ZENDNN_PARENT_FOLDER%
)

set BENCHMARKS_GIT_ROOT=%ZENDNN_PARENT_FOLDER%\benchmarks
echo BENCHMARKS_GIT_ROOT=%BENCHMARKS_GIT_ROOT%

cd %ONNXRUNTIME_GIT_ROOT%
echo:
echo Please set below environment variable explicitly as per the platform you are using!!
echo:
echo     set OMP_NUM_THREADS=X
echo:
