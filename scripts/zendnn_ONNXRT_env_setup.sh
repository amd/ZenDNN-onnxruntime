#*******************************************************************************
# Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#******************************************************************************

#!/bin/bash

#-----------------------------------------------------------------------------
#   zendnn_ONNXRT_env_setup.sh
#   Prerequisite: This script needs to run first to setup environment variables
#                 before ONNXRT setup
#
#   This script does following:
#   -Checks if important env variables are declared
#   -Checks and print version informations for following:
#       -make, gcc, g++, ld, python
#   -Sets important environment variables for benchmarking:
#       -OMP_NUM_THREADS, OMP_WAIT_POLICY, OMP_PROC_BIND
#   -Calls script to gather HW, OS, Kernel, Bios information
#   -exports LD_LIBRARY_PATH
#----------------------------------------------------------------------------

#Function to check mandatory prerequisites
function check_mandatory_prereqs() {
    if type -P make >/dev/null 2>&1;
    then
        echo "make is installed"
        echo `make -v | grep "GNU Make"`
    else
        echo "make is not installed, install make"
        return
    fi

    if type -P gcc >/dev/null 2>&1;
    then
        echo "gcc is installed"
        echo `gcc --version | grep "gcc "`
    else
        echo "gcc is not installed, install gcc"
        return
    fi

    if type -P g++ >/dev/null 2>&1;
    then
        echo "g++ is installed"
        echo `g++ --version | grep "g++ "`
    else
        echo "g++ is not installed, install g++"
        return
    fi

    if type -P ld >/dev/null 2>&1;
    then
        echo "ld is installed"
        echo `ld --version | grep "GNU ld "`
    else
        echo "ld is not installed, install ld"
        return
    fi

    if type -P python3 >/dev/null 2>&1;
    then
        echo "python3 is installed"
        echo `python3 --version`
    else
        echo "python3 is not installed, install python3"
        return
    fi
}

#Function to check optional prerequisites
function check_optional_prereqs() {
    if type -P lscpu >/dev/null 2>&1;
    then
        echo "lscpu is installed"
        echo `lscpu --version`
    else
        echo "lscpu is not installed, install lscpu"
    fi

    # Check if hwloc/lstopo-no-graphics is installed
    if type -P lstopo-no-graphics >/dev/null 2>&1;
    then
        echo "lstopo-no-graphics is installed"
        echo `lstopo-no-graphics --version`
    else
        echo "lstopo-no-graphics is not installed, install hwloc/lstopo-no-graphics"
    fi

    # Check if uname is installed
    if type -P uname >/dev/null 2>&1;
    then
        echo "uname is installed"
        echo `uname --version`
    else
        echo "uname is not installed, install uname"
    fi

    # Check if dmidecode is installed
    if type -P dmidecode >/dev/null 2>&1;
    then
        echo "dmidecode is installed"
        echo `dmidecode --version`
    else
        echo "dmidecode is not installed, install dmidecode"
    fi
}

#------------------------------------------------------------------------------
# Check if mandatory prerequisites are installed
echo "Checking mandatory prerequisites"
check_mandatory_prereqs

echo "Checking optional prerequisites"
# Check if optional prerequisites are installed
check_optional_prereqs
echo""

#------------------------------------------------------------------------------
if [ -z "$ZENDNN_LOG_OPTS" ];
then
    export ZENDNN_LOG_OPTS=ALL:0
    echo "ZENDNN_LOG_OPTS=$ZENDNN_LOG_OPTS"
else
    echo "ZENDNN_LOG_OPTS=$ZENDNN_LOG_OPTS"
fi

if [ -z "$OMP_NUM_THREADS" ];
then
    export OMP_NUM_THREADS=96
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
else
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
fi

if [ -z "$OMP_WAIT_POLICY" ];
then
    export OMP_WAIT_POLICY=ACTIVE
    echo "OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
else
    echo "OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
fi

if [ -z "$OMP_PROC_BIND" ];
then
    export OMP_PROC_BIND=FALSE
    echo "OMP_PROC_BIND=$OMP_PROC_BIND"
else
    echo "OMP_PROC_BIND=$OMP_PROC_BIND"
fi

#If the environment variable OMP_DYNAMIC is set to true, the OpenMP implementation
#may adjust the number of threads to use for executing parallel regions in order
#to optimize the use of system resources. ZenDNN depend on a number of threads
#which should not be modified by runtime, doing so can cause incorrect execution
export OMP_DYNAMIC=FALSE
echo "OMP_DYNAMIC=$OMP_DYNAMIC"

#Disable ONNXRT check for training ops and stop execution if any training ops
#found in ONNXRT graph. By default, its enabled
export ZENDNN_INFERENCE_ONLY=1
echo "ZENDNN_INFERENCE_ONLY=$ZENDNN_INFERENCE_ONLY"

# INT8 support  is disabled by default
export ZENDNN_INT8_SUPPORT=0
echo "ZENDNN_INT8_SUPPORT=$ZENDNN_INT8_SUPPORT"

# INT8 Relu6 fusion support is disabled by default
export ZENDNN_RELU_UPPERBOUND=0
echo "ZENDNN_RELU_UPPERBOUND=$ZENDNN_RELU_UPPERBOUND"

# ZENDNN_GEMM_ALGO is set to 3 by default
export ZENDNN_GEMM_ALGO=3
echo "ZENDNN_GEMM_ALGO=$ZENDNN_GEMM_ALGO"

#Use local copy of ZenDNN library source code when building ONNXRT
#Default is build from local source for development and verification.
#For release, export ZENDNN_ONNXRT_USE_LOCAL_ZENDNN=0
if [ -z "$ZENDNN_ONNXRT_USE_LOCAL_ZENDNN" ];
then
    export ZENDNN_ONNXRT_USE_LOCAL_ZENDNN=0
fi
echo "ZENDNN_ONNXRT_USE_LOCAL_ZENDNN=$ZENDNN_ONNXRT_USE_LOCAL_ZENDNN"

#Use local copy of BLIS library source code when building ONNXRT
#Default is download and build from external git for development and verification
#For verification with early drop pf BLIS, export ZENDNN_ONNXRT_USE_LOCAL_BLIS=1
if [ -z "$ZENDNN_ONNXRT_USE_LOCAL_BLIS" ];
then
    export ZENDNN_ONNXRT_USE_LOCAL_BLIS=0
fi
echo "ZENDNN_ONNXRT_USE_LOCAL_BLIS=$ZENDNN_ONNXRT_USE_LOCAL_BLIS"

export ZENDNN_ONNXRT_VERSION="1.15.1"
echo "ZENDNN_ONNXRT_VERSION: $ZENDNN_ONNXRT_VERSION"

export ZENDNN_ONNX_VERSION="1.14.0"
echo "ZENDNN_ONNX_VERSION: $ZENDNN_ONNX_VERSION"

# Primitive Caching Capacity
export ZENDNN_PRIMITIVE_CACHE_CAPACITY=1024
echo "ZENDNN_PRIMITIVE_CACHE_CAPACITY: $ZENDNN_PRIMITIVE_CACHE_CAPACITY"

# Enable primitive create and primitive execute logs. By default it is disabled
export ZENDNN_PRIMITIVE_LOG_ENABLE=0
echo "ZENDNN_PRIMITIVE_LOG_ENABLE: $ZENDNN_PRIMITIVE_LOG_ENABLE"

# Enable LIBM, By default, its disabled
export ZENDNN_ENABLE_LIBM=0
echo "ZENDNN_ENABLE_LIBM=$ZENDNN_ENABLE_LIBM"

#check if ZENDNN_LIBM_PATH is defined, otherwise return error
if [ "$ZENDNN_ENABLE_LIBM" = "1" ];
then
    if [ -z "$ZENDNN_LIBM_PATH" ];
    then
        echo "Error: Environment variable ZENDNN_LIBM_PATH needs to be set"
        return
    else
        echo "ZENDNN_LIBM_PATH: $ZENDNN_LIBM_PATH"
    fi
fi

# Flags for optimized execution of ONNXRT model
# Convolution Direct Algo with Blocked inputs and filter
export ZENDNN_CONV_ALGO=3
echo "ZENDNN_CONV_ALGO=$ZENDNN_CONV_ALGO"

export ZENDNN_CONV_ADD_FUSION_ENABLE=0
echo "ZENDNN_CONV_ADD_FUSION_ENABLE: $ZENDNN_CONV_ADD_FUSION_ENABLE"

export ZENDNN_RESNET_STRIDES_OPT1_ENABLE=0
echo "ZENDNN_RESNET_STRIDES_OPT1_ENABLE: $ZENDNN_RESNET_STRIDES_OPT1_ENABLE"

export ZENDNN_CONV_CLIP_FUSION_ENABLE=0
echo "ZENDNN_CONV_CLIP_FUSION_ENABLE: $ZENDNN_CONV_CLIP_FUSION_ENABLE"

export ZENDNN_BN_RELU_FUSION_ENABLE=0
echo "ZENDNN_BN_RELU_FUSION_ENABLE: $ZENDNN_BN_RELU_FUSION_ENABLE"

export ZENDNN_CONV_ELU_FUSION_ENABLE=0
echo "ZENDNN_CONV_ELU_FUSION_ENABLE: $ZENDNN_CONV_ELU_FUSION_ENABLE"

export ZENDNN_LN_FUSION_ENABLE=0
echo "ZENDNN_LN_FUSION_ENABLE $ZENDNN_LN_FUSION_ENABLE"

export ZENDNN_CONV_RELU_FUSION_ENABLE=1
echo "ZENDNN_CONV_RELU_FUSION_ENABLE: $ZENDNN_CONV_RELU_FUSION_ENABLE"

export ORT_ZENDNN_ENABLE_INPLACE_CONCAT=0
echo "ORT_ZENDNN_ENABLE_INPLACE_CONCAT: $ORT_ZENDNN_ENABLE_INPLACE_CONCAT"

export ZENDNN_ENABLE_MATMUL_BINARY_ELTWISE=1
echo "ZENDNN_ENABLE_MATMUL_BINARY_ELTWISE: $ZENDNN_ENABLE_MATMUL_BINARY_ELTWISE"

export ZENDNN_ENABLE_GELU=1
echo "ZENDNN_ENABLE_GELU: $ZENDNN_ENABLE_GELU"

export ZENDNN_ENABLE_FAST_GELU=1
echo "ZENDNN_ENABLE_FAST_GELU: $ZENDNN_ENABLE_FAST_GELU"

export ZENDNN_REMOVE_MATMUL_INTEGER=1
echo "ZENDNN_REMOVE_MATMUL_INTEGER: $ZENDNN_REMOVE_MATMUL_INTEGER"

export ZENDNN_MATMUL_ADD_FUSION_ENABLE=1
echo "ZENDNN_MATMUL_ADD_FUSION_ENABLE: $ZENDNN_MATMUL_ADD_FUSION_ENABLE"

# By default build Open sourced ONNXRuntime and ZenDNN
export ZENDNN_ONNXRUNTIME_USE_OPENSOURCE=1
echo "ZENDNN_ONNXRUNTIME_USE_OPENSOURCE: $ZENDNN_ONNXRUNTIME_USE_OPENSOURCE"

#-------------------------------------------------------------------------------
# Path related details
#-----------------------------------------------------------------------------
#Check if below declaration of ONNXRUNTIME_GIT_ROOT is correct
export ONNXRUNTIME_GIT_ROOT=$(pwd)
if [ -z "$ONNXRUNTIME_GIT_ROOT" ];
then
    echo "Error: Environment variable ONNXRUNTIME_GIT_ROOT needs to be set"
    echo "Error: \$ONNXRUNTIME_GIT_ROOT points to root of ZenDNN_ONNXRuntime repo"
    return
else
    [ ! -d "$ONNXRUNTIME_GIT_ROOT" ] && echo "Directory ZenDNN_ONNXRuntime DOES NOT exists!"
    echo "ONNXRUNTIME_GIT_ROOT: $ONNXRUNTIME_GIT_ROOT"
fi

#Change ZENDNN_PARENT_FOLDER as per need in future
#Current assumption, ONNXRuntime is located parallel to ZenDNN
cd ..
export ZENDNN_PARENT_FOLDER=$(pwd)
cd ~-

if [ -z "$ZENDNN_PARENT_FOLDER" ];
then
    echo "Error: Environment variable ZENDNN_PARENT_FOLDER needs to be set"
    echo "Error: \$ZENDNN_PARENT_FOLDER points to parent of ONNXRT repo"
else
    echo "ZENDNN_PARENT_FOLDER: $ZENDNN_PARENT_FOLDER"
fi

#Check if below declaration of ZENDNN_GIT_ROOT is correct
export ZENDNN_GIT_ROOT=$ZENDNN_PARENT_FOLDER/ZenDNN
if [ -z "$ZENDNN_GIT_ROOT" ];
then
    echo "Error: Environment variable ZENDNN_GIT_ROOT needs to be set"
    echo "Error: \$ZENDNN_GIT_ROOT points to root of ZENDNN repo"
    return
else
    [ ! -d "$ZENDNN_GIT_ROOT" ] && echo "Directory ZenDNN DOES NOT exists!"
    echo "ZENDNN_GIT_ROOT: $ZENDNN_GIT_ROOT"
fi

#Change ZENDNN_UTILS_GIT_ROOT as per need in future
export ZENDNN_UTILS_GIT_ROOT=$ZENDNN_PARENT_FOLDER/ZenDNN_utils
if [ -z "$ZENDNN_UTILS_GIT_ROOT" ];
then
    echo "Error: Environment variable ZENDNN_UTILS_GIT_ROOT needs to be set"
else
    [ ! -d "$ZENDNN_UTILS_GIT_ROOT" ] && echo "Directory ZenDNN_utils DOES NOT exists!"
    echo "ZENDNN_UTILS_GIT_ROOT: $ZENDNN_UTILS_GIT_ROOT"
fi

#Change ZENDNN_TOOLS_GIT_ROOT as per need in future
export ZENDNN_TOOLS_GIT_ROOT=$ZENDNN_PARENT_FOLDER/ZenDNN_tools
if [ -z "$ZENDNN_TOOLS_GIT_ROOT" ];
then
    echo "Error: Environment variable ZENDNN_TOOLS_GIT_ROOT needs to be set"
else
    [ ! -d "$ZENDNN_TOOLS_GIT_ROOT" ] && echo "Directory ZenDNN_tools DOES NOT exists!"
    echo "ZENDNN_TOOLS_GIT_ROOT: $ZENDNN_TOOLS_GIT_ROOT"
fi

export BENCHMARKS_GIT_ROOT=$ZENDNN_PARENT_FOLDER/benchmarks
echo "BENCHMARKS_GIT_ROOT: $BENCHMARKS_GIT_ROOT"

# Export PATH and LD_LIBRARY_PATH for ZenDNN
export PATH=$PATH:$ZENDNN_GIT_ROOT/_out/tests
if [ "$ZENDNN_ENABLE_LIBM" = "1" ];
then
    export LD_LIBRARY_PATH=$ZENDNN_LIBM_PATH/lib/:$LD_LIBRARY_PATH
fi

export LD_LIBRARY_PATH=$ZENDNN_GIT_ROOT/external/googletest/lib:$LD_LIBRARY_PATH

echo "LD_LIBRARY_PATH: "$LD_LIBRARY_PATH
#-------------------------------------------------------------------------------
# HW, HW architecture, Cache, OS, Kernel details
#-----------------------------------------------------------------------------
# Go to ZENDNN_GIT_ROOT
cd $ZENDNN_GIT_ROOT

chmod u+x scripts/gather_hw_os_kernel_bios_info.sh
echo "scripts/gather_hw_os_kernel_bios_info.sh"
source scripts/gather_hw_os_kernel_bios_info.sh true > system_hw_os_kernel_bios_info.txt

#-------------------------------------------------------------------------------
# Go to ONNXRUNTIME_GIT_ROOT
cd $ONNXRUNTIME_GIT_ROOT
echo -e "\n"
echo "Please set below environment variables explicitly as per the platform you are using!!"
echo -e "\tOMP_NUM_THREADS, GOMP_CPU_AFFINITY"
echo -e "\n"
