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

#include <unordered_map>
#include <mutex>

#include "zendnn_util.h"
#include "zendnn.hpp"
#include "zendnn_types.h"
#include "core/common/common.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace ort_zendnn {
namespace zendnn_util {

enum GPUInfo {
    AVALIBLITY,
    BF16SUPPORT
};

std::once_flag flag1, flag2;
// zendnn::engin::kind::gpu represents actual HW and we want to limit how much we instantiate the hardware
// This code has been designed so that it can be called multiple times.  The engine will only be created
// the first call.
// Wrapped in the `call_once` code we create a gpu engine.
//   if GPU engine is successful created set gpuRuntimeFound=true
//   Use the engine to create a bfloat16 matmul primitive if successful set gpuBF16Supported=true
bool GetGPUInfo(GPUInfo gpu_info) {
    static bool gpuRuntimeFound = false;
    static bool gpuBF16Supported = false;
#ifdef ZENDNN_GPU_RUNTIME
#if (ZENDNN_GPU_RUNTIME == ZENDNN_RUNTIME_OCL) || (ZENDNN_GPU_RUNTIME == ZENDNN_RUNTIME_SYCL)
    std::call_once(flag1, []() {
        zendnn::engine gpu_engine;
        if (zendnn_engine_get_count(zendnn_engine_kind_t::zendnn_gpu)) {
            gpu_engine = zendnn::engine(zendnn::engine::kind::gpu, 0);
        }
        if (gpu_engine) {
            gpuRuntimeFound = true;
            // attempt to make a zendnn::matmul::desc. If we are able to successfully make a bf16 matmul::desc
            // assume the GPU supports all BF16 operations.
            zendnn::primitive_attr attr;
            attr.set_scales_mask(ZENDNN_ARG_SRC, 0);
            attr.set_zero_points_mask(ZENDNN_ARG_SRC, /* mask */ 0);
            auto src0_md = zendnn::memory::desc({1, 1}, zendnn::memory::data_type::bf16,
                                                zendnn::memory::format_tag::ab);
            auto src1_md = zendnn::memory::desc({1, 1}, zendnn::memory::data_type::bf16,
                                                zendnn::memory::format_tag::ab);
            auto dst_md = zendnn::memory::desc({1, 1}, zendnn::memory::data_type::bf16,
                                               zendnn::memory::format_tag::ab);
            try {
                auto matmul_pd = zendnn::matmul::primitive_desc(gpu_engine, src0_md, src1_md,
                                 dst_md, attr);
                gpuBF16Supported = true;
            }
            catch (const zendnn::error &e) {
                if (e.status == zendnn_unimplemented) {
                    gpuBF16Supported = false;
                }
            }
        }
    });
#endif  // (ZENDNN_GPU_RUNTIME == ZENDNN_RUNTIME_OCL) || (ZENDNN_GPU_RUNTIME == ZENDNN_RUNTIME_SYCL)
#endif  // defined(ZENDNN_GPU_RUNTIME)
    switch (gpu_info) {
    case AVALIBLITY: {
        return gpuRuntimeFound;
        break;
    }
    case BF16SUPPORT: {
        return gpuBF16Supported;
        break;
    }
    default:
        return false;
        break;
    }
}

bool IsGPURuntimeAvalible() {
    return GetGPUInfo(AVALIBLITY);
}

bool IsGPUBF16Supported() {
    return GetGPUInfo(BF16SUPPORT);
}

bool IsBF16Supported() {
    static bool use_all_bf16_hardware = false;
    if (IsGPURuntimeAvalible() && IsGPUBF16Supported()) {
        return true;
    }
    std::call_once(flag2, []() {
        const std::string bf16_env =
            onnxruntime::GetEnvironmentVar("ORT_ZENDNN_USE_ALL_BF16_HW");
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

zendnn::algorithm OrtOperatorToZendnnAlgorithm(std::string op) {
    std::unordered_map<std::string, zendnn::algorithm> operator_to_algorithm = {
        // binary algorithms
        {"Add", zendnn::algorithm::binary_add},
        {"Mul", zendnn::algorithm::binary_mul},
        {"Sub", zendnn::algorithm::binary_sub},
        {"Div", zendnn::algorithm::binary_div},
        // qbinary algorithms
        {"QLinearAdd", zendnn::algorithm::binary_add},
        // eltwise algorithms
        {"Abs", zendnn::algorithm::eltwise_abs},
        {"BiasGelu", zendnn::algorithm::eltwise_gelu_erf},
        {"Elu", zendnn::algorithm::eltwise_elu},  // algorithm requires alpha value
        {"Equal", zendnn::algorithm::binary_eq},
        {"Exp", zendnn::algorithm::eltwise_exp},
        {"FastGelu", zendnn::algorithm::eltwise_gelu_tanh},
        {"Gelu", zendnn::algorithm::eltwise_gelu_erf},
        {"Greater", zendnn::algorithm::binary_gt},
        {"GreaterOrEqual", zendnn::algorithm::binary_ge},
        {"LeakyRelu", zendnn::algorithm::eltwise_relu},  // algorithm requires alpha value
        {"Less", zendnn::algorithm::binary_lt},
        {"LessOrEqual", zendnn::algorithm::binary_le},
        {"Log", zendnn::algorithm::eltwise_log},
        {"Relu", zendnn::algorithm::eltwise_relu},
        {"Round", zendnn::algorithm::eltwise_round},
        // ZenDNN eltwise_logistic is defined as 1/(1 + exp(-x)) which matches the definition of "Sigmoid" in ONNX
        {"Sigmoid", zendnn::algorithm::eltwise_logistic},
        // ZenDNN eltwise_soft_relu is defined as ln(1 + exp(x)) which matches the definition of "Softplus" in ONNX
        {"Softplus", zendnn::algorithm::eltwise_soft_relu},
        {"Sqrt", zendnn::algorithm::eltwise_sqrt},
        {"Clip", zendnn::algorithm::eltwise_clip},
        {"Tanh", zendnn::algorithm::eltwise_tanh},
        // Reduction algorithms
        {"ReduceL1", zendnn::algorithm::reduction_norm_lp_power_p_sum},
        {"ReduceL2", zendnn::algorithm::reduction_norm_lp_sum},
        {"ReduceLogSum", zendnn::algorithm::reduction_sum},
        {"ReduceLogSumExp", zendnn::algorithm::reduction_sum},
        {"ReduceMax", zendnn::algorithm::reduction_max},
        {"ReduceMean", zendnn::algorithm::reduction_mean},
        {"ReduceMin", zendnn::algorithm::reduction_min},
        {"ReduceProd", zendnn::algorithm::reduction_mul},
        {"ReduceSum", zendnn::algorithm::reduction_sum},
        {"ReduceSumSquare", zendnn::algorithm::reduction_sum}
    };

    auto found = operator_to_algorithm.find(op);
    if (found == operator_to_algorithm.end()) {
        ORT_THROW("op type not supported");
    }
    else {
        return found->second;
    }
}

}  // namespace zendnn_util
}  // namespace ort_zendnn
}  // namespace onnxruntime
