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

#include "zendnn_layernorm.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnLayerNorm::ZendnnLayerNorm() {}

/*
Layer Normalization and Skip Layer Normalization implementation:
Layer Norm:
  Inputs:
    0) X - Input Tensor
    1) G - Gamma Tensor (Scaling factor in the LN formula)
    2) B - Bias Tensor (Shift value in the LN formula. Optional)
  Outputs:
    0) Y - Output Tensor
    1) M - Mean Tensor (Optional)
    2) I - Inverse std Tensor (Optional)

               +-----------+
(X) ---------->+           +----------> (Y)
               |           |
(G) ---------->+ LayerNorm +----------> (M)
               |           |
(B) ---------->+           +----------> (I)
               +-----------+



Skip Layer Norm:
  Inputs:
    0) X - Input Tensor
    1) S - Skip Tensor
    2) G - Gamma Tensor (Scaling factor in the LN formula)
    4) E - Beta Tensor (Shift value in the LN formula. Optional)
    4) B - Bias Tensor (Bias when adding X + S. Optional)
  Outputs:
    0) Y - Output Tensor
    1) M - Mean Tensor (Optional)
    2) I - Inverse std Tensor (Optional)

               +-----------+
(X) ---------->+           |                   +-----------+
               |           |   (X + S + B)     |           |
(S) ---------->+ BuildSLN  +------------------>+           +----------> (Y)
               |           |                   |           |
(B) ---------->+           |      (G) -------->+ LayerNorm +----------> (M)
               +-----------+                   |           |
                                  (E) -------->+           +----------> (I)
                                               |           |
                                               +-----------+

Attributes (epsilon)
*/
void ZendnnLayerNorm::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                      ZendnnNode &node) {

    // Get engine
    auto zendnn_engine = sp.GetEngine();

    using dt =  zendnn::memory::data_type;
    bool zendnn_enable_bf16 = false;
    const std::string enable_bf16_env = onnxruntime::GetEnvironmentVar("ZENDNN_ONNXRT_ENABLE_BF16_SUPPORT");
    if (!enable_bf16_env.empty())
        zendnn_enable_bf16 = (std::stoi(enable_bf16_env) == 0 ? false : true);

    // Make sure every input's dimension follows the spec
    ValidateDims(sp, node);

    // Optional input flag
    bool shift_exists = false;

    // Input positions
    int shift_pos, scale_pos;

    // Get src mem
    auto src_mem = sp.GetMemory(node.Input(IN_INPUT));
    auto data_type = src_mem.get_desc().data_type();
    if(zendnn_enable_bf16 && data_type != dt::bf16)
    {
        auto src_mem_dims = src_mem.get_desc().dims();
        auto src_bf16_mem = zendnn::memory(
                zendnn::memory::desc(src_mem_dims,dt::bf16,sp.GetZendnnFormat(src_mem_dims.size())),
                zendnn_engine);
        sp.AddPrimitive(zendnn::reorder(src_mem, src_bf16_mem),{{ZENDNN_ARG_SRC, src_mem},
                                                                {ZENDNN_ARG_DST, src_bf16_mem}});
        src_mem = src_bf16_mem;
    }
    auto src_md = src_mem.get_desc();

    // This contains the layer norm op and its parameters
    ln_components op_comps;
    if (node.OpType() == "SkipLayerNormalization") {

        // Check if shift is available
        shift_exists = node.Input(IN_BETA).Exists();

        // Fix positions for arguments
        shift_pos = IN_BETA;
        scale_pos = IN_SLN_GAMMA;

        // Build SLN and get modified mem
        src_mem = BuildSLN(sp, node, zendnn_engine);

    }
    else if (node.OpType() == "LayerNormalization") {

        // Check if shift is available
        shift_exists = node.Input(IN_LN_BIAS).Exists();

        // Fix positions for arguments
        shift_pos = IN_LN_BIAS;
        scale_pos = IN_LN_GAMMA;

        // Move the src to GPU if needed
        src_mem = sp.GetMemoryAndReshape(node.Input(IN_INPUT), src_mem.get_desc(),
                                         zendnn_engine);

    }
    else {
        ORT_THROW("Unknown LayerNormalization flavor");
    }

    // X = LayerNornm(X)
    // Check if we are training and need the extra outputs for backprop
    zendnn::prop_kind prop_kind;
#if 0 //defined(ENABLE_TRAINING)
    prop_kind = zendnn::prop_kind::forward_training;
#else
    prop_kind = zendnn::prop_kind::forward_inference;
#endif  // ENABLE_TRAINING

    // If beta is available use shift, else only scale
    zendnn::normalization_flags op_flags = zendnn::normalization_flags::use_scale;
    if (shift_exists) {
        op_flags |= zendnn::normalization_flags::use_shift;
    }

    // Get epsilon to avoid zero division
    float epsilon = GetEpsilon(node);
    // Operation desciptor
    auto lnorm_desc = zendnn::layer_normalization_forward::desc(prop_kind, src_md,
                      epsilon, op_flags);
    // Primitive desciptor
    auto lnorm_pd = zendnn::layer_normalization_forward::primitive_desc(lnorm_desc,
                    zendnn_engine);
    // Primitive
    auto lnorm_prim = zendnn::layer_normalization_forward(lnorm_pd);

    // Get gamma
    auto gamma_mem = sp.GetMemory(node.Input(scale_pos));
    gamma_mem = sp.GetMemoryAndReshape(node.Input(scale_pos), gamma_mem.get_desc(),
                                       zendnn_engine);

    sp.IncMemoryRefCount(src_mem);
    // Define primitive arguments
    std::unordered_map<int, zendnn::memory> lnorm_args = {{ZENDNN_ARG_SRC, src_mem},
        {ZENDNN_ARG_SCALE, gamma_mem},
        {ZENDNN_ARG_DST, src_mem}
    };

    // Get Beta and add shift if available
    if (shift_exists) {
        auto beta_mem = sp.GetMemory(node.Input(shift_pos));
        beta_mem = sp.GetMemoryAndReshape(node.Input(shift_pos), beta_mem.get_desc(),
                                          zendnn_engine);
        lnorm_args.insert({ZENDNN_ARG_SHIFT, beta_mem});
    }

// Check outputs used for training
#if 0 //defined(ENABLE_TRAINING)
    // If Mean exists
    if (node.OutputCount() > 1) {
        if (node.Output(OUT_MEAN).Exists()) {
            auto mean_mem = zendnn::memory(lnorm_pd.mean_desc(), zendnn_engine);
            lnorm_args.insert({ZENDNN_ARG_MEAN, mean_mem});
            sp.SetMemory(node.Output(OUT_MEAN), mean_mem);
        }
        // If Variance exists
        if (node.Output(OUT_INV_STD_VAR).Exists()) {
            auto variance_mem = zendnn::memory(lnorm_pd.variance_desc(), zendnn_engine);
            lnorm_args.insert({ZENDNN_ARG_VARIANCE, variance_mem});
            sp.SetMemory(node.Output(OUT_INV_STD_VAR), variance_mem);
        }
    }
#endif  // ENABLE_TRAINING

    sp.AddPrimitive(lnorm_prim, lnorm_args);

    sp.SetMemory(node.Output(OUT_OUTPUT), src_mem, true);
}

zendnn::memory ZendnnLayerNorm::BuildSLN(ZendnnSubgraphPrimitive &sp,
        ZendnnNode &node, zendnn::engine zendnn_engine) {
    // X += SKIP
    // Get input and skip info
    auto input_md = sp.GetMemory(node.Input(IN_INPUT)).get_desc();
    auto skip_md = sp.GetMemory(node.Input(IN_SKIP)).get_desc();
    auto skip_dims = skip_md.dims();

    // Create primitive input map
    std::unordered_map<int, zendnn::memory> skip_bias_args;

    // Create md for the add op, according to the spec the output type should be
    // the same as the input, so we can support inplace ops
    auto add_skip_dst_md = zendnn::memory::desc(skip_dims,
                           node.Output(OUT_OUTPUT).Type(), zendnn::memory::format_tag::any);
    // Create desc for the op and primitive
    auto add_skip_d = zendnn::binary::desc(zendnn::algorithm::binary_add, input_md,
                                           skip_md, add_skip_dst_md);

    // Create primitive descriptor container
    zendnn::binary::primitive_desc add_skip_pd;
    // Add post op bias
    if (node.Input(IN_SLN_BIAS).Exists()) {
        // X += BIAS
        // Get bias md
        auto bias_md = sp.GetMemory(node.Input(IN_SLN_BIAS)).get_desc();
        auto bias_dims = bias_md.dims();
        // To follow the spec means our bias will always have less dimensions that our input}
        // so we add the extra dimensions, reshape it and let ZenDNN broadcast the value
        while (bias_dims.size() < skip_dims.size()) {
            bias_dims.insert(bias_dims.begin(), 1);
        }
        bias_md = bias_md.reshape(bias_dims);

        zendnn::post_ops bias_add;
        zendnn::primitive_attr binary_attr;
        bias_add.append_binary(zendnn::algorithm::binary_add, bias_md);
        binary_attr.set_post_ops(bias_add);
        // Add post op to scale result
        add_skip_pd = zendnn::binary::primitive_desc(add_skip_d, binary_attr,
                      zendnn_engine);

        // Get bias mem
        auto bias_mem = sp.GetMemoryAndReshape(node.Input(IN_SLN_BIAS), bias_md,
                                               zendnn_engine);
        // Add bias arg
        skip_bias_args.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1, bias_mem});

    }
    else {
        add_skip_pd = zendnn::binary::primitive_desc(add_skip_d, zendnn_engine);
    }

    // Move the memory to the target device
    auto src_mem = sp.GetMemoryAndReshape(node.Input(IN_INPUT),
                                          add_skip_pd.src0_desc(), zendnn_engine);
    auto skip_mem = sp.GetMemoryAndReshape(node.Input(IN_SKIP),
                                           add_skip_pd.src1_desc(), zendnn_engine);

    // Add args
    skip_bias_args.insert({ZENDNN_ARG_SRC_0, src_mem});
    skip_bias_args.insert({ZENDNN_ARG_SRC_1, skip_mem});
    skip_bias_args.insert({ZENDNN_ARG_DST, src_mem});

    // Create and add primitive
    auto add_skip_prim = zendnn::binary(add_skip_pd);
    sp.AddPrimitive(add_skip_prim, skip_bias_args);

    // Return src
    return src_mem;
}

void ZendnnLayerNorm::ValidateDims(ZendnnSubgraphPrimitive &sp,
                                   ZendnnNode &node) {
    // Get input and evaluate
    auto input_dims = sp.GetMemory(node.Input(IN_INPUT)).get_desc().dims();
    auto input_dims_size = input_dims.size();

    // Check the inputs are supported by ZenDNN, this is mandatory since sometimes
    // we can't check the input size in the node capability
    if ((input_dims_size > 5) || (input_dims_size < 2)) {
        ORT_THROW("Input tensor dimensionality is not supported by ZenDNN, got ",
                  input_dims_size);
    }

    // To make this function compliant with all possible layernorm flavors,
    // define gamma and shift input position, depending on the operation
    int gamma_pos, shift_pos;
    if (node.OpType() == "SkipLayerNormalization") {
        // For SkipLayerNorm the spec defines the input as a 3D tensor
        if (input_dims_size != 3) {
            // We support 2D arrays but the expected is 3D
            ORT_THROW("Input tensor is expected to have 3 dimensions, got ",
                      input_dims_size);
        }

        // Get skip and evaluate
        auto skip_dims = sp.GetMemory(node.Input(IN_SKIP)).get_desc().dims();
        if (input_dims != skip_dims) {
            ORT_THROW("Input and skip dimmentions do not match");
        }

        // Check if bias was provided and evaluate
        if (node.Input(IN_SLN_BIAS).Exists()) {
            auto bias_dims = sp.GetMemory(node.Input(IN_SLN_BIAS)).get_desc().dims();
            if (bias_dims.size() != 1) {
                ORT_THROW("Bias is expected to have 1 dimension, got ", bias_dims.size());
            }
            if (bias_dims[0] != input_dims[2]) {
                ORT_THROW("Last dimension of bias and input does not match");
            }
        }

        // Define the input position when using SLN
        gamma_pos = IN_SLN_GAMMA;
        shift_pos = IN_BETA;

        // If the op is LayerNorm
    }
    else {
        // Define the input position when using LN
        gamma_pos = IN_LN_GAMMA;
        shift_pos = IN_LN_BIAS;
    }

    // Get gamma and evaluate
    auto gamma_dims = sp.GetMemory(node.Input(gamma_pos)).get_desc().dims();
    if (gamma_dims.size() != 1) {
        ORT_THROW("Gamma is expected to have 1 dimension, got ", gamma_dims.size());
    }
    if (gamma_dims[0] != input_dims[input_dims_size - 1]) {
        ORT_THROW("Last dimension of gamma and input does not match");
    }

    // Check if shift was provided and evaluate
    if (node.Input(shift_pos).Exists()) {
        auto beta_dims = sp.GetMemory(node.Input(shift_pos)).get_desc().dims();
        if (beta_dims.size() != 1) {
            ORT_THROW("Beta is expected to have 1 dimension, got ", beta_dims.size());
        }
        if (beta_dims[0] != input_dims[input_dims_size - 1]) {
            ORT_THROW("Last dimension of beta and input does not match");
        }
    }
}

float ZendnnLayerNorm::GetEpsilon(ZendnnNode &node) {
    auto attr = node.Attributes().find("epsilon");
    float epsilon = 1e-05f;  // Default value according to ONNX spec
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
        epsilon = attr->second().f();
    }
    return epsilon;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime