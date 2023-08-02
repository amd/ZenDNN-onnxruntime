/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

// Copyright(C) 2021 Intel Corporation
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

#include <float.h>

#include "zendnn_conv.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"
#include <cassert>

namespace onnxruntime {
namespace ort_zendnn {

ZendnnConv::ZendnnConv() {}

void ZendnnConv:: GetDestMemoryDims(ZendnnSubgraphPrimitive &sp,
                                    ZendnnNode &node, zendnn::memory::dims &dst_dims) {
    auto zendnn_engine = sp.GetEngine();
    auto src_dims = sp.GetMemory(node.Input(IN_X)).get_desc().dims();
    auto weights_dims = sp.GetMemory(node.Input(IN_W)).get_desc().dims();
    auto kernel_shape = GetKernelShape(node);
    ConvShape shape = static_cast<ConvShape>(kernel_shape.size());
    assert(shape != SHAPE_UNKNOWN);
    auto strides = GetStrides(node, shape);
    auto dilations = GetDilations(node, shape);
    // Use GetInferedPads here instead of GetPads since this will acount for the `auto_pad` attribute in its return value
    auto padding = GetInferedPads(node, src_dims, dilations, kernel_shape, strides);
    auto padding_left = GetPaddingLeft(padding, shape);
    auto padding_right = GetPaddingRight(padding, shape);
    // Figure out the output shape based on the inputs
    dst_dims = InferOutputShape(node, src_dims, weights_dims, kernel_shape, strides,
                                dilations, padding);
}

void ZendnnConv::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                 ZendnnNode &node) {

    auto zendnn_engine = sp.GetEngine();

    using tag = zendnn::memory::format_tag;
    using dt = zendnn::memory::data_type;

    bool zendnn_enable_bf16 = false;
    const std::string enable_bf16_env = onnxruntime::GetEnvironmentVar("ZENDNN_ONNXRT_ENABLE_BF16_SUPPORT");
    if (!enable_bf16_env.empty()) {
       zendnn_enable_bf16 = (std::stoi(enable_bf16_env) == 0 ? false : true);
    }

    auto conv_src_mem = sp.GetMemory(node.Input(IN_X));
    auto src_dims = conv_src_mem.get_desc().dims();

    auto conv_weights_mem = sp.GetMemory(node.Input(IN_W));
    auto weight_dims_original = conv_weights_mem.get_desc().dims();
    zendnn::memory::dims weight_dims = weight_dims_original;

    zendnn::memory::desc src_md, weight_md, bias_md;
    if (zendnn_enable_bf16) {
        src_md = zendnn::memory::desc(src_dims, dt::bf16, tag::any);
        weight_md = zendnn::memory::desc(weight_dims, dt::bf16, tag::any);
    } else {
        src_md = conv_src_mem.get_desc();
        src_md.data.format_kind = zendnn_format_kind_t::zendnn_format_kind_any;
        weight_md = conv_weights_mem.get_desc();
        weight_md.data.format_kind = zendnn_format_kind_t::zendnn_format_kind_any;
    }
    bool bias_exists  = false;
    bool min_exists = false;
    bool max_exists  = false;

    auto bin_input_index = -1;
    auto min_input_index = -1;
    auto max_input_index = -1;

    float min = -(FLT_MAX);
    float max = FLT_MAX;
    if (node.OpType() == "ConvAdd" || node.OpType() == "ConvAddRelu") {
        if (node.InputCount() == 4) {
            /* X, W, B, BINARY at 0, 1, 2, 3 positions respectively. */
            bias_exists = true;
            bin_input_index = IN_BINARY; //=3
        }
        else if (node.InputCount() == 3) {
            /* X, W, B(BINARY) at 0, 1, 2. Binary exists, but at IN_B position. */
            bias_exists = false;
            bin_input_index = IN_B; //IN_BINARY-1
        }
    }
    else if (node.OpType() == "ConvClip") {
        if (node.InputCount() == 5) {
            bias_exists  = true;
            min_exists = true;
            max_exists  = true;
            min_input_index = IN_B + 1;
            max_input_index = IN_B + 2;
        }
        else if (node.InputCount() == 4) {
            bias_exists = false;
            min_exists = true;
            max_exists  = true;
            min_input_index = IN_B;
            max_input_index = IN_B + 1;
        }
        else if (node.InputCount() == 3) {
            bias_exists = true;
            min_exists = false;
            max_exists  = false;
        }
    }
    else {
        bias_exists = node.InputCount() == 3;
    }
    zendnn::memory conv_bias_mem;
    if (bias_exists) {
    conv_bias_mem = sp.GetMemory(node.Input(IN_B));
    zendnn::memory::dims bias_dims = conv_bias_mem.get_desc().dims();
    if (zendnn_enable_bf16) {
        bias_md = zendnn::memory::desc(bias_dims, dt::bf16, tag::x);
    } else {
        bias_md = conv_bias_mem.get_desc();
    }
  }

    if (min_exists) {
        auto conv_min_mem = sp.GetMemory(node.Input(min_input_index));
        if (conv_min_mem.get_desc().get_size() != 0) {
            min = *((float *)conv_min_mem.get_data_handle());
        }
    }
    if (max_exists) {
        auto conv_max_mem = sp.GetMemory(node.Input(max_input_index));
        if (conv_max_mem.get_desc().get_size() != 0) {
            max = *((float *)conv_max_mem.get_data_handle());
        }
    }

    /*
    * Get any inputs required for the zendnn::convolution_forward::desc
    * beyond the zendnn:memory::desc:
    *  -dilations
    *  - strides
    *  - padding_left and padding_right
    */
    auto kernel_shape = GetKernelShape(node);
    ConvShape shape = static_cast<ConvShape>(kernel_shape.size());
    assert(shape != SHAPE_UNKNOWN);

    auto group = GetGroup(node);
    if (group != 1) {
        weight_dims.insert(weight_dims.begin(), group);
        weight_dims[1] = static_cast<int64_t>(weight_dims_original[0] / group);
        zendnn::memory::format_tag format = zendnn::memory::format_tag::any;
        switch (shape) {
        case onnxruntime::ort_zendnn::ZendnnConv::SHAPE_UNKNOWN: {
            // use format_tag::any
            break;
        }
        case onnxruntime::ort_zendnn::ZendnnConv::SHAPE_1D: {
            format = zendnn::memory::format_tag::goiw;
            break;
        }
        case onnxruntime::ort_zendnn::ZendnnConv::SHAPE_3D: {
            format = zendnn::memory::format_tag::goidhw;
            break;
        }
        default:
            // use format_tag::any
            break;
        }
        if (zendnn_enable_bf16) {
            weight_md = zendnn::memory::desc({weight_dims}, dt::bf16, format);
        } else {
            weight_md = zendnn::memory::desc({weight_dims}, node.Input(IN_W).Type(), format);
        }
    }

    auto strides = GetStrides(node, shape);
    auto dilations = GetDilations(node, shape);
    // Use GetInferedPads here instead of GetPads since this will acount for the `auto_pad` attribute in its return value
    auto padding = GetInferedPads(node, src_dims, dilations, kernel_shape, strides);
    auto padding_left = GetPaddingLeft(padding, shape);
    auto padding_right = GetPaddingRight(padding, shape);

    // Figure out the output shape based on the inputs
    auto dst_mem_dims = InferOutputShape(node, src_dims, weight_dims_original,
                                         kernel_shape, strides, dilations, padding);
    zendnn::memory::desc dst_md = zendnn::memory::desc({dst_mem_dims}, node.Input(
                                      IN_X).Type(), zendnn::memory::format_tag::any);

    if (node.isInplaceMemoryNode) {
        dst_md = _ldst_md;
    }

#ifdef ENABLE_TRAINING
    auto prop_kind = zendnn::prop_kind::forward_training;
#else
    auto prop_kind = zendnn::prop_kind::forward_inference;
#endif  // ENABLE_TRAINING

    zendnn::primitive_attr attr;
    if (node.OpType() == "ConvRelu") {
        const float ops_scale = 1.f;
        const float ops_alpha = GetReluAlpha(node);
        const float ops_beta = 0.f;
        zendnn::post_ops ops;
        ops.append_eltwise(ops_scale, zendnn::algorithm::eltwise_relu, ops_alpha,
                           ops_beta);
        attr.set_post_ops(ops);
    }
    else if (node.OpType() == "ConvClip") {
        const float ops_scale = 1.f;
        if (!min_exists) {
            min = GetMin(node, /*default_alpha*/-(FLT_MAX));
        }
        if (!max_exists) {
            max = GetMax(node, /*default_alpha*/FLT_MAX);
        };
        zendnn::post_ops ops;
        ops.append_eltwise(ops_scale, zendnn::algorithm::eltwise_clip, min, max);
        attr.set_post_ops(ops);
    }
    else if (node.OpType() == "ConvElu") {
        const float ops_scale = 1.f;
        const float ops_alpha = GetAlpha(node);
        const float ops_beta = 0.f;
        zendnn::post_ops ops;
        ops.append_eltwise(ops_scale, zendnn::algorithm::eltwise_elu, ops_alpha,
                           ops_beta);
        attr.set_post_ops(ops);
    }
    else if (node.OpType() == "ConvSwish") {
        const float ops_scale = 1.f;
        const float ops_alpha = GetAlpha(node);
        const float ops_beta = 0.f;
        zendnn::post_ops ops;
        ops.append_eltwise(ops_scale, zendnn::algorithm::eltwise_swish, ops_alpha,
                           ops_beta);
        attr.set_post_ops(ops);
    }
    else if (node.OpType() == "ConvAddRelu") {
        zendnn::post_ops ops;
        const float ops_scale = 1.f;
        const float ops_alpha = 0.f;
        const float ops_beta = 0.f;
        ops.append_sum(ops_scale);
        ops.append_eltwise(ops_scale, zendnn::algorithm::eltwise_relu, ops_alpha,
                           ops_beta);
        attr.set_post_ops(ops);
    }
    else if (node.OpType() == "ConvAdd") {
        zendnn::post_ops ops;
        const float ops_scale = 1.f;
        ops.append_sum(ops_scale);
        attr.set_post_ops(ops);
    }

    zendnn::convolution_forward::primitive_desc conv_pd;
    if (bias_exists) {
        auto conv_desc = zendnn::convolution_forward::desc(
                             prop_kind, zendnn::algorithm::convolution_direct,
                             src_md, weight_md, bias_md, dst_md,
                             strides, dilations, padding_left, padding_right);
        conv_pd = zendnn::convolution_forward::primitive_desc(conv_desc, attr,
                  zendnn_engine);
    }
    else {
        auto conv_desc = zendnn::convolution_forward::desc(
                             prop_kind, zendnn::algorithm::convolution_direct,
                             src_md, weight_md, dst_md,
                             strides, dilations, padding_left, padding_right);
        conv_pd = zendnn::convolution_forward::primitive_desc(conv_desc, attr,
                  zendnn_engine);
    }

    // If using GPU this will move the memory from the CPU to the GPU.
    conv_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), conv_pd.src_desc(),
                                          zendnn_engine);
    if (group != 1 && shape == onnxruntime::ort_zendnn::ZendnnConv::SHAPE_2D) {
        weight_md = zendnn::memory::desc({weight_dims}, node.Input(IN_W).Type(),
                                         zendnn::memory::format_tag::goihw);
        auto mem_from = zendnn::memory(weight_md, zendnn_engine,
                                       sp.GetMemory(node.Input(IN_W)).get_data_handle());
        auto mem_to = zendnn::memory(conv_pd.weights_desc(), zendnn_engine);
        zendnn::stream s{zendnn_engine};
        zendnn::reorder(mem_from, mem_to).execute(s, mem_from, mem_to);
        s.wait();
        conv_weights_mem = mem_to;
    }
    else {
        conv_weights_mem = sp.GetMemoryAndReshape(node.Input(IN_W),
                           conv_pd.weights_desc(), zendnn_engine);
    }
    if (bias_exists) {
        conv_bias_mem = sp.GetMemoryAndReshape(node.Input(IN_B), conv_pd.bias_desc(),
                                               zendnn_engine);
    }
    int out_links = (int)node.Output(OUT_Y).GetConsumersCount();
    if ((node.isInplaceMemoryNode && concat_order !=0) ||
            node.OpType() == "ConvAdd" || node.OpType() == "ConvAddRelu") {
        out_links = 0;
    }
    if (node.isInplaceMemoryNode && concat_order ==0) {
        out_links = ref_count;
    }
    PrimitiveMemInfo mem_info;
    mem_info.ref_count = out_links;
    mem_info.is_dynamic = (out_links > 0 && sp.UseCPUAllocator()) ? true : false;
    if (node.isInplaceMemoryNode && concat_order ==0) {
        mem_info.mem_desc  = _dst_mem.get_desc();
    }
    else {
        mem_info.mem_desc  = conv_pd.dst_desc();
    }

    zendnn::memory conv_dst_mem;

    // Add the convolution layer to the subgraph
    auto conv_op = zendnn::convolution_forward(conv_pd);

    if (node.OpType() == "ConvAdd" || node.OpType() == "ConvAddRelu") {
        /*
        Conv_1_Y.mem = ops_scale*(Conv_1_Y.mem + Conv2(X, W, B).mem)
        conv_dst_mem = binary_post_op_mem(viz. Conv_1_Y.mem)
        */
        if(zendnn_enable_bf16) {
            auto binary_post_op_mem = sp.GetMemoryAndReshape(node.Input(bin_input_index),conv_pd.dst_desc(),zendnn_engine);
            conv_dst_mem = binary_post_op_mem;
        }
        else {
            auto binary_post_op_mem = sp.GetMemory(node.Input(bin_input_index).Name());
            conv_dst_mem=binary_post_op_mem;
            sp.IncMemoryRefCount(conv_dst_mem);
        }
    }
    else {
        if (node.isInplaceMemoryNode) {
            conv_dst_mem = _dst_mem;
        }
        else if (mem_info.is_dynamic) {
            conv_dst_mem = zendnn::memory(conv_pd.dst_desc(), zendnn_engine, NULL);
        }
        else {
            conv_dst_mem = zendnn::memory(conv_pd.dst_desc(), zendnn_engine);
        }
    }

    if (bias_exists) {
        sp.AddPrimitive(conv_op, {{ZENDNN_ARG_SRC, conv_src_mem},
            {ZENDNN_ARG_WEIGHTS, conv_weights_mem},
            {ZENDNN_ARG_BIAS, conv_bias_mem},
            {ZENDNN_ARG_DST, conv_dst_mem}
        }, mem_info);
    }
    else {
        sp.AddPrimitive(conv_op, {{ZENDNN_ARG_SRC, conv_src_mem},
            {ZENDNN_ARG_WEIGHTS, conv_weights_mem},
            {ZENDNN_ARG_DST, conv_dst_mem}
        }, mem_info);
    }

    if (add_output) {
        sp.SetMemory(node.Output(OUT_Y), conv_dst_mem);
    }
}

void ZendnnConv::SetDestinationMemoryInfo(zendnn::memory::desc &ldst_md,
        zendnn::memory &dst_mem, int concatOrder, int refCount, bool addOutput) {
    _ldst_md = ldst_md;
    _dst_mem = dst_mem;
    add_output = addOutput;
    concat_order= concatOrder;
    ref_count = refCount;
}

std::vector<int64_t> ZendnnConv::GetInferedPads(ZendnnNode &node,
        const zendnn::memory::dims &src_dims,
        const zendnn::memory::dims &dilations,
        const std::vector<int64_t> &kernel_shape,
        const zendnn::memory::dims &strides) {
    AutoPadType auto_pad = GetAutoPad(node);
    ConvShape shape = static_cast<ConvShape>(kernel_shape.size());
    std::vector<int64_t> pads;
    if (auto_pad == AutoPadType::NOTSET) {
        pads = GetPads(node);
        if (pads.empty()) {
            // 'shape * 2' because we want the pad at the start and end of each dim.
            pads.resize(shape * 2, 0);
        }
        return pads;
    }

    pads.resize(shape * 2, 0);

    int64_t pad_head = 0;
    int64_t pad_tail = 0;
    assert(src_dims.size() == shape + 2);
    for (size_t i = 0; i < shape; ++i) {
        if (ComputePad(src_dims[2 + i], strides[i], kernel_shape[i], (dilations[i] + 1),
                       auto_pad, pad_head, pad_tail)) {
            pads[i] = pad_head;
            pads[shape + i] = pad_tail;
        }
    }
    return pads;
}

zendnn::memory::dims ZendnnConv::GetPaddingLeft(const std::vector<int64_t>
        &onnx_padding, ConvShape shape) {
    assert(onnx_padding.size() == shape * 2);
    zendnn::memory::dims padding_left;
    padding_left.assign(onnx_padding.begin(), onnx_padding.begin() + shape);
    return padding_left;
}

zendnn::memory::dims ZendnnConv::GetPaddingRight(const std::vector<int64_t>
        &onnx_padding, ConvShape shape) {
    assert(onnx_padding.size() == shape * 2);
    zendnn::memory::dims padding_right;
    padding_right.assign(onnx_padding.begin() + shape, onnx_padding.end());
    return padding_right;
}

AutoPadType ZendnnConv::GetAutoPad(ZendnnNode &node) {
    std::string auto_pad;
    auto attr = node.Attributes().find("auto_pad");
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
        auto_pad = attr->second().s();
    }
    return ((auto_pad != "") ? StringToAutoPadType(auto_pad) : AutoPadType::NOTSET);
}

zendnn::memory::dims ZendnnConv::GetDilations(ZendnnNode &node,
        ConvShape shape) {
    auto attr = node.Attributes().find("dilations");
    std::vector<int64_t> dilations;
    if (attr != node.Attributes().end()) {
        dilations.reserve(attr->second().ints_size());
        for (int i = 0; i < attr->second().ints_size(); ++i) {
            // ZenDNN dilations are always one less than Onnx dilations
            dilations.push_back(attr->second().ints(i) - 1);
        }
    }
    else {
        dilations.resize(shape, 0);
    }
    return zendnn::memory::dims(dilations.begin(), dilations.end());
}
int64_t ZendnnConv::GetGroup(ZendnnNode &node) {
    auto attr = node.Attributes().find("group");
    if (attr != node.Attributes().end()) {
        return attr->second().i();
    }
    return 1;
}

float ZendnnConv::GetAlpha(ZendnnNode &node) {
    auto attr = node.Attributes().find("alpha");
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return 1.0f;
}

float ZendnnConv::GetMin(ZendnnNode &node, float default_min) {
    auto attr = node.Attributes().find("min");
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return default_min;
}

float ZendnnConv::GetMax(ZendnnNode &node, float default_max) {
    auto attr = node.Attributes().find("max");
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return default_max;
}

std::vector<int64_t> ZendnnConv::GetKernelShape(ZendnnNode &node) {
    auto attr = node.Attributes().find("kernel_shape");
    std::vector<int64_t> kernel_shape;
    if (attr != node.Attributes().end()) {
        kernel_shape.reserve(attr->second().ints_size());
        for (int i = 0; i < attr->second().ints_size(); ++i) {
            kernel_shape.push_back(attr->second().ints(i));
        }
        return kernel_shape;
    }
    // Infer the Kernel shape from the input weights
    auto weight_dims = node.Input(IN_W).Dim();
    kernel_shape = std::vector<int64_t>(weight_dims.begin() + 2, weight_dims.end());
    return kernel_shape;
}

std::vector<int64_t> ZendnnConv::GetPads(ZendnnNode &node) {
    auto attr = node.Attributes().find("pads");
    if (attr != node.Attributes().end()) {
        std::vector<int64_t> pads;
        pads.reserve(attr->second().ints_size());
        for (int i = 0; i < attr->second().ints_size(); ++i) {
            pads.push_back(attr->second().ints(i));
        }
        return pads;
    }
    return {};
}

zendnn::memory::dims ZendnnConv::GetStrides(ZendnnNode &node, ConvShape shape) {
    if (node.GetStridesOpt() == 1) {
        return std::vector<int64_t> {1, 1};
    }
    else if (node.GetStridesOpt() == 2) {
        return std::vector<int64_t> {2, 2};
    }
    auto attr = node.Attributes().find("strides");
    std::vector<int64_t> strides;
    if (attr != node.Attributes().end()) {
        strides.reserve(attr->second().ints_size());
        for (int i = 0; i < attr->second().ints_size(); ++i) {
            strides.push_back(attr->second().ints(i));
        }
    }
    else {
        strides.resize(shape, 1);
    }
    return zendnn::memory::dims(strides.begin(), strides.end());
}

// ComputePad is copy/paste of a the ComputePad found in core/providers/common.h
// With some minor modifications.
// ComputePad is not exposed to the shared library so this copy is used instead.
bool ZendnnConv::ComputePad(const int64_t in_dim,
                            const int64_t stride,
                            const int64_t kernel,
                            const int64_t dilation,
                            AutoPadType pad_type,
                            int64_t &pad_head, /* output param */
                            int64_t &pad_tail, /* output param */
                            bool force_symmetric_auto_padding /*= false*/) {
    pad_head = 0;
    pad_tail = 0;
    switch (pad_type) {
    case AutoPadType::NOTSET:
        break;
    case AutoPadType::VALID:
        break;
    case AutoPadType::SAME_UPPER:
    //[[fallthrough]] //fallthrough attribute requires C++17
    case AutoPadType::SAME_LOWER: {
        if (1 != dilation) {
            LOGS_DEFAULT(ERROR) <<
                                "Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.";
            return false;
        }

        // The ONNX spec says if `auto_pad` attribute is set, pad until the `legacy_target_size`
        // is `ceil (in_dim / stride)`. The following line of code is essentially just that and
        // is retained as is
        int64_t legacy_target_size = (in_dim + stride - 1) / stride;
        int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
        // make sure padding is symmetric
        if (force_symmetric_auto_padding) {
            // Inlining math::roundUpPow2() from util/math.h to avoid bringing in the transitive dependencies.
            pad_needed = (pad_needed + 1) & ~1;
        }

        if (pad_type == AutoPadType::SAME_LOWER) {
            pad_head = (pad_needed + 1) / 2;
        }
        else {
            pad_head = pad_needed / 2;
        }
        pad_tail = pad_needed - pad_head;
    }
    break;
    default:
        LOGS_DEFAULT(ERROR) << "ComputePad: pad_type attribute not supported.";
        return false;
    }
    return true;
}

float ZendnnConv::GetReluAlpha(ZendnnNode &node) {
    auto attr = node.Attributes().find("alpha");
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return 0.0f;
}


zendnn::memory::dims ZendnnConv::InferOutputShape(ZendnnNode &node,
        const zendnn::memory::dims &src_dims,
        const zendnn::memory::dims &weight_dims,
        const std::vector<int64_t> &kernel_shape,
        const zendnn::memory::dims &strides,
        const zendnn::memory::dims &dilations,
        const std::vector<int64_t> &pads) {
    auto pad_type = GetAutoPad(node);
    ConvShape shape = static_cast<ConvShape>(kernel_shape.size());
    zendnn::memory::dims output_shape;

    output_shape.push_back(src_dims[0]);
    output_shape.push_back(weight_dims[0]);
    for (size_t dim = 0; dim < shape; ++dim) {
        if (dim >= strides.size() || dim >= kernel_shape.size() ||
                dim >= dilations.size() || dim >= pads.size() ||
                shape + dim >= pads.size()) {
            LOGS_DEFAULT(ERROR) << "Out of bound access to array";
            return {};
        }
        int64_t dkernel = (dilations[dim] + 1) * (kernel_shape[dim] - 1) + 1;
        switch (pad_type) {
        case onnxruntime::AutoPadType::NOTSET: {
            output_shape.push_back(static_cast<int64_t>(static_cast<float>
                                   (src_dims[dim + 2] + pads[dim] + pads[dim + shape] - dkernel) / strides[dim] +
                                   1));
        }
        break;
        case onnxruntime::AutoPadType::VALID: {
            output_shape.push_back((src_dims[dim + 2] - dkernel) / strides[dim] + 1);
        }
        break;
        case onnxruntime::AutoPadType::SAME_UPPER: {
            if (dilations[dim] != 0) {
                LOGS_DEFAULT(ERROR) <<
                                    "Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.";
                return {};
            }
            int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) /
                                         strides[dim];
            int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim]
                                 - src_dims[dim + 2];
            output_shape.push_back((src_dims[dim + 2] + pad_needed - dkernel) / strides[dim]
                                   + 1);
        }
        break;
        case onnxruntime::AutoPadType::SAME_LOWER: {
            int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) /
                                         strides[dim];
            int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim]
                                 - src_dims[dim + 2];
            output_shape.push_back((src_dims[dim + 2] + pad_needed - dkernel) / strides[dim]
                                   + 1);
        }
        break;
        default:
            break;
        }
    }
    return output_shape;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
