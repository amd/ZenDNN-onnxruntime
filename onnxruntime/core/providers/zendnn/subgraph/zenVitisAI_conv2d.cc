/*******************************************************************************
* Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "zenVitisAI_conv2d.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"
#include <cassert>
#include <cmath>

#define PRINT_RES

namespace onnxruntime {
namespace ort_zendnn {

ZenVitisAIConv2D::ZenVitisAIConv2D() {}

void ZenVitisAIConv2D::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                       ZendnnNode &node) {
    auto zendnn_engine = sp.GetEngine();

    using tag = zendnn::memory::format_tag;
    using dt =  zendnn::memory::data_type;

    /* Read attributes to take necessary steps for int8 convolutions. */
    bool is_relu =  get_string_attribute(node, "is_relu") == "true" ? true : false;
    bool is_sum =  get_string_attribute(node, "is_sum") == "true" ? true : false;

    auto Tinput = get_string_attribute(node, "Tinput");
    auto Toutput = get_string_attribute(node, "Toutput");
    auto Tfilter = get_string_attribute(node, "Tfilter");
    auto Tbias = get_string_attribute(node, "Tbias");
    auto Tsum = get_string_attribute(node, "Tsum");

    int input_scale = get_int_attribute(node, "in_scale");
    int output_scale = get_int_attribute(node,"out_scale");
    int filter_scale = get_int_attribute(node, "weight_scale");
    int add_out_scale = get_int_attribute(node, "add_out_scale");
    int sum_scale = get_int_attribute(node, "sum_scale");
    /*
    For Conv nodes which are in middle of the graph and have input as fp32 and output as int8,
    instead of input_scale attribute, intermediate_float_scale is used for the calculation of input scale
    example - inception model has such conv node.
    */
    int intermediate_float_scale = get_int_attribute(node,
                                   "intermediate_float_scale");
    float relu_alpha = get_float_attribute(node, "relu_alpha");

    /* Obtain src, weghts and bias memory from ort subgraph. */
    auto src_mem = sp.GetMemory(node.Input(IN_X));
    auto src_dims = src_mem.get_desc().dims();

    auto weights_mem = sp.GetMemory(node.Input(IN_W));
    auto weight_dims_original = weights_mem.get_desc().dims();
    zendnn::memory::dims weight_dims = weight_dims_original;

    zendnn::memory bias_mem;
    zendnn::memory::desc conv_bias_md;
    zendnn::memory::desc bias_md;

    if (node.OpType() != "VitisAIConv2DWithoutBias") {
        bias_mem = sp.GetMemory(node.Input(IN_B));
        bias_md = bias_mem.get_desc();
    }

    size_t depth = 1;
    std::vector<float> scales(depth);
    scales[0] = (float)std::pow(2, -input_scale - filter_scale + output_scale);

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

    auto strides = GetStrides(node, shape);
    auto dilations = GetDilations(node, shape);

    zendnn::memory::desc depthwise_weights_md;
    /* Use GetInferedPads here instead of GetPads.
     * Since this will acount for the `auto_pad` attribute in its return value
     */
    auto padding = GetPads(node);
    if (node.OpType() == "VitisAIDepthwiseConv2D") {
        auto group = GetGroup(node);
        if (group != 1) {
            weight_dims.insert(weight_dims.begin(), group);
            weight_dims[1] = static_cast<int64_t>(weight_dims_original[0] / group);
            zendnn::memory::format_tag format = zendnn::memory::format_tag::any;
            switch (shape) {
            case onnxruntime::ort_zendnn::ZenVitisAIConv2D::SHAPE_UNKNOWN: {
                // use format_tag::any
                break;
            }
            case onnxruntime::ort_zendnn::ZenVitisAIConv2D::SHAPE_1D: {
                format = zendnn::memory::format_tag::goiw;
                break;
            }
            case onnxruntime::ort_zendnn::ZenVitisAIConv2D::SHAPE_2D: {
                format = zendnn::memory::format_tag::goihw;
                break;
            }
            case onnxruntime::ort_zendnn::ZenVitisAIConv2D::SHAPE_3D: {
                format = zendnn::memory::format_tag::goidhw;
                break;
            }
            default:
                // use format_tag::any
                break;
            }
            depthwise_weights_md = zendnn::memory::desc({weight_dims}, node.Input(
                                       IN_W).Type(), format);
        }
    }
    auto padding_left = GetPaddingLeft(padding, shape);
    auto padding_right = GetPaddingRight(padding, shape);

    // Figure out the output shape based on the inputs
    auto dst_mem_dims = InferOutputShape(node, src_dims, weight_dims_original,
                                         kernel_shape, strides, dilations, padding);

    dt src_type = dt::s8;
    if (Tinput == "DT_QINT8" || Tinput == "DT_QUINT8") {
        src_type = node.Type(Tinput);
    }
    else {
        // If the code is here, attr == DT_FLOAT or something. Need input reorder from f32(or whatever) to s8.
        LOGS_DEFAULT(INFO) <<"Need input reorder !!";
    }
    auto conv_src_md = zendnn::memory::desc({src_dims}, src_type, tag::any);
    auto conv_weights_md = zendnn::memory::desc({weight_dims}, dt::s8, tag::any);
    if (node.OpType() != "VitisAIConv2DWithoutBias") {
        conv_bias_md = zendnn::memory::desc({bias_md.dims()}, dt::s32, tag::x);
    }
    //output from conv is s8 which is default now
    dt dst_type = dt::s8;
    //If output from conv is s8 or u8 tensor, read and set from attr accordingly
    if (Toutput == "DT_QINT8" || Toutput == "DT_QUINT8") {
        dst_type = node.Type(Toutput);
    }
    else {
        // If the code is here, attr == DT_FLOAT or something. Need output reorder from u8 to f32(or whatever).
        LOGS_DEFAULT(INFO) <<"Need output reorder !!";
    }
    auto conv_dst_md = zendnn::memory::desc({dst_mem_dims}, dst_type, tag::any);

    /* Prepare convolution desc and conv primitive desc. */

    float relu_scale = 1.0f;
    // Handling Asymetric Scaling here
    // Sum nodes can have different scaling for different inputs
    zendnn::primitive_attr conv_attr;
    zendnn::post_ops post_ops;
    if (is_sum) {
        float s_scale = 1.0f;
        if (sum_scale != output_scale || sum_scale != add_out_scale) {
            scales[0] = scales[0] / (float)std::pow(2, output_scale);
            s_scale = (float)std::pow(2, -sum_scale);
            relu_scale = (float)std::pow(2, add_out_scale);
        }
        post_ops.append_sum(s_scale);
    }

    // @TODO: Clipping to INT range to be implemented
    // If relu_alpha is present, this means that it would be relu6
    // if relu6, use eltwise_bounded_relu with 6 * pow(2, output_scale) as upper
    // bound If not, use eltwise_relu with 0 as negative slope.
    if (is_relu) {
        auto relu_algo = zendnn::algorithm::eltwise_relu;
        if (relu_alpha) {
            relu_algo = zendnn::algorithm::eltwise_bounded_relu;
            relu_alpha = 6.0f * (float)std::pow(2, output_scale);
        }
        post_ops.append_eltwise(relu_scale, relu_algo, relu_alpha, 0.0f);
    }
    conv_attr.set_output_scales(0, scales);
    conv_attr.set_post_ops(post_ops);
    zendnn::convolution_forward::primitive_desc prim_desc;
    if (node.OpType() != "VitisAIConv2DWithoutBias") {
        auto conv_desc = zendnn::convolution_forward::desc(
                             zendnn::prop_kind::forward_inference,
                             zendnn::algorithm::convolution_direct,
                             conv_src_md, conv_weights_md, conv_bias_md,
                             conv_dst_md,
                             strides, dilations,
                             padding_left, padding_right);

        prim_desc = zendnn::convolution_forward::primitive_desc(conv_desc, conv_attr,
                    zendnn_engine);
    }
    else {
        auto conv_desc = zendnn::convolution_forward::desc(
                             zendnn::prop_kind::forward_inference,
                             zendnn::algorithm::convolution_direct,
                             conv_src_md, conv_weights_md,
                             conv_dst_md,
                             strides, dilations,
                             padding_left, padding_right);
        prim_desc = zendnn::convolution_forward::primitive_desc(conv_desc, conv_attr,
                    zendnn_engine);
    }

    /* Reorder ops. Derive src, weights, bias and perform the reorders(if necessary).
     * We are doing a a full conv in INT8. So we have the below memory descriptions.
     */
    zendnn::memory src_int8_mem;
    auto weights_int8_mem = zendnn::memory(prim_desc.weights_desc(), zendnn_engine);

    zendnn::memory bias_int32_mem;
    if (node.OpType() != "VitisAIConv2DWithoutBias") {
        bias_int32_mem = zendnn::memory(prim_desc.bias_desc(), zendnn_engine);
    }
    auto dst_int8_mem = zendnn::memory(prim_desc.dst_desc(), zendnn_engine);

    // Input reorders. If input is f32, reorder with scale to s8
    if (Tinput == "DT_FLOAT") {
        src_int8_mem = zendnn::memory(prim_desc.src_desc(), zendnn_engine);
        zendnn::primitive_attr inp_reorder_attr;
        std::vector<float> si(1);
        si[0] = (float)std::pow(2, input_scale);
        if (intermediate_float_scale >= 0) {
            si[0] = (float)std::pow(2, intermediate_float_scale);
        }
        inp_reorder_attr.set_output_scales(0, si);
        sp.AddPrimitive(zendnn::reorder(src_mem, src_int8_mem, inp_reorder_attr), {
            {ZENDNN_ARG_SRC, src_mem},
            {ZENDNN_ARG_DST, src_int8_mem}
        });
    }
    else if (Tinput == "DT_QUINT8" || Tinput == "DT_QINT8") {
        src_int8_mem = sp.GetMemoryAndReshapeByHandle(node.Input(IN_X),
                       prim_desc.src_desc(),
                       zendnn_engine);
    }
    else {
        LOGS_DEFAULT(ERROR) <<"Unsupported/Unknown input format: Should we reorder ?";
    }

    zendnn::stream strm{zendnn_engine};

    // Weights reorders. No integer weights, do VitisAI methods.
    zendnn::primitive_attr weight_reorder_attr;
    std::vector<float> ws(1);
    ws[0] = (float)std::pow(2, filter_scale);
    weight_reorder_attr.set_output_scales(0, ws);
    if (node.OpType() == "VitisAIDepthwiseConv2D") {
        auto weights_f32_mem = sp.GetMemoryAndReshapeByHandle(node.Input(IN_W),
                               depthwise_weights_md, zendnn_engine);
        zendnn::reorder(weights_f32_mem, weights_int8_mem,
                        weight_reorder_attr).execute(strm, weights_mem, weights_int8_mem);
    }
    else {
        zendnn::reorder(weights_mem, weights_int8_mem,
                        weight_reorder_attr).execute(strm, weights_mem, weights_int8_mem);
    }

    if (node.OpType() != "VitisAIConv2DWithoutBias") {
        // Bias reorders. No integer bias, do VitisAI methods.
        std::vector<float> bs(depth);
        bs[0] = (float)std::pow(2, input_scale + filter_scale);
        zendnn::primitive_attr bias_reorder_attr;
        bias_reorder_attr.set_output_scales(0, bs);
        zendnn::reorder(bias_mem, bias_int32_mem,
                        bias_reorder_attr).execute(strm, bias_mem, bias_int32_mem);
    }

    if (node.OpType() == "VitisAIConv2DWithSum") {
        auto bin_int8_mem = sp.GetMemory(node.Input(IN_BINARY));
        dst_int8_mem = bin_int8_mem;
    }
    // Convolution preparations
    if (node.OpType() != "VitisAIConv2DWithoutBias") {
        sp.AddPrimitive(zendnn::convolution_forward(prim_desc), {
            {ZENDNN_ARG_SRC, src_int8_mem},
            {ZENDNN_ARG_WEIGHTS, weights_int8_mem},
            {ZENDNN_ARG_BIAS, bias_int32_mem},
            {ZENDNN_ARG_DST, dst_int8_mem}
        });
    }
    else {
        sp.AddPrimitive(zendnn::convolution_forward(prim_desc), {
            {ZENDNN_ARG_SRC, src_int8_mem},
            {ZENDNN_ARG_WEIGHTS, weights_int8_mem},
            {ZENDNN_ARG_DST, dst_int8_mem}
        });
    }

    // Output reorder, if necessary; based on the Toutput attribute
    if (Toutput == "DT_FLOAT") {
        auto dst_f32_desc = zendnn::memory::desc({prim_desc.dst_desc().dims(), dt::f32, tag::nchw});
        auto dst_fp32_mem = zendnn::memory(dst_f32_desc, zendnn_engine);
        zendnn::primitive_attr output_reorder_attr;
        std::vector<float> os(1);
        os[0] = (float)std::pow(2, -output_scale);
        output_reorder_attr.set_output_scales(0, os);
        sp.AddPrimitive(zendnn::reorder(dst_int8_mem, dst_fp32_mem,
        output_reorder_attr), {
            {ZENDNN_ARG_SRC, dst_int8_mem},
            {ZENDNN_ARG_DST, dst_fp32_mem}
        });
        sp.SetMemory(node.Output(OUT_Y), dst_fp32_mem);
    }
    else {
        sp.SetMemory(node.Output(OUT_Y), dst_int8_mem);
    }
}

int64_t ZenVitisAIConv2D::GetGroup(ZendnnNode &node) {
    auto attr = node.Attributes().find("group");
    if (attr != node.Attributes().end()) {
        return attr->second().i();
    }
    return 1;
}

zendnn::memory::dims ZenVitisAIConv2D::GetPaddingLeft(const
        std::vector<int64_t> &onnx_padding, ConvShape shape) {
    assert(onnx_padding.size() == shape * 2);
    zendnn::memory::dims padding_left;
    padding_left.assign(onnx_padding.begin(), onnx_padding.begin() + shape);
    return padding_left;
}

zendnn::memory::dims ZenVitisAIConv2D::GetPaddingRight(const
        std::vector<int64_t> &onnx_padding, ConvShape shape) {
    assert(onnx_padding.size() == shape * 2);
    zendnn::memory::dims padding_right;
    padding_right.assign(onnx_padding.begin() + shape, onnx_padding.end());
    return padding_right;
}

AutoPadType ZenVitisAIConv2D::GetAutoPad(ZendnnNode &node) {
    std::string auto_pad;
    auto attr = node.Attributes().find("auto_pad");
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
        auto_pad = attr->second().s();
    }
    return ((auto_pad != "") ? StringToAutoPadType(auto_pad) : AutoPadType::NOTSET);
}

zendnn::memory::dims ZenVitisAIConv2D::GetDilations(ZendnnNode &node,
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

std::vector<int64_t> ZenVitisAIConv2D::GetKernelShape(ZendnnNode &node) {
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

std::vector<int64_t> ZenVitisAIConv2D::GetPads(ZendnnNode &node) {
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

zendnn::memory::dims ZenVitisAIConv2D::GetStrides(ZendnnNode &node,
        ConvShape shape) {
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

zendnn::memory::dims ZenVitisAIConv2D::InferOutputShape(ZendnnNode &node,
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

std::string ZenVitisAIConv2D::get_string_attribute(ZendnnNode &node,
        std::string attribute_string) {
    auto attr = node.Attributes().find(attribute_string);
    std::string atr_str = "";
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
        atr_str = attr->second().s();
    }
    return atr_str;
}

int ZenVitisAIConv2D::get_int_attribute(ZendnnNode &node,
                                        std::string attribute_string) {
    auto attr = node.Attributes().find(attribute_string);
    int attr_int = -1;
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
        attr_int = (int)attr->second().i();
    }
    return attr_int;
}

float ZenVitisAIConv2D::get_float_attribute(ZendnnNode &node,
        std::string attribute_string) {
    auto attr = node.Attributes().find(attribute_string);
    float attr_float = 0.0f;
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
        attr_float = attr->second().f();
    }
    return attr_float;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
