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

#include "zendnn_qpool.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnQPool::ZendnnQPool() {}

void ZendnnQPool::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                  ZendnnNode &node) {

    using tag = zendnn::memory::format_tag;
    using dt =  zendnn::memory::data_type;

    auto zendnn_engine = sp.GetEngine();

    auto src_mem = sp.GetMemory(node.Input(IN_X));
    auto src_mem_dims = src_mem.get_desc().dims();

    auto prop_kind = zendnn::prop_kind::forward_inference;

    zendnn::algorithm algo = zendnn::algorithm::pooling_max;
    if (node.OpType() == "QLinearGlobalAveragePool") {
        algo = zendnn::algorithm::pooling_avg_exclude_padding;
        if (GetCountIncludePadding(node) != 0) {
            algo = zendnn::algorithm::pooling_avg_include_padding;
        }
    }

    auto channels_last_ = GetChannelLast(node);

    auto kernel_shape = GetKernelShape(src_mem_dims, node, channels_last_);
    PoolShape shape = static_cast<PoolShape>(kernel_shape.size());
    auto strides = GetStrides(node, shape);

    auto dst_mem_dims = InferOutputDims(node, src_mem_dims, kernel_shape, strides,
                                        channels_last_);

    auto padding = InferPadding(node, src_mem_dims, kernel_shape, strides);
    auto padding_left = GetPaddingLeft(padding);
    auto padding_right = GetPaddingRight(padding);

    zendnn::memory::desc src_u8_md, dst_u8_md;
    if (channels_last_) {
        src_u8_md = zendnn::memory::desc({src_mem_dims, node.Input(IN_X).Type(), tag::nhwc});
        dst_u8_md = zendnn::memory::desc({dst_mem_dims, node.Input(IN_X).Type(), tag::nhwc});
    }
    else {
        src_u8_md = zendnn::memory::desc({src_mem_dims, node.Input(IN_X).Type(), tag::nchw});
        dst_u8_md = zendnn::memory::desc({dst_mem_dims, node.Input(IN_X).Type(), tag::nchw});
    }

    float *x_scale = (float *)sp.GetMemory(node.Input(
            IN_X_SCALE)).get_data_handle();
    float *y_scale = (float *)sp.GetMemory(node.Input(
            IN_Y_SCALE)).get_data_handle();
    /* Reading like scalars. TODO: read like arrays (if need arises) */

    uint8_t *src_zp = (uint8_t *)sp.GetMemory(node.Input(
                          IN_X_ZERO_POINT)).get_data_handle();
    std::vector<int> src_zero_points(1);
    uint8_t *dst_zp = (uint8_t *)sp.GetMemory(node.Input(
                          IN_Y_ZERO_POINT)).get_data_handle();
    std::vector<int> dst_zero_points(1);

    zendnn::primitive_attr pool_attr;
    zendnn::post_ops pool_post_ops;
    std::vector<float> scales(1);

    zendnn::memory scale_mem;
    bool scale_pool_op = ((*src_zp == (uint8_t)0) &&
                          (*dst_zp == (uint8_t)0))?true:false;

    zendnn::memory src_fp32_mem, dst_fp32_mem;

    if (scale_pool_op) {
        scales[0] = (*x_scale)/(*y_scale);
        auto scales_md = sp.GetMemory(node.Input(IN_X_SCALE)).get_desc();
        Padd(&scales_md, dst_u8_md.dims().size());  //Need to provide broadcasted data
        if (channels_last_) {
            scale_mem = zendnn::memory({scales_md.dims(), dt::f32, tag::nhwc},
                                       zendnn_engine);
        }
        else {
            scale_mem = zendnn::memory({scales_md.dims(), dt::f32, tag::nchw},
                                       zendnn_engine);
        }
        scale_mem.set_data_handle(scales.data());
        pool_post_ops.append_eltwise(1.0, zendnn::algorithm::eltwise_linear, scales[0],
                                     0.0);
        pool_attr.set_post_ops(pool_post_ops);

        auto pool_desc = zendnn::pooling_forward::desc(prop_kind, algo,
                         src_u8_md, dst_u8_md,
                         strides, kernel_shape,
                         padding_left, padding_right);

        auto qpool_pd = zendnn::pooling_forward::primitive_desc(pool_desc, pool_attr,
                        zendnn_engine);
        auto qpool_op = zendnn::pooling_forward(qpool_pd);

        auto src_u8_mem = sp.GetMemoryAndReshape(node.Input(IN_X), qpool_pd.src_desc(),
                          zendnn_engine);
        auto dst_u8_mem = zendnn::memory(qpool_pd.dst_desc(), zendnn_engine);
        sp.AddPrimitive(qpool_op, {{ZENDNN_ARG_SRC, src_u8_mem},
            {ZENDNN_ARG_DST, dst_u8_mem}
        });
        sp.SetMemory(node.Output(OUT_Y), dst_u8_mem);
    }
    else {
        zendnn::primitive_attr src_reorder_attr;
        std::vector<float> src_scale_vec(1);
        src_scale_vec[0] = *x_scale;
        src_reorder_attr.set_output_scales(0, src_scale_vec);
        std::vector<int> src_zp_vec(1);


        if (node.Input(IN_X_ZERO_POINT).Type() == dt::s8) {
            int8_t *src_zp_s8 = (int8_t *)sp.GetMemory(node.Input(
                                    IN_X_ZERO_POINT)).get_data_handle();
            src_zp_vec[0] = *src_zp_s8;
        }
        else if (node.Input(IN_X_ZERO_POINT).Type() == dt::u8)  {
            src_zp_vec[0] = *src_zp;
        }

        src_reorder_attr.set_zero_points(ZENDNN_ARG_SRC, 0, src_zp_vec);

        if (channels_last_) {
            src_fp32_mem = zendnn::memory({{src_mem_dims}, dt::f32, tag::nhwc},
            zendnn_engine);
            dst_fp32_mem = zendnn::memory({dst_mem_dims, dt::f32, tag::nhwc},
                                          zendnn_engine);
        }
        else {
            src_fp32_mem = zendnn::memory({{src_mem_dims}, dt::f32, tag::nchw},
            zendnn_engine);
            dst_fp32_mem = zendnn::memory({dst_mem_dims, dt::f32, tag::nchw},
                                          zendnn_engine);
        }

        sp.AddPrimitive(zendnn::reorder(src_mem, src_fp32_mem, src_reorder_attr), {{ZENDNN_ARG_SRC, src_mem},
            {ZENDNN_ARG_DST, src_fp32_mem}
        });


        auto pool_desc = zendnn::pooling_forward::desc(prop_kind, algo,
                         src_fp32_mem.get_desc(), dst_fp32_mem.get_desc(),
                         strides, kernel_shape,
                         padding_left, padding_right);

        auto qpool_pd = zendnn::pooling_forward::primitive_desc(pool_desc, pool_attr,
                        zendnn_engine);
        auto qpool_op = zendnn::pooling_forward(qpool_pd);
        sp.AddPrimitive(qpool_op, {{ZENDNN_ARG_SRC, src_fp32_mem},
            {ZENDNN_ARG_DST, dst_fp32_mem}
        });

        auto dst_zp_mem = sp.GetMemory(node.Input(IN_Y_ZERO_POINT));
        auto dst_zp_md = dst_zp_mem.get_desc();
        Padd(&dst_zp_md, dst_fp32_mem.get_desc().dims().size());
        zendnn::memory dst_zp_u8_mem;
        if (channels_last_) {
            dst_zp_u8_mem = zendnn::memory({dst_zp_md.dims(), node.Input(IN_Y_ZERO_POINT).Type(), tag::nhwc},
                                           zendnn_engine);
        }
        else {
            dst_zp_u8_mem = zendnn::memory({dst_zp_md.dims(), node.Input(IN_Y_ZERO_POINT).Type(), tag::nchw},
                                           zendnn_engine);
        }
        dst_zp_u8_mem.set_data_handle(dst_zp_mem.get_data_handle());

        auto dst_scale_mem = sp.GetMemory(node.Input(IN_Y_SCALE));
        auto dst_scale_md = dst_scale_mem.get_desc();
        Padd(&dst_scale_md, dst_fp32_mem.get_desc().dims().size());
        zendnn::memory dst_scale_f32_mem;
        if (channels_last_) {
            dst_scale_f32_mem = zendnn::memory({dst_scale_md.dims(), dt::f32, tag::nhwc},
                                               zendnn_engine);
        }
        else {
            dst_scale_f32_mem = zendnn::memory({dst_scale_md.dims(), dt::f32, tag::nchw},
                                               zendnn_engine);
        }
        dst_scale_f32_mem.set_data_handle(dst_scale_mem.get_data_handle());

        zendnn::memory dst_u8_mem;
        if (channels_last_) {
            dst_u8_mem = zendnn::memory({dst_mem_dims, node.Input(IN_Y_ZERO_POINT).Type(), tag::nhwc},
                                        zendnn_engine);
        }
        else {
            dst_u8_mem = zendnn::memory({dst_mem_dims, node.Input(IN_Y_ZERO_POINT).Type(), tag::nchw},
                                        zendnn_engine);
        }
        auto binary_desc = zendnn::binary::desc(zendnn::algorithm::binary_div,
                                                dst_fp32_mem.get_desc(), dst_scale_f32_mem.get_desc(), dst_u8_mem.get_desc());

        zendnn::post_ops binary_ops;
        zendnn::primitive_attr binary_attr;
        binary_ops.append_binary(zendnn::algorithm::binary_add,
                                 dst_zp_u8_mem.get_desc());
        binary_attr.set_post_ops(binary_ops);
        auto binary_pd = zendnn::binary::primitive_desc(binary_desc, binary_attr,
                         zendnn_engine);

        std::unordered_map<int, zendnn::memory> arg_map;
        arg_map.insert({ZENDNN_ARG_SRC_0, dst_fp32_mem});
        arg_map.insert({ZENDNN_ARG_SRC_1, dst_scale_f32_mem});
        arg_map.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1, dst_zp_u8_mem});
        arg_map.insert({ZENDNN_ARG_DST, dst_u8_mem});

        auto binary_ql_prim = zendnn::binary(binary_pd);
        sp.AddPrimitive(binary_ql_prim, arg_map);
        sp.SetMemory(node.Output(OUT_Y), dst_u8_mem);
    }
}

void ZendnnQPool::Padd(zendnn::memory::desc *target_md, size_t pad) {
    // Pads an input to broadcast the op correctly
    auto target_dims = target_md->dims();   // Add back padd
    while (target_dims.size() < pad) {
        target_dims.insert(target_dims.end(), 1);
    }
    *target_md = target_md->reshape(target_dims);
}


AutoPadType ZendnnQPool::GetAutoPad(ZendnnNode &node) {
    std::string auto_pad;
    auto attr = node.Attributes().find("auto_pad");
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
        auto_pad = attr->second().s();
    }
    return ((auto_pad != "") ? StringToAutoPadType(auto_pad) : AutoPadType::NOTSET);
}

int64_t ZendnnQPool::GetCeilMode(ZendnnNode &node) {
    auto attr = node.Attributes().find("ceil_mode");
    if (attr != node.Attributes().end()) {
        return attr->second().i();
    }
    return false;
}

int64_t ZendnnQPool::GetCountIncludePadding(ZendnnNode &node) {
    auto attr = node.Attributes().find("count_include_pad");
    if (attr != node.Attributes().end()) {
        return attr->second().i();
    }
    return 0;
}

zendnn::memory::dims ZendnnQPool::GetDilations(ZendnnNode &node,
        PoolShape shape) {
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

zendnn::memory::dims ZendnnQPool::GetKernelShape(const zendnn::memory::dims
        &src_mem_dims, ZendnnNode &node, size_t channels_last_) {
    /* For the nodes sent by ORT, attributes are filled-up
    * But for a dummy maxpool node created here in ZenDNN-EP,
    * no attributes are present. kernel_shape_ is initialized here.
    */
    if (dummy_maxpool_node) {
        return std::vector<int64_t> {1, 1};
    }
    auto attr = node.Attributes().find("kernel_shape");
    std::vector<int64_t> kernel_shape;
    if (attr != node.Attributes().end()) {
        kernel_shape.reserve(attr->second().ints_size());
        for (int i = 0; i < attr->second().ints_size(); ++i) {
            kernel_shape.push_back(attr->second().ints(i));
        }
        return kernel_shape;
    }
    if (!channels_last_) {
        kernel_shape = std::vector<int64_t>(src_mem_dims.begin() + 2,
                                            src_mem_dims.end());
    }
    else {
        kernel_shape = std::vector<int64_t>(src_mem_dims.begin() + 1,
                                            src_mem_dims.end() - 1);
    }
    return kernel_shape;
}

std::vector<int64_t> ZendnnQPool::InferPadding(ZendnnNode &node,
        const zendnn::memory::dims &src_mem_dims,
        const zendnn::memory::dims &kernel_shape,
        const zendnn::memory::dims &strides) {
    auto auto_pad = GetAutoPad(node);
    PoolShape shape = static_cast<PoolShape>(kernel_shape.size());
    std::vector<int64_t> padding;
    switch (auto_pad) {
    case onnxruntime::AutoPadType::NOTSET: {
        padding = GetPadding(node, shape);
        return padding;
        break;
    }
    case onnxruntime::AutoPadType::VALID: {
        padding.resize(shape * 2, 0);
        return padding;
        break;
    }
    case onnxruntime::AutoPadType::SAME_UPPER: {
        padding.resize(shape * 2, 0);
        for (size_t dim = 0; dim < src_mem_dims.size() - 2; ++dim) {
            int64_t legacy_target_size = (src_mem_dims[dim + 2] + strides[dim] - 1) /
                                         strides[dim];
            int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim]
                                 - src_mem_dims[dim + 2];
            int64_t pad_head = pad_needed / 2;
            int64_t pad_tail = pad_needed - pad_head;
            padding[dim] = pad_head;
            padding[dim + shape] = pad_tail;
        }
        return padding;
        break;
    }
    case onnxruntime::AutoPadType::SAME_LOWER: {
        padding.resize(shape * 2, 0);
        for (size_t dim = 0; dim < src_mem_dims.size() - 2; ++dim) {
            int64_t legacy_target_size = (src_mem_dims[dim + 2] + strides[dim] - 1) /
                                         strides[dim];
            int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim]
                                 - src_mem_dims[dim + 2];
            int64_t pad_head = (pad_needed + 1) / 2;
            int64_t pad_tail = pad_needed - pad_head;
            padding[dim] = pad_head;
            padding[dim + shape] = pad_tail;
        }
        return padding;
        break;
    }
    default:
        ORT_THROW("Unsupported AutoPad Type.");
        break;
    }
}

std::vector<int64_t> ZendnnQPool::GetPadding(ZendnnNode &node,
        PoolShape shape) {
    auto attr = node.Attributes().find("pads");
    std::vector<int64_t> pads;
    if (attr != node.Attributes().end() && !IsGlobalPooling(node)) {
        pads.reserve(attr->second().ints_size());
        for (int i = 0; i < attr->second().ints_size(); ++i) {
            pads.push_back(attr->second().ints(i));
        }
    }
    if (pads.empty()) {
        // 'shape * 2' because we want the pad at the start and end of each dim.
        pads.resize(shape * 2, 0);
    }
    return pads;
}

zendnn::memory::dims ZendnnQPool::GetPaddingLeft(const std::vector<int64_t>
        padding) {
    return zendnn::memory::dims(padding.begin(),
                                padding.begin() + (padding.size() / 2));
}

zendnn::memory::dims ZendnnQPool::GetPaddingRight(const std::vector<int64_t>
        padding) {
    return zendnn::memory::dims(padding.begin() + (padding.size() / 2),
                                padding.end());
}

int64_t ZendnnQPool::GetChannelLast(ZendnnNode &node) {
    auto attr = node.Attributes().find("channels_last");
    if (attr != node.Attributes().end()) {
        return static_cast<int>(attr->second().i());
    }
    return 0;
}

zendnn::memory::dims ZendnnQPool::GetStrides(ZendnnNode &node,
        PoolShape shape) {
    /* For the nodes sent by ORT, attributes are filled-up
    * But for a dummy maxpool node created here in ZenDNN-EP,
    * no attributes are present. kernel_shape_ is initialized here.
    */
    if (dummy_maxpool_node) {
        return std::vector<int64_t> {2, 2};
    }
    auto attr = node.Attributes().find("strides");
    std::vector<int64_t> strides;
    if (attr != node.Attributes().end() && !IsGlobalPooling(node)) {
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

zendnn::memory::dims ZendnnQPool::InferOutputDims(ZendnnNode &node,
        const zendnn::memory::dims &src_mem_dims,
        const zendnn::memory::dims &kernel_shape,
        const zendnn::memory::dims &strides, size_t channel_last) {
    ORT_ENFORCE(src_mem_dims.size() >= 2);

    zendnn::memory::dims output_dims;
    output_dims.push_back(src_mem_dims[0]);
    if (!channel_last) {
        output_dims.push_back(src_mem_dims[1]);
    }
    if (IsGlobalPooling(node)) {
        for (size_t dim = 0; dim < src_mem_dims.size() - 2; ++dim) {
            output_dims.push_back(1);
        }
        if (channel_last) {
            output_dims.push_back(src_mem_dims[3]);
        }
        return output_dims;
    }

    auto auto_pad = GetAutoPad(node);
    switch (auto_pad) {
    case onnxruntime::AutoPadType::NOTSET: {
        PoolShape shape = static_cast<PoolShape>(kernel_shape.size());
        std::vector<int64_t> padding = GetPadding(node, shape);
        for (size_t dim = 0; dim < src_mem_dims.size() - 2; ++dim) {
            output_dims.push_back(static_cast<int64_t>(static_cast<float>
                                  (src_mem_dims[dim + 2] + padding[dim] + padding[dim + shape] -
                                   kernel_shape[dim]) /
                                  strides[dim] + 1));
        }
        return output_dims;
        break;
    }
    case onnxruntime::AutoPadType::VALID: {
        for (size_t dim = 0; dim < src_mem_dims.size() - 2; ++dim) {
            output_dims.push_back((src_mem_dims[dim + 2] - kernel_shape[dim]) / strides[dim]
                                  +
                                  1);
        }
        return output_dims;
        break;
    }
    case onnxruntime::AutoPadType::SAME_UPPER: {
        for (size_t dim = 0; dim < src_mem_dims.size() - 2; ++dim) {
            int64_t legacy_target_size = (src_mem_dims[dim + 2] + strides[dim] - 1) /
                                         strides[dim];
            int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim]
                                 - src_mem_dims[dim + 2];
            int64_t out_size = (src_mem_dims[dim + 2] + pad_needed - kernel_shape[dim]) /
                               strides[dim] + 1;
            output_dims.push_back(out_size);
        }
        return output_dims;
        break;
    }
    case onnxruntime::AutoPadType::SAME_LOWER: {
        for (size_t dim = 0; dim < src_mem_dims.size() - 2; ++dim) {
            int64_t legacy_target_size = (src_mem_dims[dim + 2] + strides[dim] - 1) /
                                         strides[dim];
            int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim]
                                 - src_mem_dims[dim + 2];
            int64_t out_size = (src_mem_dims[dim + 2] + pad_needed - kernel_shape[dim]) /
                               strides[dim] + 1;
            output_dims.push_back(out_size);
        }
        return output_dims;
        break;
    }
    default:
        ORT_THROW("Unsupported AutoPad Type.");
        break;
    }
}

bool ZendnnQPool::IsGlobalPooling(ZendnnNode &node) const {
    return (node.OpType() == "QLinearGlobalAveragePool");
}

}  // namespace ort_zendnn
}  // namespace onnxruntime