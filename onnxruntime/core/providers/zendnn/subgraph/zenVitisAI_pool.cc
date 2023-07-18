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

#include "zenVitisAI_pool.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

ZenVitisAIPool::ZenVitisAIPool() {}

void ZenVitisAIPool::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                     ZendnnNode &node) {
    auto zendnn_engine = sp.GetEngine();

    using tag = zendnn::memory::format_tag;
    using dt = zendnn::memory::data_type;

    auto Tinput = get_string_attribute(node, "Tinput");
    auto Toutput = get_string_attribute(node, "Toutput");

    auto src_mem = sp.GetMemory(node.Input(IN_X));
    auto src_mem_dim = src_mem.get_desc().dims();

    auto prop_kind = zendnn::prop_kind::forward_inference;

    zendnn::algorithm algo = zendnn::algorithm::pooling_max;
    if (node.OpType() == "VitisAIAvgPool") {
        algo = zendnn::algorithm::pooling_avg_exclude_padding;
    }
    auto kernel_shape = GetKernelShape(src_mem_dim, node);
    PoolShape shape = static_cast<PoolShape>(kernel_shape.size());
    auto strides = GetStrides(node, shape);

    auto dst_mem_dims = InferOutputDims(node, src_mem_dim, kernel_shape, strides);

    auto padding = GetPadding(node, shape);
    auto padding_left = GetPaddingLeft(padding);
    auto padding_right = GetPaddingRight(padding);

    /* A limitation in current ZenDNN lib.
     * src mem desc as type=u8/s8 and format_tag=nchw/abcd is not working. Primitive creation failure.
     * Therefore, we insert a reorder by forcing a working type+tag combination (u8 and acdb)
     */

    zendnn::memory::desc pool_src_md({src_mem_dim}, node.Input(IN_X).Type(),
                                     tag::acdb);

    dt dst_type = dt::u8;
    if (Toutput == "DT_QINT8" || Toutput == "DT_QUINT8") {
        dst_type = node.Type(Toutput);
    }
    auto pool_dst_md = zendnn::memory::desc({dst_mem_dims}, dst_type, tag::any);

    auto pool_desc = zendnn::pooling_forward::desc(prop_kind, algo,
                     pool_src_md, pool_dst_md,
                     strides, kernel_shape,
                     padding_left, padding_right);

    auto pool_pd = zendnn::pooling_forward::primitive_desc(pool_desc,
                   zendnn_engine);

    //Input is in int8 so no reorder needed.
    auto dst_int8_mem = zendnn::memory(pool_pd.dst_desc(), zendnn_engine);

    // Reorder added
    src_mem = sp.GetMemoryAndReshapeByHandle(node.Input(IN_X), pool_pd.src_desc(),
              zendnn_engine);
    sp.AddPrimitive(zendnn::pooling_forward(pool_pd), {{ZENDNN_ARG_SRC, src_mem},
        {ZENDNN_ARG_DST, dst_int8_mem}
    });

    if (Toutput == "DT_FLOAT") {
        auto dst_f32_desc = zendnn::memory::desc({pool_pd.dst_desc().dims(), dt::f32, tag::acdb});
        auto dst_fp32_mem = zendnn::memory(dst_f32_desc, zendnn_engine);

        sp.AddPrimitive(zendnn::reorder(dst_int8_mem, dst_fp32_mem), {{ZENDNN_ARG_SRC, dst_int8_mem},
            {ZENDNN_ARG_DST, dst_fp32_mem}
        });
        sp.SetMemory(node.Output(OUT_Y), dst_fp32_mem);
    }
    else {
        sp.SetMemory(node.Output(OUT_Y), dst_int8_mem);
    }
}

zendnn::memory::dims ZenVitisAIPool::GetKernelShape(const
        zendnn::memory::dims &src_dims, ZendnnNode &node) {
    auto attr = node.Attributes().find("kernel_shape");
    std::vector<int64_t> kernel_shape;
    if (attr != node.Attributes().end()) {
        kernel_shape.reserve(attr->second().ints_size());
        for (int i = 0; i < attr->second().ints_size(); ++i) {
            kernel_shape.push_back(attr->second().ints(i));
        }
        return kernel_shape;
    }

    kernel_shape = std::vector<int64_t>(src_dims.begin() + 2, src_dims.end());
    return kernel_shape;
}

std::string ZenVitisAIPool::get_string_attribute(ZendnnNode &node,
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

zendnn::memory::dims ZenVitisAIPool::GetPaddingLeft(const
        std::vector<int64_t> padding) {
    return zendnn::memory::dims(padding.begin(),
                                padding.begin() + (padding.size() / 2));
}

zendnn::memory::dims ZenVitisAIPool::GetPaddingRight(
    const std::vector<int64_t> padding) {
    return zendnn::memory::dims(padding.begin() + (padding.size() / 2),
                                padding.end());
}

AutoPadType ZenVitisAIPool::GetAutoPad(ZendnnNode &node) {
    std::string auto_pad;
    auto attr = node.Attributes().find("auto_pad");
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
        auto_pad = attr->second().s();
    }
    return ((auto_pad != "") ? StringToAutoPadType(auto_pad) : AutoPadType::NOTSET);
}

std::vector<int64_t> ZenVitisAIPool::GetPadding(ZendnnNode &node,
        PoolShape shape) {
    auto attr = node.Attributes().find("pads");
    std::vector<int64_t> pads;
    if (attr != node.Attributes().end()) {
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

zendnn::memory::dims ZenVitisAIPool::GetStrides(ZendnnNode &node,
        PoolShape shape) {
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

zendnn::memory::dims ZenVitisAIPool::InferOutputDims(ZendnnNode &node,
        const zendnn::memory::dims &src_dims, const zendnn::memory::dims &kernel_shape,
        const zendnn::memory::dims &strides) {
    ORT_ENFORCE(src_dims.size() >= 2);

    zendnn::memory::dims output_dims;
    output_dims.push_back(src_dims[0]);
    output_dims.push_back(src_dims[1]);

    auto auto_pad = GetAutoPad(node);
    switch (auto_pad) {
    case onnxruntime::AutoPadType::NOTSET: {
        PoolShape shape = static_cast<PoolShape>(kernel_shape.size());
        std::vector<int64_t> padding = GetPadding(node, shape);
        for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
            output_dims.push_back(static_cast<int64_t>(static_cast<float>
                                  (src_dims[dim + 2] + padding[dim] + padding[dim + shape] - kernel_shape[dim]) /
                                  strides[dim] + 1));
        }
        return output_dims;
        break;
    }
    case onnxruntime::AutoPadType::VALID: {
        for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
            output_dims.push_back((src_dims[dim + 2] - kernel_shape[dim]) / strides[dim] +
                                  1);
        }
        return output_dims;
        break;
    }
    case onnxruntime::AutoPadType::SAME_UPPER: {
        for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
            int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) /
                                         strides[dim];
            int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim]
                                 - src_dims[dim + 2];
            int64_t out_size = (src_dims[dim + 2] + pad_needed - kernel_shape[dim]) /
                               strides[dim] + 1;
            output_dims.push_back(out_size);
        }
        return output_dims;
        break;
    }
    case onnxruntime::AutoPadType::SAME_LOWER: {
        for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
            int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) /
                                         strides[dim];
            int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim]
                                 - src_dims[dim + 2];
            int64_t out_size = (src_dims[dim + 2] + pad_needed - kernel_shape[dim]) /
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

}  // namespace ort_zendnn
}  // namespace onnxruntime
