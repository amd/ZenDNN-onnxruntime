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

#include "zendnn_pool.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

using tag = zendnn::memory::format_tag;
using dt =  zendnn::memory::data_type;

namespace onnxruntime {
namespace ort_zendnn {

ZendnnPool::ZendnnPool() {}

void ZendnnPool::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                 ZendnnNode &node) {

    /* This flag indicates, the current maxpool node is a dummy node if !=0
    * A dummy node is not part of ORT graph, but created in ZenDNN-EP for optimizations.
    * Maxpool node with appropriate kernel_shape and strides are created accordingly. */
    if (node.GetStridesOpt() != 0) {
        dummy_maxpool_node = true;
    }
    auto zendnn_engine = sp.GetEngine();

    using tag = zendnn::memory::format_tag;
    using dt =  zendnn::memory::data_type;

    zendnn::memory::desc src_md,dst_md;
    bool zendnn_enable_bf16 = false;
    const std::string enable_bf16_env = onnxruntime::GetEnvironmentVar("ZENDNN_ONNXRT_ENABLE_BF16_SUPPORT");
    if (!enable_bf16_env.empty()) {
        zendnn_enable_bf16 = (std::stoi(enable_bf16_env) == 0 ? false : true);
    }

#ifdef ENABLE_TRAINING
    // When using training the memory needs to be in a format known to pool_forward and the
    // pool_backward primitives. Since we don't currently have a way to pass the memory format
// from pool_forward to pool_backward; we are choosing to use Onnxruntime's memory format
// as the common memory format to be used by both forward and the backward primitives.
    auto pool_src_mem = sp.GetMemoryInOrtFormat(node.Input(IN_X), zendnn_engine);
#else
    auto pool_src_mem = sp.GetMemory(node.Input(IN_X));
#endif  // ENABLE_TRAINING
    auto src_dims = pool_src_mem.get_desc().dims();
    if (node.Input(IN_X).Type() == dt::u8 ||
            node.Input(IN_X).Type() == dt::s8) {
        src_md = zendnn::memory::desc({src_dims, node.Input(IN_X).Type(), tag::acdb});
    }
    else if(zendnn_enable_bf16) {
        src_md = zendnn::memory::desc({src_dims}, dt::bf16, sp.GetZendnnFormat(src_dims.size()));
    }
    else {
        src_md = pool_src_mem.get_desc();
    }

#ifdef ENABLE_TRAINING
    auto prop_kind = zendnn::prop_kind::forward;
#else
    auto prop_kind = zendnn::prop_kind::forward_inference;
#endif  // ENABLE_TRAINING

    zendnn::algorithm algo = zendnn::algorithm::pooling_max;
    if (node.OpType() == "AveragePool" || node.OpType() == "GlobalAveragePool") {
        algo = zendnn::algorithm::pooling_avg_exclude_padding;
        if (GetCountIncludePadding(node) != 0) {
            algo = zendnn::algorithm::pooling_avg_include_padding;
        }
    }

    auto kernel_shape = GetKernelShape(src_dims, node);
    PoolShape shape = static_cast<PoolShape>(kernel_shape.size());
    auto strides = GetStrides(node, shape);

    auto dst_mem_dims = InferOutputDims(node, src_dims, kernel_shape, strides);
    if(zendnn_enable_bf16)
        dst_md = zendnn::memory::desc(dst_mem_dims,
                                    dt::bf16,sp.GetZendnnFormat(dst_mem_dims.size()));
    else
        dst_md = zendnn::memory::desc(dst_mem_dims,
                                    node.Input(IN_X).Type(), zendnn::memory::format_tag::any);
    auto padding = InferPadding(node, src_dims, kernel_shape, strides);
    auto padding_left = GetPaddingLeft(padding);
    auto padding_right = GetPaddingRight(padding);



    auto pool_desc = zendnn::pooling_forward::desc(prop_kind, algo,
                     src_md, dst_md,
                     strides, kernel_shape,
                     padding_left, padding_right);

    auto pool_pd = zendnn::pooling_forward::primitive_desc(pool_desc,
                   zendnn_engine);

#ifndef ENABLE_TRAINING
    // If using GPU this will move the memory from the CPU to the GPU.
    pool_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), pool_pd.src_desc(),
                                          zendnn_engine);
#endif
    int out_links = (int)node.Output(OUT_Y).GetConsumersCount();
    zendnn::memory pool_dst_mem;
    PrimitiveMemInfo mem_info;
    mem_info.ref_count = out_links;
    mem_info.mem_desc  = pool_pd.dst_desc();
    mem_info.is_dynamic = (out_links > 0 && sp.UseCPUAllocator()) ? true : false;

    if (mem_info.is_dynamic) {
        pool_dst_mem = zendnn::memory(pool_pd.dst_desc(), zendnn_engine, NULL);
    }
    else {
        pool_dst_mem = zendnn::memory(pool_pd.dst_desc(), zendnn_engine);
    }

    auto pool_op = zendnn::pooling_forward(pool_pd);
#ifdef ENABLE_TRAINING
    auto pool_workspace_mem = zendnn::memory(pool_pd.workspace_desc(),
                              zendnn_engine);

    sp.AddPrimitive(pool_op, {{ZENDNN_ARG_SRC, pool_src_mem},
        {ZENDNN_ARG_WORKSPACE, pool_workspace_mem},
        {ZENDNN_ARG_DST, pool_dst_mem}
    });
#else
    sp.AddPrimitive(pool_op, {{ZENDNN_ARG_SRC, pool_src_mem},
        {ZENDNN_ARG_DST, pool_dst_mem}
    }, mem_info);
#endif  //ENABLE_TRAINING

    if(zendnn_enable_bf16)
    {
        auto dst_dims = pool_dst_mem.get_desc().dims();
        auto mem_to_fp32_pd = zendnn::memory::desc(dst_dims,dt::f32,sp.GetZendnnFormat(dst_dims.size()));
        auto mem_to_fp32 = zendnn::memory(mem_to_fp32_pd, zendnn_engine);
        zendnn::stream s{zendnn_engine};
        zendnn::reorder(pool_dst_mem, mem_to_fp32).execute(s, pool_dst_mem, mem_to_fp32);
        pool_dst_mem = mem_to_fp32;
    }
    sp.SetMemory(node.Output(OUT_Y), pool_dst_mem);
#ifdef ENABLE_TRAINING
    if (node.OutputCount() == 2) {
        sp.SetMemory(node.Output(OUT_INDICES), pool_workspace_mem);
    }
#endif  // ENABLE_TRAINING
}


AutoPadType ZendnnPool::GetAutoPad(ZendnnNode &node) {
    std::string auto_pad;
    auto attr = node.Attributes().find("auto_pad");
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
        auto_pad = attr->second().s();
    }
    return ((auto_pad != "") ? StringToAutoPadType(auto_pad) : AutoPadType::NOTSET);
}

int64_t ZendnnPool::GetCeilMode(ZendnnNode &node) {
    auto attr = node.Attributes().find("ceil_mode");
    if (attr != node.Attributes().end()) {
        return attr->second().i();
    }
    return false;
}

int64_t ZendnnPool::GetCountIncludePadding(ZendnnNode &node) {
    auto attr = node.Attributes().find("count_include_pad");
    if (attr != node.Attributes().end()) {
        return attr->second().i();
    }
    return 0;
}

zendnn::memory::dims ZendnnPool::GetDilations(ZendnnNode &node,
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

zendnn::memory::dims ZendnnPool::GetKernelShape(const zendnn::memory::dims
        &src_dims, ZendnnNode &node) {
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

    kernel_shape = std::vector<int64_t>(src_dims.begin() + 2, src_dims.end());
    return kernel_shape;
}

std::vector<int64_t> ZendnnPool::InferPadding(ZendnnNode &node,
        const zendnn::memory::dims &src_dims, const zendnn::memory::dims &kernel_shape,
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
        for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
            int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) /
                                         strides[dim];
            int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim]
                                 - src_dims[dim + 2];
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
        for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
            int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) /
                                         strides[dim];
            int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim]
                                 - src_dims[dim + 2];
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

std::vector<int64_t> ZendnnPool::GetPadding(ZendnnNode &node, PoolShape shape) {
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

zendnn::memory::dims ZendnnPool::GetPaddingLeft(const std::vector<int64_t>
        padding) {
    return zendnn::memory::dims(padding.begin(),
                                padding.begin() + (padding.size() / 2));
}

zendnn::memory::dims ZendnnPool::GetPaddingRight(const std::vector<int64_t>
        padding) {
    return zendnn::memory::dims(padding.begin() + (padding.size() / 2),
                                padding.end());
}

int64_t ZendnnPool::GetStorageOrder(ZendnnNode &node) {
    auto attr = node.Attributes().find("storage_order");
    if (attr != node.Attributes().end()) {
        return static_cast<int>(attr->second().i());
    }
    return 0;
}

zendnn::memory::dims ZendnnPool::GetStrides(ZendnnNode &node, PoolShape shape) {
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

zendnn::memory::dims ZendnnPool::InferOutputDims(ZendnnNode &node,
        const zendnn::memory::dims &src_dims, const zendnn::memory::dims &kernel_shape,
        const zendnn::memory::dims &strides) {
    ORT_ENFORCE(src_dims.size() >= 2);

    zendnn::memory::dims output_dims;
    output_dims.push_back(src_dims[0]);
    output_dims.push_back(src_dims[1]);
    if (IsGlobalPooling(node)) {
        for (size_t dim = 0; dim < src_dims.size() - 2; ++dim) {
            output_dims.push_back(1);
        }
        return output_dims;
    }

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

// Note ZenDNN does not yet support LpPool or GlobalLpPool even though GlobalLpPool is included here.
bool ZendnnPool::IsGlobalPooling(ZendnnNode &node) const {
    return (node.OpType() == "GlobalAveragePool" ||
            node.OpType() == "GlobalMaxPool" || node.OpType() == "GlobalLpPool");
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
