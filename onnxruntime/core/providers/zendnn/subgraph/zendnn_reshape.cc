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

#include "zendnn_reshape.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"
#include "core/providers/cpu/tensor/reshape_helper.h"

namespace onnxruntime {
namespace ort_zendnn {
ZendnnReshape::ZendnnReshape() { }

void ZendnnReshape::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                    ZendnnNode &node) {
    auto zendnn_engine = sp.GetEngine();

    using dt = zendnn::memory::data_type;
    bool zendnn_enable_bf16 = false;
    const std::string enable_bf16_env = onnxruntime::GetEnvironmentVar("ZENDNN_ONNXRT_ENABLE_BF16_SUPPORT");
    if (!enable_bf16_env.empty())
        zendnn_enable_bf16 = (std::stoi(enable_bf16_env) == 0 ? false : true);

    // the input shape assumes OrtFormat so we get the memory in OrtFormat.
    auto data_mem = sp.GetMemoryInOrtFormat(node.Input(IN_DATA), zendnn_engine);
    zendnn::memory::dims data_dims = data_mem.get_desc().dims();

    auto shape_mem = sp.GetMemory(node.Input(IN_SHAPE));
    zendnn::memory::dims shape_dims = shape_mem.get_desc().dims();
    int64_t *shape_data = (int64_t *)shape_mem.get_data_handle();

    // Reshape helper will take input data_dims shape and the reshape_shape and replace the -1 and 0s with the calculated
    // Output values. The Reshape helper also does a lot of error checking to make sure the Reshape is possible.
    const auto data_dims_span = gsl::span<const int64_t>(data_dims.data(),
                                data_dims.size());
    TensorShapeVector reshape_shape(shape_data, shape_data + shape_dims[0]);
    ReshapeHelper helper(TensorShape(data_dims_span), reshape_shape,
                         GetAllowZero(node));

    zendnn::memory::dims reshape_shape_dims(reshape_shape.cbegin(),
                                            reshape_shape.cend());
    //the zendnn::memory::desc.reshape(shape) failed on some models so we instead create a new zendnn:memory::desc
    zendnn::memory::desc reshaped_md(reshape_shape_dims,
                                     zendnn_enable_bf16? dt::bf16 : node.Input(IN_DATA).Type(),
                                     sp.GetZendnnFormat(reshape_shape.size()));

    zendnn::memory reshaped_mem = zendnn::memory(reshaped_md, zendnn_engine,
                                  nullptr);
    sp.AddReshape(data_mem, reshaped_mem);

    sp.SetMemory(node.Output(OUT_RESHAPED), reshaped_mem, true);
}

bool ZendnnReshape::GetAllowZero(ZendnnNode &node) {
    auto attr = node.Attributes().find("allowzero");
    int64_t allowzero = 0;  //Default value according to ONNX spec
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
        allowzero = attr->second().i();
    }
    return !(allowzero == 0);
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
