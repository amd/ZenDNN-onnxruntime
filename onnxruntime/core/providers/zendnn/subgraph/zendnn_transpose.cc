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

#include "zendnn_transpose.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

#include <iostream>

namespace onnxruntime {
namespace ort_zendnn {

ZendnnTranspose::ZendnnTranspose() {}

/*
Transpose:
  Inputs:
    0) DATA - Input Tensor
  Outputs:
    0) TRANSPOSED - Output Tensor

    (DATA)     +-----------+ (TRANSPOSED)
    ---------->+ Transpose +-------------->
               +-----------+

Attributes (perm) - A list of integers. By default, reverse the dimensions,
                    otherwise permute the axes according to the values given.
*/
void ZendnnTranspose::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                      ZendnnNode &node) {

    auto zendnn_engine = sp.GetEngine();

    using dt = zendnn::memory::data_type;
    bool zendnn_enable_bf16 = false;
    const std::string enable_bf16_env = onnxruntime::GetEnvironmentVar("ZENDNN_ONNXRT_ENABLE_BF16_SUPPORT");
    if (!enable_bf16_env.empty())
        zendnn_enable_bf16 = (std::stoi(enable_bf16_env) == 0 ? false : true);

    auto data_mem = sp.GetMemory(node.Input(IN_DATA));
    auto data_dims = data_mem.get_desc().dims();
    auto ndata_dims = data_dims.size();

    auto perm = GetPerm(node);
    if (perm.size() == 0) {
        perm.reserve(ndata_dims);
        for (size_t i = 0; i < ndata_dims; ++i) {
            perm.push_back(static_cast<int64_t>(ndata_dims - i - 1));
        }
    }

    zendnn::memory::dims transposed_dims(ndata_dims, 0);
    zendnn::memory::dims strides(ndata_dims, 0);
    zendnn::memory::dim total_stride = 1;
    for (int i = (int)ndata_dims - 1 ; i >= 0; i--) {
        transposed_dims[i] = data_dims[perm[i]];
        strides[perm[i]] = total_stride;
        total_stride *= data_dims[perm[i]];
    }

    zendnn::memory::dims strides_inverse;
    strides_inverse.reserve(ndata_dims);
    for (size_t i = 0; i < ndata_dims; ++i) {
        strides_inverse.push_back(strides[ndata_dims - i - 1]);
    }

    // Memory descriptor describes the memory reorder but will not have the correct output dimentions or the correct zendnn::memory::format
    zendnn::memory::desc intermediate_md = zendnn::memory::desc(data_dims,
                                           zendnn_enable_bf16 ? dt::bf16 : node.Input(IN_DATA).Type(), strides);
    zendnn::memory intermediate_mem = zendnn::memory(intermediate_md,
                                      zendnn_engine);

    auto traspose_primitive = zendnn::reorder(data_mem, intermediate_mem);
    sp.AddPrimitive(traspose_primitive, {{ZENDNN_ARG_FROM, data_mem},
        {ZENDNN_ARG_TO, intermediate_mem}
    });

    // The reorder from above will get the memory in the right order. The next few lines will create a memory and memory descriptor
    // that will have the correct dimentions and correct memory::format
    zendnn::memory::desc transposed_md = zendnn::memory::desc(transposed_dims,
                                         zendnn_enable_bf16 ? dt::bf16 : node.Input(IN_DATA).Type(),
                                         sp.GetZendnnFormat(data_dims.size()));
    zendnn::memory transposed_mem = zendnn::memory(transposed_md, zendnn_engine,
                                    nullptr);
    void *handle = intermediate_mem.get_data_handle();
    transposed_mem.set_data_handle(handle);

    sp.SetMemory(node.Output(OUT_TRANSPOSED), transposed_mem, true);
}

std::vector<int64_t> ZendnnTranspose::GetPerm(ZendnnNode &node) {
    auto attr = node.Attributes().find("perm");
    std::vector<int64_t> perm;
    if (attr != node.Attributes().end()) {
        perm.reserve(attr->second().ints_size());
        for (int i = 0; i < attr->second().ints_size(); ++i) {
            perm.push_back(attr->second().ints(i));
        }
    }
    return perm;
}
}  // namespace ort_zendnn
}  // namespace onnxruntime
