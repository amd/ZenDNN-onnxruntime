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

#pragma once
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

class ZendnnPoolGrad {
  public:
    enum InputTensors : int {
        IN_DY = 0,
        IN_INDICES = 1
    };

    enum OutputTensors : int {
        OUT_DX = 0
    };

    enum PoolShape : size_t {
        SHAPE_UNKNOWN = 0,
        SHAPE_1D = 1,
        SHAPE_2D = 2,
        SHAPE_3D = 3
    };

    ZendnnPoolGrad();
    void CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);

  private:
    AutoPadType GetAutoPad(ZendnnNode &node);
    int64_t GetCeilMode(ZendnnNode &node);
    int64_t GetCountIncludePadding(ZendnnNode &node);
    zendnn::memory::dims GetDilations(ZendnnNode &node, PoolShape shape);
    zendnn::memory::dims GetKernelShape(ZendnnNode &node);
    /* This will return the calculated padding taking into account the DEPRECATED auto_pad attribute */
    std::vector<int64_t> InferPadding(ZendnnNode &node,
                                      const zendnn::memory::dims &src_dims, const zendnn::memory::dims &kernel_shape,
                                      const zendnn::memory::dims &strides);
    std::vector<int64_t> GetPadding(ZendnnNode &node, PoolShape shape);
    zendnn::memory::dims GetPaddingLeft(const std::vector<int64_t> padding);
    zendnn::memory::dims GetPaddingRight(const std::vector<int64_t> padding);
    int64_t GetStorageOrder(ZendnnNode &node);
    zendnn::memory::dims GetStrides(ZendnnNode &node, PoolShape shape);

    zendnn::memory::dims InferOutputDims(ZendnnNode &node,
                                         const zendnn::memory::dims &src_dims, const zendnn::memory::dims &kernel_shape,
                                         const zendnn::memory::dims &strides);
    bool IsGlobalPooling(ZendnnNode &node) const;
};

}  // namespace ort_zendnn
}  // namespace onnxruntime