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

#pragma once
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_zendnn {

class ZendnnQPool {
  public:
    enum InputTensors : int {
        IN_X = 0,
        IN_X_SCALE = 1,
        IN_X_ZERO_POINT = 2,
        IN_Y_SCALE = 3,
        IN_Y_ZERO_POINT = 4
    };

    enum OutputTensors : int {
        OUT_Y = 0
    };

    enum PoolShape : size_t {
        SHAPE_UNKNOWN = 0,
        SHAPE_1D = 1,
        SHAPE_2D = 2,
        SHAPE_3D = 3
    };

    ZendnnQPool();
    void CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);

  private:
    void Padd(zendnn::memory::desc *target, size_t pad);
    AutoPadType GetAutoPad(ZendnnNode &node);
    int64_t GetCeilMode(ZendnnNode &node);
    int64_t GetCountIncludePadding(ZendnnNode &node);
    zendnn::memory::dims GetDilations(ZendnnNode &node, PoolShape shape);
    zendnn::memory::dims GetKernelShape(const zendnn::memory::dims &src_dims,
                                        ZendnnNode &node, size_t channels_last_);
    /* This will return the calculated padding taking into account the DEPRECATED auto_pad attribute */
    std::vector<int64_t> InferPadding(ZendnnNode &node,
                                      const zendnn::memory::dims &src_dims, const zendnn::memory::dims &kernel_shape,
                                      const zendnn::memory::dims &strides);
    std::vector<int64_t> GetPadding(ZendnnNode &node, PoolShape shape);
    zendnn::memory::dims GetPaddingLeft(const std::vector<int64_t> padding);
    zendnn::memory::dims GetPaddingRight(const std::vector<int64_t> padding);
    int64_t GetChannelLast(ZendnnNode &node);
    zendnn::memory::dims GetStrides(ZendnnNode &node, PoolShape shape);

    zendnn::memory::dims InferOutputDims(ZendnnNode &node,
                                         const zendnn::memory::dims &src_dims, const zendnn::memory::dims &kernel_shape,
                                         const zendnn::memory::dims &strides, size_t channel_last);
    bool IsGlobalPooling(ZendnnNode &node) const;
    bool dummy_maxpool_node = false;
};

}  // namespace ort_zendnn
}  // namespace onnxruntime