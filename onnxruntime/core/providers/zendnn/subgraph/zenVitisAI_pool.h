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

class ZenVitisAIPool {
  public:
    enum InputTensors : int {
        IN_X = 0
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

    ZenVitisAIPool();
    void CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);

  private:
    zendnn::memory::dims GetKernelShape(const zendnn::memory::dims &src_dims,
                                        ZendnnNode &node);
    AutoPadType GetAutoPad(ZendnnNode &node);
    std::vector<int64_t> GetPadding(ZendnnNode &node, PoolShape shape);
    zendnn::memory::dims GetStrides(ZendnnNode &node, PoolShape shape);
    std::string get_string_attribute(ZendnnNode &node, std::string tstring);
    zendnn::memory::dims InferOutputDims(ZendnnNode &node,
                                         const zendnn::memory::dims &src_dims,
                                         const zendnn::memory::dims &kernel_shape, const zendnn::memory::dims &strides);
    zendnn::memory::dims GetPaddingLeft(const std::vector<int64_t> padding);
    zendnn::memory::dims GetPaddingRight(const std::vector<int64_t> padding);
};

}  // namespace ort_zendnn
}  // namespace onnxruntime