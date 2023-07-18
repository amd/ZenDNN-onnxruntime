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

class ZenVitisAIConv2D {
  public:
    enum InputTensors : int {
        IN_X = 0,
        IN_W = 1,
        IN_B = 2,
        IN_BINARY = 3
    };

    enum OutputTensors : int {
        OUT_Y = 0
    };

    enum ConvShape : size_t {
        SHAPE_UNKNOWN = 0,
        SHAPE_1D = 1,
        SHAPE_2D = 2,
        SHAPE_3D = 3
    };

    ZenVitisAIConv2D();
    void CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);

  private:
    std::string get_string_attribute(ZendnnNode &node, std::string tstring);
    int get_int_attribute(ZendnnNode &node, std::string tstring);
    float get_float_attribute(ZendnnNode &node, std::string tstring);

    zendnn::memory::dims GetPaddingLeft(const std::vector<int64_t> &onnx_padding,
                                        ConvShape shape);

    zendnn::memory::dims GetPaddingRight(const std::vector<int64_t> &onnx_padding,
                                         ConvShape shape);

    int64_t GetGroup(ZendnnNode &node);

    AutoPadType GetAutoPad(ZendnnNode &node);

    zendnn::memory::dims GetDilations(ZendnnNode &node, ConvShape shape);

    float GetAlpha(ZendnnNode &node);

    std::vector<int64_t> GetKernelShape(ZendnnNode &node);

    std::vector<int64_t> GetPads(ZendnnNode &node);

    zendnn::memory::dims GetStrides(ZendnnNode &node, ConvShape shape);

    zendnn::memory::dims InferOutputShape(ZendnnNode &node,
                                          const zendnn::memory::dims &x_shape,
                                          const zendnn::memory::dims &w_shape,
                                          const std::vector<int64_t> &kernel_shape,
                                          const zendnn::memory::dims &strides,
                                          const zendnn::memory::dims &dilations,
                                          const std::vector<int64_t> &pads);
};

}  // namespace ort_zendnn
}  // namespace onnxruntime