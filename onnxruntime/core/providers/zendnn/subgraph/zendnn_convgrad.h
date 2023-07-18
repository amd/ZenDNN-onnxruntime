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

//class ZendnnSubgraphPrimitive;
//class ZendnnNode;

namespace onnxruntime {
namespace ort_zendnn {

class ZendnnConvGrad {
  public:
    enum InputTensors : int {
        IN_DY = 0,
        IN_X = 1,
        IN_W = 2
    };

    enum OutputTensors : int {
        OUT_DX = 0,
        OUT_DW = 1,
        OUT_DB = 2
    };

    enum ConvShape : size_t {
        SHAPE_UNKNOWN = 0,
        SHAPE_1D = 1,
        SHAPE_2D = 2,
        SHAPE_3D = 3
    };

    ZendnnConvGrad();
    void CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);

  private:
    std::vector<int64_t> GetKernelShape(ZendnnNode &node);
    /* Get the 'pads' attribute */
    zendnn::memory::dims GetPads(ZendnnNode &node, ConvShape shape);
    /* Get the padding left values from the infered pads */
    zendnn::memory::dims GetPaddingLeft(const std::vector<int64_t> &onnx_padding,
                                        ConvShape shape);
    /* Get the padding right values from the infered pads */
    zendnn::memory::dims GetPaddingRight(const std::vector<int64_t> &onnx_padding,
                                         ConvShape shape);
    /*
     * Get the 'dilations' attribute.
     *  Note dilations in ZenDNN and Onnx differ:
     *    - For Onnx a non-dilated kernel would be all 1s
     *    - For ZenDNN a non-dilated kernel would be all 0s
     *
     * The memory dimentions returned is in the form expected for ZenDNN each dilation dimention
     * will be 1 less than the dilated dimention expected by Onnx specification. Be aware of this
     * fact as 'dilations' are used in any calcuations since this could result in an off-by-one
     * error.
     */
    zendnn::memory::dims GetDilations(ZendnnNode &node, ConvShape shape);
    /* Get the 'strides' attribute */
    zendnn::memory::dims GetStrides(ZendnnNode &node, ConvShape shape);
    /* Get the 'group' attributes */
    int64_t GetGroup(ZendnnNode &node);
};

}  // namespace ort_zendnn
}  // namespace ort_zendnn