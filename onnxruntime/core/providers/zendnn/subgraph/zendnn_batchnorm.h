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

class ZendnnBatchNorm {
  public:
    enum InputTensors : int {
        IN_X = 0,
        IN_SCALE = 1,
        IN_B = 2,
        IN_MEAN = 3,
        IN_VAR = 4
    };

    enum OutputTensors : int {
        OUT_Y = 0
    };
    ZendnnBatchNorm();
    void CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);
    float ReadEpsilon(ZendnnNode &node);
};

}  // namespace ort_zendnn
}  // namespace onnxruntime