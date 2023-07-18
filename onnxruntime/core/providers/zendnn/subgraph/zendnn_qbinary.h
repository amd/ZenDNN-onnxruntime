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

class ZendnnQBinary {
  public:
    enum InputTensors : int {
        IN_A = 0,
        IN_A_SCALE = 1,
        IN_A_ZERO_POINT = 2,
        IN_B = 3,
        IN_B_SCALE = 4,
        IN_B_ZERO_POINT = 5,
        IN_C_SCALE = 6,
        IN_C_ZERO_POINT = 7,
    };

    enum OutputTensors : int {
        OUT_Y = 0
    };

    ZendnnQBinary();
    void CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);
    void Padd(zendnn::memory::desc *target_md, size_t pad);
};

}  // namespace ort_zendnn
}  // namespace onnxruntime
