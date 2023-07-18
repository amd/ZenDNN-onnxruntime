/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

// Copyright(C) 2022 Intel Corporation
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

class ZendnnDequantizeLinear {
  public:
    enum InputTensors : int {
        IN_X = 0,
        IN_X_SCALE = 1,
        IN_X_ZERO_POINT = 2,  // Optional
    };

    enum OutputTensors : int {
        OUT_Y = 0,
    };

    ZendnnDequantizeLinear() = default;
    void CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);

  private:
    bool isZeroPointNonZero(zendnn::memory *zp_mem);
    int64_t GetAxis(ZendnnNode &node, size_t x_dims);
    void Padd(zendnn::memory::desc *target, size_t front_pad, size_t back_pad);
    void ValidateDims(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);
    void ValidateType(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);
};

}  // namespace ort_zendnn
}  // namespace onnxruntime