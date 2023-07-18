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

class ZendnnDynamicQuantizeLinear {
  public:
    enum InputTensors : int {
        IN_X = 0,  // Input tensor float32
    };

    enum OutputTensors : int {
        OUT_Y = 0,        // Quantized output tensor, tensor uint8
        OUT_Y_SCALE = 1,  // Output scale. It's a scalar, which means a per-tensor/layer quantization, tensor float32
        OUT_Y_ZP = 2,     // Output zero point. It's a scalar, which means a per-tensor/layer quantization, tensor uint8
    };

    ZendnnDynamicQuantizeLinear() = default;
    void CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);

  private:
    void WriteZeroToMem(zendnn::memory &mem);
    zendnn::memory::desc ChangeMemoryDescDataType(zendnn::memory::desc md,
            zendnn::memory::data_type dt);
};

}  // namespace ort_zendnn
}  // namespace onnxruntime