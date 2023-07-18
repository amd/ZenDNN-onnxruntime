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

class ZendnnLayerNorm {
  public:

    typedef std::pair<zendnn::layer_normalization_forward, std::unordered_map<int, zendnn::memory>>
            ln_components;

    enum InputTensorsSLN : int {
        IN_INPUT = 0,
        IN_SKIP = 1,
        IN_SLN_GAMMA = 2,
        IN_BETA = 3,        // Optional
        IN_SLN_BIAS = 4     // Optional
    };

    enum InputTensorsLN : int {
        // IN_INPUT = 0,
        IN_LN_GAMMA = 1,
        IN_LN_BIAS = 2      // Optional
    };

    enum OutputTensors : int {
        OUT_OUTPUT = 0,
        OUT_MEAN = 1,        // Optional
        OUT_INV_STD_VAR = 2  // Optional
    };
    ZendnnLayerNorm();

    void CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);

  private:
    zendnn::memory BuildSLN(ZendnnSubgraphPrimitive &sp, ZendnnNode &node,
                            zendnn::engine zendnn_engine);
    void ValidateDims(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);
    float GetEpsilon(ZendnnNode &node);
};

}  // namespace ort_zendnn
}  // namespace onnxruntime