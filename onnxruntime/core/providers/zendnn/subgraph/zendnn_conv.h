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

class ZendnnConv {
  public:
    enum InputTensors : int {
        IN_X = 0,
        IN_W = 1,
        IN_B = 2,
        IN_BINARY = 3
    };

    enum OutputTensors : int {
        OUT_Y = 0,
    };

    enum ConvShape : size_t {
        SHAPE_UNKNOWN = 0,
        SHAPE_1D = 1,
        SHAPE_2D = 2,
        SHAPE_3D = 3
    };

    ZendnnConv();
    void CreatePrimitive(ZendnnSubgraphPrimitive &sp, ZendnnNode &node);
    /* Get destination memory dims (number of elements). Neccessary for calculating offset address. */
    void GetDestMemoryDims(ZendnnSubgraphPrimitive &sp, ZendnnNode &node,
                           zendnn::memory::dims &dst_dims);
    /* Set destination memory (handle+offset). Neccessary for inplace concat. */
    void SetDestinationMemoryInfo(zendnn::memory::desc &ldst_md,
                                  zendnn::memory &dst_mem, int concat_order, int refCount,
                                  bool addOutput = false);

  private:
    /* These variables are necessary for getting memory info for performing inplace operations */
    zendnn::memory::desc _ldst_md;
    zendnn::memory _dst_mem;

    int concat_order = 0;
    int ref_count = 0;

    bool add_output = true;
    /*
    * Return the infered padding.
    *
    * The padding will be based on the specified padding or will infered based on the
    * Onnx 'auto_pad' attributes.
    *
    * This will return the padding in the format specified in the Onnx specification.
    * > Format should be as follows [x1_begin, x2_begin...x1_end, x2_end,...],
    * > where xi_begin the number of pixels added at the beginning of axis `i`
    * > and xi_end, the number of pixels added at the end of axis `i`.
    */
    std::vector<int64_t> GetInferedPads(ZendnnNode &node,
                                        const zendnn::memory::dims &src_dims,
                                        const zendnn::memory::dims &dilations,
                                        const std::vector<int64_t> &kernel_shape,
                                        const zendnn::memory::dims &strides);
    /* Get the padding left values from the infered pads */
    zendnn::memory::dims GetPaddingLeft(const std::vector<int64_t> &onnx_padding,
                                        ConvShape shape);
    /* Get the padding right values from the infered pads */
    zendnn::memory::dims GetPaddingRight(const std::vector<int64_t> &onnx_padding,
                                         ConvShape shape);

    /*
    * Collection of functions to get OnnxRuntime attributes. Note, if the attribute is used directly by
    * ZenDNN the return type is converted to the format expected by ZenDNN not the type expected by
    * OnnxRuntime. Typically this means returning `zendnn::memory::dims` instead of `vector<int64_t>`.
    */
    /* Get the 'auto_pad' attribute */
    AutoPadType GetAutoPad(ZendnnNode &node);

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

    /* Get the 'group' attributes */
    int64_t GetGroup(ZendnnNode &node);

    /* Get the 'alpha' attribute */
    float GetAlpha(ZendnnNode &node);
    float GetReluAlpha(ZendnnNode &node);

    float GetMin(ZendnnNode &node, float default_min);
    float GetMax(ZendnnNode &node, float default_max);

    /* Get the 'kernel_shape' attribute */
    std::vector<int64_t> GetKernelShape(ZendnnNode &node);

    /* Get the 'pads' attribute */
    std::vector<int64_t> GetPads(ZendnnNode &node);

    /* Get the 'strides' attribute */
    zendnn::memory::dims GetStrides(ZendnnNode &node, ConvShape shape);

    /*
    * ComputePad is copy/paste of a the ComputePad found in core/providers/common.h
    * With some minor modifications. i.e. return bool instead of status.
    * ComputePad is not exposed to the shared library so this copy is used instead.
    *
    * Returns true if pads successfully computed.
    */
    bool ComputePad(const int64_t in_dim,
                    const int64_t stride,
                    const int64_t kernel,
                    const int64_t dilation,
                    AutoPadType pad_type,
                    int64_t &pad_head, /* output param */
                    int64_t &pad_tail, /* output param */
                    bool force_symmetric_auto_padding = false);

    /*
    * Use input shapes and attributes to figure out the output shape that will
    * result from the convolution.
    */
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
