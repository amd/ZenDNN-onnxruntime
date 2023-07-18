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

#include "zendnn_reduce.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace ort_zendnn {

ZendnnReduce::ZendnnReduce() {}

// assume all dims are available
void ZendnnReduce::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                   ZendnnNode &node) {

    using namespace zendnn;

    // get the engine, currently only support either single gpu or single cpu device
    auto zendnn_engine = sp.GetEngine();

    enum ReduceOp {
        ReduceL1,
        ReduceL2,
        ReduceLogSum,
        ReduceLogSumExp,
        ReduceMax,
        ReduceMean,
        ReduceMin,
        ReduceProd,
        ReduceSum,
        ReduceSumSquare
    };

    ReduceOp reduce_op = ReduceSum;
    zendnn::algorithm algo = zendnn::algorithm::reduction_sum;
    if (node.OpType() == "ReduceL1") {
        reduce_op = ReduceL1;
        algo = zendnn::algorithm::reduction_norm_lp_power_p_sum;
    }
    else if (node.OpType() == "ReduceL2") {
        reduce_op = ReduceL2;
        algo = zendnn::algorithm::reduction_norm_lp_sum;
    }
    else if (node.OpType() == "ReduceLogSum") {
        reduce_op = ReduceLogSum;
        algo = zendnn::algorithm::reduction_sum;
    }
    else if (node.OpType() == "ReduceLogSumExp") {
        reduce_op = ReduceLogSumExp;
        algo = zendnn::algorithm::reduction_sum;
    }
    else if (node.OpType() == "ReduceMax") {
        reduce_op = ReduceMax;
        algo = zendnn::algorithm::reduction_max;
    }
    else if (node.OpType() == "ReduceMean") {
        reduce_op = ReduceMean;
        algo = zendnn::algorithm::reduction_mean;
    }
    else if (node.OpType() == "ReduceMin") {
        reduce_op = ReduceMin;
        algo = zendnn::algorithm::reduction_min;
    }
    else if (node.OpType() == "ReduceProd") {
        reduce_op = ReduceProd;
        algo = zendnn::algorithm::reduction_mul;
    }
    else if (node.OpType() == "ReduceSum") {
        reduce_op = ReduceSum;
        algo = zendnn::algorithm::reduction_sum;
    }
    else if (node.OpType() == "ReduceSumSquare") {
        reduce_op = ReduceSumSquare;
        algo = zendnn::algorithm::reduction_sum;
    }



    auto opset = node.SinceVersion();
    zendnn::memory::dims axes;
    if (reduce_op == ReduceSum) {
        // in ReduceSum opset older than version 13 the Axes came in as an attribute
        // after version 13 the axis is an optional tensor input.
        if (opset < 13) {
            axes = ReadAxes(node);
        }
        else {
            if (node.Input(IN_AXES).Exists()) {
                auto axes_mem = sp.GetMemory(node.Input(IN_AXES));
                zendnn::memory::dims axes_dims = axes_mem.get_desc().dims();
                int64_t *p_axes_data = (int64_t *)axes_mem.get_data_handle();
                axes = std::vector<int64_t>(p_axes_data, p_axes_data + axes_dims[0]);
            }
        }
    }
    else {
        axes = ReadAxes(node);
    }

    auto src_mem = sp.GetMemoryInOrtFormat(node.Input(IN_DATA), zendnn_engine);
    auto src_md = src_mem.get_desc();

    if (reduce_op == ReduceSum) {
        // If axes is empty and the noop_with_empty_axes != 0 return the IN_DATA as the output.
        if (axes.empty()) {
            if (NoOpWithEmptyAxes(node)) {
                sp.SetMemory(node.Output(OUT_REDUCED), src_mem, true);
                return;
            }
        }
    }

    //We need to calculate output tensor shape
    //First we initialize it with input shape and then we modify it based on the attribute values
    //This is because the ZENDNN primitive functionality is determined by the input and output shapes.
    auto src_dims = src_md.dims();
    auto ndim = src_dims.size();

    // convert negative axis values to the positive axis
    for (size_t i = 0; i < axes.size(); ++i) {
        axes[i] = HandleNegativeAxis(axes[i], ndim);
    }
    // Handle out of order and repeating dims.
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());

    // if axes is empty change all non-zero shape dims to 1
    if (axes.size() == 0) {
        for (size_t i = 0; i < ndim; ++i) {
            if (src_dims[i] != 0) {
                src_dims[i] = 1;
            }
        }
        //If there is axis, then make the respective dimensions 1, keeping the other dimension values untouched.
    }
    else {
        for (size_t i = 0; i < axes.size(); i++) {
            if (src_dims[axes[i]] != 0) {
                src_dims[axes[i]] = 1;
            }
        }
    }

    auto dst_shape = TensorShape(src_dims.data(), ndim);
    zendnn::memory::dims dst_dims_zendnn(dst_shape.GetDims().begin(),
                                      dst_shape.GetDims().end());
    auto dst_md = zendnn::memory::desc({dst_dims_zendnn}, src_md.data_type(),
                                       zendnn::memory::format_tag::any);

    // Check to see if the destination shape and source shape are the same.
    bool src_and_dst_dims_equal = true;
    if (src_md.dims().size() == dst_md.dims().size()) {
        for (size_t i = 0; i < src_md.dims().size(); ++i) {
            if (src_md.dims()[i] != dst_md.dims()[i]) {
                src_and_dst_dims_equal = false;
                break;
            }
        }
    }

    /*
    * ZenDNN will return an error if a reduction algorithm is called that does not result in a
    * shape reduction. For this reason we have code paths that are taken if the source dimensions and
    * destination dimensions are equal that will not call the reduction op.
    *
    * "ReduceLogSum" is equivelent to Log(ReduceSum(input))
    *   - if the reduction op is called then the eltwise_log post op will added to the reduction primitive.
    *   - if the reduction op is not called then the eltwise_log primitive is added as its own primitive
    *   - NOTE "ReduceLogSum" follows the code flow of "All other reduce ops" with the exception of the added
    *          post op and an extra check if src_dims == dest_dims.
    * "ReduceLogSumExp" is equivelent to Log(ReduceSum(Exp(input)))
    *   - if the reduction op is called then the eltwise_exp primitive is added before the reduction op
    *     the eletwise_log post op will be added to the reduction primitive
    *   - if the reduction op is not called then the input is not modified since Log(Exp(input) == input
    * "ReduceSumSquare" is equivelent to ReduceSum(Square(input))
    *   - the eltwise_square primitive is added before the reduction op
    *   - if the source and destination dimensions are not equal the reduction op is called
    * All other reduce ops
    *   - if the source and destination dimensions are not equal call the reduction op
    *   - otherwise don't modify the input.
    *
    * After the Reduction check the "KeepDims" attribute
    *  - if KeepDims == 1 the output is the result of the reduction op
    *  - if KeepDims == 0 we perform a squeeze operation on the output of the reduction op
    *  - NOTE: Even if reduction op is not called KeepDims attribute can result in the output being modified
    */
    zendnn::memory reduce_src_mem;
    zendnn::memory reduce_dst_mem;
    zendnn::primitive_attr zendnn_primitive_attr;
    if ((reduce_op == ReduceLogSum || reduce_op == ReduceLogSumExp) &&
            !src_and_dst_dims_equal) {
        zendnn::post_ops eltwise_post_op;
        eltwise_post_op.append_eltwise(1.0f, zendnn::algorithm::eltwise_log, 1.0f,
                                       1.0f);
        zendnn_primitive_attr.set_post_ops(eltwise_post_op);
    }

    if (reduce_op == ReduceLogSumExp) {
        if (!src_and_dst_dims_equal) {
            auto elementwise_desc = zendnn::eltwise_forward::desc(
                                        zendnn::prop_kind::forward_inference, zendnn::algorithm::eltwise_exp, src_md);
            auto elementwise_pd = zendnn::eltwise_forward::primitive_desc(elementwise_desc,
                                  zendnn_engine);

            auto elementwise_dst_mem = zendnn::memory(elementwise_pd.dst_desc(),
                                       zendnn_engine);

            auto elemenwise_primitive = zendnn::eltwise_forward(elementwise_pd);
            sp.AddPrimitive(elemenwise_primitive, {{ZENDNN_ARG_SRC, src_mem},
                {ZENDNN_ARG_DST, elementwise_dst_mem}
            });
            auto reduce_desc = zendnn::reduction::desc(algo, src_md, dst_md, 0.f, 0.f);
            auto reduce_pd = zendnn::reduction::primitive_desc(reduce_desc,
                             zendnn_primitive_attr, zendnn_engine);

            reduce_dst_mem = zendnn::memory(reduce_pd.dst_desc(), zendnn_engine);

            auto reducemean_op = zendnn::reduction(reduce_pd);
            sp.AddPrimitive(reducemean_op, {{ZENDNN_ARG_SRC, elementwise_dst_mem},
                {ZENDNN_ARG_DST, reduce_dst_mem}
            });
        }
        else {
            reduce_dst_mem = src_mem;
        }
    }
    else if (reduce_op == ReduceSumSquare) {
        auto elementwise_desc = zendnn::eltwise_forward::desc(
                                    zendnn::prop_kind::forward_inference, zendnn::algorithm::eltwise_square,
                                    src_md);
        auto elementwise_pd = zendnn::eltwise_forward::primitive_desc(elementwise_desc,
                              zendnn_engine);

        auto elementwise_dst_mem = zendnn::memory(elementwise_pd.dst_desc(),
                                   zendnn_engine);

        auto elemenwise_primitive = zendnn::eltwise_forward(elementwise_pd);
        sp.AddPrimitive(elemenwise_primitive, {{ZENDNN_ARG_SRC, src_mem},
            {ZENDNN_ARG_DST, elementwise_dst_mem}
        });
        if (!src_and_dst_dims_equal) {
            auto reduce_desc = zendnn::reduction::desc(algo, src_md, dst_md, 0.f, 0.f);
            auto reduce_pd = zendnn::reduction::primitive_desc(reduce_desc, zendnn_engine);

            reduce_dst_mem = zendnn::memory(reduce_pd.dst_desc(), zendnn_engine);

            auto reducemean_op = zendnn::reduction(reduce_pd);
            sp.AddPrimitive(reducemean_op, {{ZENDNN_ARG_SRC, elementwise_dst_mem},
                {ZENDNN_ARG_DST, reduce_dst_mem}
            });
        }
        else {
            reduce_dst_mem = elementwise_dst_mem;
        }
    }
    else {
        // If calculated source and destination shape are the same do not do the reduction operation.
        if (!src_and_dst_dims_equal) {
            float p_val = 0.f;
            if (reduce_op == ReduceL1) {
                p_val = 1.0f;
            }
            else if (reduce_op == ReduceL2) {
                p_val = 2.0f;
            }

            auto reduce_desc = zendnn::reduction::desc(algo, src_md, dst_md, p_val, 0.f);
            auto reduce_pd = zendnn::reduction::primitive_desc(reduce_desc,
                             zendnn_primitive_attr, zendnn_engine);

            // If using GPU this will move the memory from the CPU to the GPU.
            reduce_src_mem = sp.GetMemoryAndReshape(node.Input(IN_DATA),
                                                    reduce_pd.src_desc(), zendnn_engine);
            reduce_dst_mem = zendnn::memory(reduce_pd.dst_desc(), zendnn_engine);

            auto reducemean_op = zendnn::reduction(reduce_pd);
            sp.AddPrimitive(reducemean_op, {{ZENDNN_ARG_SRC, reduce_src_mem},
                {ZENDNN_ARG_DST, reduce_dst_mem}
            });
        }
        else {
            if (reduce_op == ReduceLogSum) {
                auto elementwise_desc = zendnn::eltwise_forward::desc(
                                            zendnn::prop_kind::forward_inference, zendnn::algorithm::eltwise_log, src_md);
                auto elementwise_pd = zendnn::eltwise_forward::primitive_desc(elementwise_desc,
                                      zendnn_engine);

                reduce_dst_mem = zendnn::memory(elementwise_pd.dst_desc(), zendnn_engine);

                auto elemenwise_primitive = zendnn::eltwise_forward(elementwise_pd);
                sp.AddPrimitive(elemenwise_primitive, {{ZENDNN_ARG_SRC, src_mem},
                    {ZENDNN_ARG_DST, reduce_dst_mem}
                });
            }
            else {
                reduce_dst_mem = src_mem;
            }
        }
    }


    // If keepdims != 0 set the output to the reduce op results
    auto keepdims = Keepdims(node);
    if (keepdims) {
        if (src_and_dst_dims_equal) {
            sp.SetMemory(node.Output(OUT_REDUCED), reduce_dst_mem, true);
        }
        else {
            sp.SetMemory(node.Output(OUT_REDUCED), reduce_dst_mem);
        }
        // if keepdims == 0 we do a squeeze operation on reduce output shape.
    }
    else {
        std::vector<int64_t> output_shape;
        size_t j = 0;
        for (size_t i = 0; i < ndim; ++i) {
            if ((j < axes.size() && axes[j] == static_cast<int64_t>(i)) ||
                    (axes.size() == 0 && src_dims[i] == 1)) {
                if (src_dims[i] != 1) {
                    auto dims_span = gsl::make_span(src_dims);
                    ORT_ENFORCE(src_dims[i] == 1, "Dimension of input ", i,
                                " must be 1 instead of ", src_dims[i],
                                ". shape=", TensorShape(dims_span));
                }
                ++j;
                continue;
            }

            if ((j < axes.size() && axes[j] == static_cast<int64_t>(i) &&
                    src_dims[i] == 0) ||
                    (axes.size() == 0 && src_dims[i] == 0)) {
                if (!keepdims) {
                    auto dims = src_md.dims();
                    ORT_ENFORCE(keepdims,
                                "Can't reduce on dim with value of 0 if 'keepdims' is false. "
                                "Invalid output shape would be produced. input_shape:",
                                TensorShape(gsl::make_span(dims)));
                }
            }
            output_shape.push_back(src_dims[i]);
        }

        // ZenDNN does not support scalar output if the output shape is {} change it to {1}
        bool is_scalar_output = false;
        if (output_shape.empty()) {
            output_shape.push_back(1);
            is_scalar_output = true;
        }
        zendnn::memory::desc squeeze_md(output_shape, node.Input(IN_DATA).Type(),
                                        sp.GetZendnnFormat(output_shape.size()));
        zendnn::memory squeeze_mem = zendnn::memory(squeeze_md, zendnn_engine, nullptr);
        // if the src and dst dims are equal then we will have a valid data handle here.
        // Otherwise we must get the data handle at runtime using the AddReshape function.
        // reading the data handle directy is more efficent if is it possible.
        if (!src_and_dst_dims_equal) {
            squeeze_mem.set_data_handle(reduce_dst_mem.get_data_handle());
        }
        else {
            sp.AddReshape(reduce_dst_mem, squeeze_mem);
        }
        sp.SetMemory(node.Output(OUT_REDUCED), squeeze_mem, true, is_scalar_output);
    }
}

std::vector<int64_t> ZendnnReduce::ReadAxes(ZendnnNode &node) {
    auto attr = node.Attributes().find("axes");
    std::vector<int64_t> axes;
    if (attr != node.Attributes().end()) {
        auto &proto = attr->second();
        axes.reserve(proto.ints_size());
        for (int i = 0; i < proto.ints_size(); i++) {
            axes.push_back(proto.ints(i));
        }
    }
    return axes;
}

bool ZendnnReduce::Keepdims(ZendnnNode &node) {
    auto attr = node.Attributes().find("keepdims");
    if (attr != node.Attributes().end() &&
            attr->second().i() == 0) {
        return false;
    }
    return true;
}

bool ZendnnReduce::NoOpWithEmptyAxes(ZendnnNode &node) {
    auto attr = node.Attributes().find("noop_with_empty_axes");
    if (attr != node.Attributes().end() &&
            attr->second().i() != 0) {
        return true;
    }
    return false;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
