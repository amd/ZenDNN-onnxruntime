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

#include <float.h>
#include "zendnn_qconv.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"
#include <cassert>

#define DST_SCALE_INPLACE

namespace onnxruntime {
namespace ort_zendnn {

ZendnnQConv::ZendnnQConv() {}

void ZendnnQConv:: GetDestMemoryDims(ZendnnSubgraphPrimitive &sp,
                                     ZendnnNode &node, zendnn::memory::dims &dst_dims) {
    auto zendnn_engine = sp.GetEngine();
    auto src_dims = sp.GetMemory(node.Input(IN_X)).get_desc().dims();
    auto weights_dims = sp.GetMemory(node.Input(IN_W)).get_desc().dims();
    auto kernel_shape = GetKernelShape(node);
    ConvShape shape = static_cast<ConvShape>(kernel_shape.size());
    assert(shape != SHAPE_UNKNOWN);
    auto strides = GetStrides(node, shape);
    auto dilations = GetDilations(node, shape);
    // Use GetInferedPads here instead of GetPads since this will acount for the `auto_pad` attribute in its return value
    auto padding = GetInferedPads(node, src_dims, dilations, kernel_shape, strides);
    auto padding_left = GetPaddingLeft(padding, shape);
    auto padding_right = GetPaddingRight(padding, shape);
    // Figure out the output shape based on the inputs
    dst_dims = InferOutputShape(node, src_dims, weights_dims, kernel_shape, strides,
                                dilations, padding);
}

void ZendnnQConv::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                  ZendnnNode &node) {
    using tag = zendnn::memory::format_tag;
    using dt =  zendnn::memory::data_type;

    // Depending on the dimensions of QLinearConv,
    // LPGEMM path will be taken
    int use_lpgemm = 0;
    // If disabled, s32 API of LPGEMM will be used.
    // s32 API is for Genoa systems.
    // s16 API works on all systems.
    bool s16_lpgemm_enabled = 0;
    // Set environment variable S16_LPGEMM_ENABLED to 1
    // to use S16 API of LPGEMM (must for Milan / Systems
    // without AVX512 support)
    const std::string s16_env = onnxruntime::GetEnvironmentVar("S16_LPGEMM_ENABLED");
    if (!s16_env.empty()) {
        s16_lpgemm_enabled = (std::stoi(s16_env) == 0 ? false : true);
    }
    if(s16_lpgemm_enabled != 0) {
            s16_lpgemm_enabled = 1;
    }

    auto zendnn_engine = sp.GetEngine();

    auto src_dims = sp.GetMemory(node.Input(IN_X)).get_desc().dims();
    auto conv_weights_mem = sp.GetMemory(node.Input(IN_W));
    auto weight_md = conv_weights_mem.get_desc();
    weight_md.data.format_kind = zendnn_format_kind_t::zendnn_format_kind_any;
    auto weight_dims_original = conv_weights_mem.get_desc().dims();
    zendnn::memory::dims weight_dims = weight_dims_original;

    bool bias_exists  = (node.InputCount() > 8)?true:false;
    zendnn::memory conv_bias_mem;
    if (bias_exists) {
        conv_bias_mem = sp.GetMemory(node.Input(IN_B));
    }

    /*
    * Get any inputs required for the zendnn::convolution_forward::desc
    * beyond the zendnn:memory::desc:
    *  -dilations
    *  - strides
    *  - padding_left and padding_right
    */
    auto kernel_shape = GetKernelShape(node);
    ConvShape shape = static_cast<ConvShape>(kernel_shape.size());
    assert(shape != SHAPE_UNKNOWN);

    auto group = GetGroup(node);
    if (group != 1) {
        weight_dims.insert(weight_dims.begin(), group);
        weight_dims[1] = static_cast<int64_t>(weight_dims_original[0] / group);
        zendnn::memory::format_tag format = zendnn::memory::format_tag::any;
        switch (shape) {
        case onnxruntime::ort_zendnn::ZendnnQConv::SHAPE_UNKNOWN: {
            // use format_tag::any
            break;
        }
        case onnxruntime::ort_zendnn::ZendnnQConv::SHAPE_1D: {
            format = zendnn::memory::format_tag::goiw;
            break;
        }
        case onnxruntime::ort_zendnn::ZendnnQConv::SHAPE_3D: {
            format = zendnn::memory::format_tag::goidhw;
            break;
        }
        default:
            // use format_tag::any
            break;
        }
        weight_md = zendnn::memory::desc({weight_dims}, node.Input(IN_W).Type(),
                                         format);
    }

    auto strides = GetStrides(node, shape);
    auto dilations = GetDilations(node, shape);
    // Use GetInferedPads here instead of GetPads since this will acount for the `auto_pad` attribute in its return value
    auto padding = GetInferedPads(node, src_dims, dilations, kernel_shape, strides);
    auto padding_left = GetPaddingLeft(padding, shape);
    auto padding_right = GetPaddingRight(padding, shape);

    // Figure out the output shape based on the inputs
    auto dst_mem_dims = InferOutputShape(node, src_dims, weight_dims_original,
                                         kernel_shape, strides, dilations, padding);

#ifdef ENABLE_TRAINING
    auto prop_kind = zendnn::prop_kind::forward_training;
#else
    auto prop_kind = zendnn::prop_kind::forward_inference;
#endif  // ENABLE_TRAINING

    float *x_scale = (float *)sp.GetMemory(node.Input(
            IN_X_SCALE)).get_data_handle();
    float *w_scale = (float *)sp.GetMemory(node.Input(
            IN_W_SCALE)).get_data_handle();
    float *y_scale = (float *)sp.GetMemory(node.Input(
            IN_Y_SCALE)).get_data_handle();
    /* Reading like scalars. TODO: read like arrays (if need arises) */
    auto sum_scale = 1.0, add_out_scale = 1.0;

    std::vector<int> src_zero_points(1), dst_zero_points(1);
    if (node.Input(IN_X_ZERO_POINT).Type() == dt::u8) {
        src_zero_points[0] = (int)(*(uint8_t *)sp.GetMemory(node.Input(
                                       IN_X_ZERO_POINT)).get_data_handle());
    }
    else if (node.Input(IN_X_ZERO_POINT).Type() == dt::s8) {
        src_zero_points[0] = (int)(*(int8_t *)sp.GetMemory(node.Input(
                                       IN_X_ZERO_POINT)).get_data_handle());
    }
    else {
        LOGS_DEFAULT(ERROR) <<"Invalid. Non-integer zero-point detected.";
    }

    if (node.Input(IN_Y_ZERO_POINT).Type() == dt::u8) {
        dst_zero_points[0] = (int)(*(uint8_t *)sp.GetMemory(node.Input(
                                       IN_Y_ZERO_POINT)).get_data_handle());
    }
    else if (node.Input(IN_Y_ZERO_POINT).Type() == dt::s8) {
        dst_zero_points[0] = (int)(*(int8_t *)sp.GetMemory(node.Input(
                                       IN_Y_ZERO_POINT)).get_data_handle());
    }
    else {
        LOGS_DEFAULT(ERROR) <<"Invalid. Non-integer zero-point detected.";
    }

    std::vector<int> add_out_zero_points(1);
    /* All Conv+Add fusion variants.*/
    if (node.OpType() == "QConvAdd" ||
            node.OpType() == "QConvAdd_v1" ||
            node.OpType() == "QConvAddRelu") {
        sum_scale = (*(float *)sp.GetMemory(node.Input(
                                                IN_SUM_SCALE)).get_data_handle());
        add_out_scale = (*(float *)sp.GetMemory(node.Input(
                IN_ADD_OUT_SCALE)).get_data_handle());

        if (node.Input(IN_ADD_OUT_ZP).Type() == dt::u8) {
            add_out_zero_points[0] = (int)(*(uint8_t *)sp.GetMemory(node.Input(
                                               IN_ADD_OUT_ZP)).get_data_handle());
        }
        else if (node.Input(IN_ADD_OUT_ZP).Type() == dt::s8) {
            add_out_zero_points[0] = (int)(*(int8_t *)sp.GetMemory(node.Input(
                                               IN_ADD_OUT_ZP)).get_data_handle());
        }
        else {
            LOGS_DEFAULT(ERROR) <<"Invalid. Non-integer zero-point detected.";
        }
    }

    int min = INT_MIN;
    int max = INT_MAX;
    auto min_input_index = -1;
    auto max_input_index = -1;

    if (node.OpType() == "QConvClip") {
        min_input_index = IN_B + 1;
        max_input_index = IN_B + 2;

        auto conv_min_mem = sp.GetMemory(node.Input(min_input_index));
        if (conv_min_mem.get_desc().get_size() != 0) {
            if (node.Input(min_input_index).Type() == dt::u8){
                min = (int)*((uint8_t *)conv_min_mem.get_data_handle());
            } else if (node.Input(min_input_index).Type() == dt::s8) {
                min = (int)*((int8_t *)conv_min_mem.get_data_handle());
            } else {
                LOGS_DEFAULT(ERROR) <<"Non-integer min input for Clip op detected.";
            }
        }

        auto conv_max_mem = sp.GetMemory(node.Input(max_input_index));
        if (conv_max_mem.get_desc().get_size() != 0) {
            if (node.Input(max_input_index).Type() == dt::u8){
                max = (int)*((uint8_t *)conv_max_mem.get_data_handle());
            } else if (node.Input(max_input_index).Type() == dt::s8) {
                max = (int)*((int8_t *)conv_max_mem.get_data_handle());
            } else {
                LOGS_DEFAULT(ERROR) <<"Non-integer max input for Clip op detected.";
            }
        }
    }

    auto wScaleSize = sp.GetMemory(node.Input(IN_W_SCALE)).get_desc().dims()[0];

    zendnn::primitive_attr conv_attr;

    if (node.OpType() == "QLinearConv_v2") {
        LOGS_DEFAULT(INFO) <<"node.OpType() == QLinearConv_v2";
    }
    else {
        if (src_zero_points[0] != (int)0) {
            conv_attr.set_zero_points(ZENDNN_ARG_SRC, 0, src_zero_points);
        }
    }

    if ((node.OpType() == "QLinearConv" ||
            node.OpType() == "QLinearConv_v2") &&
            dst_zero_points[0] != (int)0) {
        conv_attr.set_zero_points(ZENDNN_ARG_DST, 0, dst_zero_points);
    }

    // LPGEMM APIs supported with ONNX RT
    // u8s8s32os8
    // s8s8s32os8
    // u8s8s16os8
    // s8s8s16os8
    auto dstType_tmp = node.Input(IN_Y_ZERO_POINT).Type();
    auto srcType_tmp = node.Input(IN_X).Type();
    // Conditions checked for LPGEMM path
    // 1x1 kernel
    // dst zero point = 0
    // stride = 1
    // dst Type = s8
    // src Type = u8 or s8
    // Conv op not fused with Add op
    if(weight_dims[2] == 1 && weight_dims[3] == 1 && dst_zero_points[0] == 0 && strides[0] == 1
                  && dstType_tmp == dt::s8
                  && (srcType_tmp == dt::u8 || srcType_tmp == dt::s8)
                  && node.OpType() == "QLinearConv") {
        use_lpgemm = 1;
    }

    // Set environment variable LPGEMM_PATH_ENABLED=1
    // to take LPGEMM path
    // By default, Direct path is taken.
    int lpgemm_check = 0;
    const std::string lpgemm_env = onnxruntime::GetEnvironmentVar("LPGEMM_PATH_ENABLED");
    if (!lpgemm_env.empty()) {
        lpgemm_check = std::stoi(lpgemm_env);
    }
    if(lpgemm_check == 1 && use_lpgemm == 1)
      use_lpgemm = 1;
    else
      use_lpgemm = 0;

    /* Sets scales and zero points appropriately!! */
    std::vector<float> scales(wScaleSize);
    for (long int i=0; i<wScaleSize; i++) {
        scales[i] = x_scale[0]*w_scale[i]/y_scale[0];
    }
    if (wScaleSize == 1) {
        conv_attr.set_output_scales(0,
                                    scales);    //Whole tensor scaling. // [N, OC, H, W] = 0[1, 2, 3, 4]
    }
    else {
        conv_attr.set_output_scales(2,
                                    scales);    //Per-channel scaling. Channel=1<<1.// [N, OC, H, W] = 0[1, 2, 3, 4]
    }

    auto srcType = node.Input(IN_X).Type();
    auto conv_src_md = zendnn::memory::desc({src_dims}, srcType, tag::any);
    if(use_lpgemm > 0) {
        // With LPGEMM, src tag is set to NHWC
        conv_src_md = zendnn::memory::desc({src_dims}, srcType, tag::nhwc);
    }
    auto conv_weights_md = zendnn::memory::desc({weight_dims}, dt::s8, tag::any);
    if(use_lpgemm > 0) {
        // With LPGEMM, weights tag is set to HWCN
        conv_weights_md = zendnn::memory::desc({weight_dims}, dt::s8, tag::hwcn);
    }
    zendnn::memory::desc conv_bias_md;
    if (bias_exists) {
        auto bias_dims = sp.GetMemory(node.Input(IN_B)).get_desc().dims();
        conv_bias_md = zendnn::memory::desc({bias_dims}, dt::s32, tag::x);
        // If S16 LPGEMM path is taken, Bias datatype is set to S16
        if(use_lpgemm > 0 && s16_lpgemm_enabled) {
              conv_bias_md = zendnn::memory::desc({bias_dims}, dt::s16, tag::x);
        }
    }

    auto dstType = node.Output(OUT_Y).Type();
    if (node.OpType() == "QLinearConv_v1" ||
            node.OpType() == "QConvAdd" ||
            node.OpType() == "QConvAdd_v1" ||
            node.OpType() == "QConvAddRelu" ||
            node.OpType() == "QConvClip") {
        dstType = dt::s8;
    }
    else {
        LOGS_DEFAULT(INFO) <<"Retaining the type as read from model";
    }

    auto conv_dst_md = zendnn::memory::desc({dst_mem_dims}, dstType, tag::any);
    if(use_lpgemm > 0) {
        conv_dst_md = zendnn::memory::desc({dst_mem_dims}, dstType, tag::nhwc);
    }

    if (node.isInplaceMemoryNode) {
        conv_dst_md = _ldst_md;
    }

    zendnn::post_ops conv_post_ops;
    float ops_scale = 1.0, qscale=1.0;

    bool reluFused = 0;
    zendnn::memory dst_zp1_mem, add_out_zp1_mem;
    if (node.OpType() == "QLinearConv_v1") {
        if (dst_zero_points[0] != (int)0) {
            auto dst_zp1_md = sp.GetMemory(node.Input(IN_Y_ZERO_POINT)).get_desc();
            Padd(&dst_zp1_md,
                 conv_dst_md.dims().size());  //Need to provide broadcasted data
            dst_zp1_mem = zendnn::memory({dst_zp1_md.dims(), node.Input(IN_Y_ZERO_POINT).Type(), tag::nhwc},
                                         zendnn_engine);
            dst_zp1_mem.set_data_handle(sp.GetMemory(node.Input(
                                            IN_Y_ZERO_POINT)).get_data_handle());

            /* The two pairs of add() and sub() of same dst_zp may seem redundant,
            * but since we are doing it on uint8 data buffer,
            * we get the saturation/clipping at 255 in between, which is necessary.
            * We can include either one of the below post ops and see the same results coming up.
            * conv_post_ops.append_eltwise(1.0f, zendnn::algorithm::eltwise_clip, 0.0, 255.0f);
            * conv_post_ops.append_eltwise(1.0f, zendnn::algorithm::eltwise_bounded_relu, 255.0, 0.0f);
            */
            conv_post_ops.append_binary(zendnn::algorithm::binary_add,
                                        dst_zp1_mem.get_desc());
            conv_post_ops.append_binary(zendnn::algorithm::binary_sub,
                                        dst_zp1_mem.get_desc());
        }
    }
    else if (node.OpType() == "QConvAdd" || node.OpType() == "QConvAddRelu") {
        if (dst_zero_points[0] != (int)0) {
            auto dst_zp1_md = sp.GetMemory(node.Input(IN_Y_ZERO_POINT)).get_desc();
            Padd(&dst_zp1_md,
                 conv_dst_md.dims().size());  //Need to provide broadcasted data
            dst_zp1_mem = zendnn::memory({dst_zp1_md.dims(), node.Input(IN_Y_ZERO_POINT).Type(), tag::nhwc},
                                         zendnn_engine);
            dst_zp1_mem.set_data_handle(sp.GetMemory(node.Input(
                                            IN_Y_ZERO_POINT)).get_data_handle());

            conv_post_ops.append_binary(zendnn::algorithm::binary_add,
                                        dst_zp1_mem.get_desc());
            conv_post_ops.append_binary(zendnn::algorithm::binary_sub,
                                        dst_zp1_mem.get_desc());
        }
        ops_scale = (float)((sum_scale)/(add_out_scale));
        qscale = (float)((*y_scale)/(add_out_scale));
        conv_post_ops.append_eltwise(1.0, zendnn::algorithm::eltwise_linear, qscale,
                                     0.0);
        conv_post_ops.append_sum(ops_scale);
        if (add_out_zero_points[0] != (int)0) {
            auto add_out_zp1_md = sp.GetMemory(node.Input(IN_ADD_OUT_ZP)).get_desc();
            Padd(&add_out_zp1_md,
                 conv_dst_md.dims().size());  //Need to provide broadcasted data
            add_out_zp1_mem = zendnn::memory({add_out_zp1_md.dims(), node.Input(IN_ADD_OUT_ZP).Type(), tag::nhwc},
                                             zendnn_engine);
            add_out_zp1_mem.set_data_handle(sp.GetMemory(node.Input(
                                                IN_ADD_OUT_ZP)).get_data_handle());
            conv_post_ops.append_binary(zendnn::algorithm::binary_add,
                                        add_out_zp1_mem.get_desc());
        }
        conv_post_ops.append_eltwise(1.0f, zendnn::algorithm::eltwise_bounded_relu,
                                     255.0, 0.0f);
    }
    else if (node.OpType() == "QConvAdd_v1") {
        if (dst_zero_points[0] != (int)0) {
            auto dst_zp1_md = sp.GetMemory(node.Input(IN_Y_ZERO_POINT)).get_desc();
            Padd(&dst_zp1_md,
                 conv_dst_md.dims().size());  //Need to provide broadcasted data
            dst_zp1_mem = zendnn::memory({dst_zp1_md.dims(), node.Input(IN_Y_ZERO_POINT).Type(), tag::nhwc},
                                         zendnn_engine);
            dst_zp1_mem.set_data_handle(sp.GetMemory(node.Input(
                                            IN_Y_ZERO_POINT)).get_data_handle());

            conv_post_ops.append_binary(zendnn::algorithm::binary_add,
                                        dst_zp1_mem.get_desc());
            conv_post_ops.append_binary(zendnn::algorithm::binary_sub,
                                        dst_zp1_mem.get_desc());
        }
        ops_scale = (float)((sum_scale)/(add_out_scale));
        qscale = (float)((*y_scale)/(add_out_scale));
        conv_post_ops.append_eltwise(1.0, zendnn::algorithm::eltwise_linear, qscale,
                                     0.0);
        conv_post_ops.append_sum(ops_scale);
        if (add_out_zero_points[0] != (int)0) {
            auto add_out_zp1_md = sp.GetMemory(node.Input(IN_ADD_OUT_ZP)).get_desc();
            Padd(&add_out_zp1_md,
                 conv_dst_md.dims().size());  //Need to provide broadcasted data
            add_out_zp1_mem = zendnn::memory({add_out_zp1_md.dims(), node.Input(IN_ADD_OUT_ZP).Type(), tag::nhwc},
                                             zendnn_engine);
            add_out_zp1_mem.set_data_handle(sp.GetMemory(node.Input(
                                                IN_ADD_OUT_ZP)).get_data_handle());
            conv_post_ops.append_binary(zendnn::algorithm::binary_add,
                                        add_out_zp1_mem.get_desc());
            conv_post_ops.append_binary(zendnn::algorithm::binary_sub,
                                        add_out_zp1_mem.get_desc());
        }
        conv_post_ops.append_eltwise(1.0f, zendnn::algorithm::eltwise_bounded_relu,
                                     255.0, 0.0f);
    }
    else if (node.OpType() == "QConvRelu") {
        reluFused = 1;
        if (dst_zero_points[0] != (int)0) {
            auto dst_zp1_md = sp.GetMemory(node.Input(IN_Y_ZERO_POINT)).get_desc();
            Padd(&dst_zp1_md,
                 conv_dst_md.dims().size());  //Need to provide broadcasted data
            dst_zp1_mem = zendnn::memory({dst_zp1_md.dims(), node.Input(IN_Y_ZERO_POINT).Type(), tag::nhwc},
                                         zendnn_engine);
            dst_zp1_mem.set_data_handle(sp.GetMemory(node.Input(
                                            IN_Y_ZERO_POINT)).get_data_handle());

            conv_post_ops.append_binary(zendnn::algorithm::binary_add,
                                        dst_zp1_mem.get_desc());
        }
        conv_post_ops.append_eltwise(1.0f, zendnn::algorithm::eltwise_relu, 0.0, 0.0f);
    }
    else if (node.OpType() == "QConvClip") {
        conv_post_ops.append_eltwise(ops_scale, zendnn::algorithm::eltwise_clip, (float)min, (float)max);
    }

    conv_attr.set_post_ops(conv_post_ops);

    zendnn::convolution_forward::primitive_desc conv_pd;
    if (use_lpgemm > 0) {
      LOGS_DEFAULT(ERROR) <<"LPGEMM PATH TAKEN = " << node.Name();

      // LPGEMM path is taken
      // By default, conv descriptor created with u8s8s16os8 API
      auto conv_desc = zendnn::convolution_forward::desc(prop_kind,
                      zendnn::algorithm::convolution_gemm_u8s8s16os8, conv_src_md, conv_weights_md,
                      conv_bias_md, conv_dst_md, strides, padding_left, padding_right, reluFused);
      // s8s8s32os8 API
      if(srcType == dt::s8 && !s16_lpgemm_enabled)
            conv_desc = zendnn::convolution_forward::desc(prop_kind,
                        zendnn::algorithm::convolution_gemm_s8s8s32os8, conv_src_md, conv_weights_md,
                        conv_bias_md, conv_dst_md, strides, padding_left, padding_right, reluFused);
      // s8s8s16os8 API
      else if(srcType == dt::s8 && s16_lpgemm_enabled)
            conv_desc = zendnn::convolution_forward::desc(prop_kind,
                        zendnn::algorithm::convolution_gemm_s8s8s16os8, conv_src_md, conv_weights_md,
                        conv_bias_md, conv_dst_md, strides, padding_left, padding_right, reluFused);
      // u8s8s32os8 API
      else if(srcType == dt::u8 && !s16_lpgemm_enabled)
            conv_desc = zendnn::convolution_forward::desc(prop_kind,
                        zendnn::algorithm::convolution_gemm_u8s8s32os8, conv_src_md, conv_weights_md,
                        conv_bias_md, conv_dst_md, strides, padding_left, padding_right, reluFused);

      conv_pd = zendnn::convolution_forward::primitive_desc(conv_desc, conv_attr, zendnn_engine);
    }
    // Direct path
    else if (bias_exists) {
        auto conv_desc = zendnn::convolution_forward::desc(
                             prop_kind, zendnn::algorithm::convolution_direct,
                             conv_src_md, conv_weights_md, conv_bias_md, conv_dst_md,
                             strides, dilations, padding_left, padding_right);
        conv_pd = zendnn::convolution_forward::primitive_desc(conv_desc, conv_attr,
                  zendnn_engine);
    }
    else {
        auto conv_desc = zendnn::convolution_forward::desc(
                             prop_kind, zendnn::algorithm::convolution_direct,
                             conv_src_md, conv_weights_md, conv_dst_md,
                             strides, dilations, padding_left, padding_right);
        conv_pd = zendnn::convolution_forward::primitive_desc(conv_desc, conv_attr,
                  zendnn_engine);
    }

    // If using GPU this will move the memory from the CPU to the GPU.
    zendnn::memory conv_src_mem;
    if (node.OpType() == "QLinearConv_v2") {
        conv_src_mem = sp.GetMemoryAndReshapeByHandle(node.Input(IN_X),
                       conv_pd.src_desc(), zendnn_engine);
    }
    else {
        conv_src_mem = sp.GetMemoryAndReshape(node.Input(IN_X), conv_pd.src_desc(),
                                              zendnn_engine);
    }

    if (group != 1 && shape == onnxruntime::ort_zendnn::ZendnnQConv::SHAPE_2D) {
        weight_md = zendnn::memory::desc({weight_dims}, node.Input(IN_W).Type(),
                                         zendnn::memory::format_tag::goihw);
        auto mem_from = zendnn::memory(weight_md, zendnn_engine,
                                       sp.GetMemory(node.Input(IN_W)).get_data_handle());
        auto mem_to = zendnn::memory(conv_pd.weights_desc(), zendnn_engine);
        zendnn::stream s{zendnn_engine};
        zendnn::reorder(mem_from, mem_to).execute(s, mem_from, mem_to);
        s.wait();
        conv_weights_mem = mem_to;
    }
    else {
        conv_weights_mem = sp.GetMemoryAndReshape(node.Input(IN_W),
                           conv_pd.weights_desc(), zendnn_engine);
    }
    if (bias_exists && (use_lpgemm == 0 || (use_lpgemm == 1 && s16_lpgemm_enabled == 0))) {
        conv_bias_mem = sp.GetMemoryAndReshape(node.Input(IN_B), conv_pd.bias_desc(),
                                               zendnn_engine);
    }
    // s32 bias converted to s16 bias if S16 LPGEMM path is taken
    // TODO: Support s16 datatype in ZenDNN reorder
    zendnn::memory conv_bias_mem2;
    if(bias_exists && use_lpgemm == 1 && s16_lpgemm_enabled) {
      int32_t * bias_array_original = (int32_t *)(conv_bias_mem.get_data_handle());
      auto bias_dims = sp.GetMemory(node.Input(IN_B)).get_desc().dims();
      int16_t * bias_array2 = new int16_t[bias_dims[0]];
      for(int j=0; j<bias_dims[0]; ++j) {
            bias_array2[j] = static_cast<int16_t>(bias_array_original[j]);
      }
      conv_bias_mem2 = zendnn::memory({{bias_dims}, dt::s16, tag::x}, zendnn_engine, bias_array2);
    }
    auto conv_dst_mem = zendnn::memory(conv_pd.dst_desc(), zendnn_engine);

    // Add the convolution layer to the subgraph
    auto conv_op = zendnn::convolution_forward(conv_pd);

    if (node.OpType() == "QConvAdd" ||
            node.OpType() == "QConvAdd_v1" ||
            node.OpType() == "QConvAddRelu") {
        /*
        Conv_1_Y.mem = ops_scale*(Conv_1_Y.mem + Conv2(X, W, B).mem)
        conv_dst_mem = binary_post_op_mem(viz. Conv_1_Y.mem)
        */
        zendnn::memory::desc binary_mem_desc;
        auto binary_post_op_mem = sp.GetMemory(node.Input(IN_BINARY).Name());
        conv_dst_mem = binary_post_op_mem;
    }
    else {
        if (node.isInplaceMemoryNode) {
            conv_dst_mem = _dst_mem;
        }
        else {
            conv_dst_mem = zendnn::memory(conv_pd.dst_desc(), zendnn_engine);
        }
    }

    std::unordered_map<int, zendnn::memory> qconv_args;
    qconv_args.insert({ZENDNN_ARG_SRC, conv_src_mem});
    qconv_args.insert({ZENDNN_ARG_WEIGHTS, conv_weights_mem});
    if (bias_exists && ((use_lpgemm == 1 && s16_lpgemm_enabled == 0) || use_lpgemm == 0)) {
        qconv_args.insert({ZENDNN_ARG_BIAS, conv_bias_mem});
    }
    else if (bias_exists && use_lpgemm == 1 && s16_lpgemm_enabled == 1) {
        qconv_args.insert({ZENDNN_ARG_BIAS, conv_bias_mem2});
    }

    if (node.OpType() == "QLinearConv_v1") {
        if (dst_zero_points[0] != (int)0) {
            qconv_args.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1, dst_zp1_mem});
            qconv_args.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1, dst_zp1_mem});
        }
    }
    else if (node.OpType() == "QConvAdd" || node.OpType() == "QConvAddRelu") {
        auto add_post_op_pos = 2;
        if (dst_zero_points[0] != (int)0) {
            qconv_args.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1, dst_zp1_mem});
            qconv_args.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1, dst_zp1_mem});
            add_post_op_pos += 2;
        }
        if (add_out_zero_points[0] != (int)0) {
            qconv_args.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(add_post_op_pos) | ZENDNN_ARG_SRC_1, add_out_zp1_mem});
        }
    }
    else if (node.OpType() == "QConvAdd_v1") {
        auto add_post_op_pos = 2;
        if (dst_zero_points[0] != (int)0) {
            qconv_args.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1, dst_zp1_mem});
            qconv_args.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(1) | ZENDNN_ARG_SRC_1, dst_zp1_mem});
            add_post_op_pos += 2;
        }
        if (add_out_zero_points[0] != (int)0) {
            qconv_args.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(add_post_op_pos) | ZENDNN_ARG_SRC_1, add_out_zp1_mem});
            qconv_args.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(add_post_op_pos + 1) | ZENDNN_ARG_SRC_1, add_out_zp1_mem});
        }
    }
    else if (node.OpType() == "QConvRelu") {
        if (dst_zero_points[0] != (int)0) {
            qconv_args.insert({ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1, dst_zp1_mem});
        }
    }

    qconv_args.insert({ZENDNN_ARG_DST, conv_dst_mem});

    sp.AddPrimitive(conv_op, qconv_args);

    if (add_output) {
        sp.SetMemory(node.Output(OUT_Y), conv_dst_mem);
    }
}

void ZendnnQConv::Padd(zendnn::memory::desc *target_md, size_t pad) {
    // Pads an input to broadcast the op correctly
    auto target_dims = target_md->dims();   // Add back padd
    while (target_dims.size() < pad) {
        target_dims.insert(target_dims.end(), 1);
    }
    *target_md = target_md->reshape(target_dims);
}

void ZendnnQConv::SetDestinationMemoryInfo(zendnn::memory::desc &ldst_md,
        zendnn::memory &dst_mem, bool addOutput) {
    _ldst_md = ldst_md;
    _dst_mem = dst_mem;
    add_output = addOutput;
}

std::vector<int64_t> ZendnnQConv::GetInferedPads(ZendnnNode &node,
        const zendnn::memory::dims &src_dims,
        const zendnn::memory::dims &dilations,
        const std::vector<int64_t> &kernel_shape,
        const zendnn::memory::dims &strides) {
    AutoPadType auto_pad = GetAutoPad(node);
    ConvShape shape = static_cast<ConvShape>(kernel_shape.size());
    std::vector<int64_t> pads;
    if (auto_pad == AutoPadType::NOTSET) {
        pads = GetPads(node);
        if (pads.empty()) {
            // 'shape * 2' because we want the pad at the start and end of each dim.
            pads.resize(shape * 2, 0);
        }
        return pads;
    }

    pads.resize(shape * 2, 0);

    int64_t pad_head = 0;
    int64_t pad_tail = 0;
    assert(src_dims.size() == shape + 2);
    for (size_t i = 0; i < shape; ++i) {
        if (ComputePad(src_dims[2 + i], strides[i], kernel_shape[i], (dilations[i] + 1),
                       auto_pad, pad_head, pad_tail)) {
            pads[i] = pad_head;
            pads[shape + i] = pad_tail;
        }
    }
    return pads;
}

zendnn::memory::dims ZendnnQConv::GetPaddingLeft(const std::vector<int64_t>
        &onnx_padding, ConvShape shape) {
    assert(onnx_padding.size() == shape * 2);
    zendnn::memory::dims padding_left;
    padding_left.assign(onnx_padding.begin(), onnx_padding.begin() + shape);
    return padding_left;
}

zendnn::memory::dims ZendnnQConv::GetPaddingRight(const std::vector<int64_t>
        &onnx_padding, ConvShape shape) {
    assert(onnx_padding.size() == shape * 2);
    zendnn::memory::dims padding_right;
    padding_right.assign(onnx_padding.begin() + shape, onnx_padding.end());
    return padding_right;
}

AutoPadType ZendnnQConv::GetAutoPad(ZendnnNode &node) {
    std::string auto_pad;
    auto attr = node.Attributes().find("auto_pad");
    if (attr != node.Attributes().end() &&
            attr->second().type() ==
            ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
        auto_pad = attr->second().s();
    }
    return ((auto_pad != "") ? StringToAutoPadType(auto_pad) : AutoPadType::NOTSET);
}

zendnn::memory::dims ZendnnQConv::GetDilations(ZendnnNode &node,
        ConvShape shape) {
    auto attr = node.Attributes().find("dilations");
    std::vector<int64_t> dilations;
    if (attr != node.Attributes().end()) {
        dilations.reserve(attr->second().ints_size());
        for (int i = 0; i < attr->second().ints_size(); ++i) {
            // ZenDNN dilations are always one less than Onnx dilations
            dilations.push_back(attr->second().ints(i) - 1);
        }
    }
    else {
        dilations.resize(shape, 0);
    }
    return zendnn::memory::dims(dilations.begin(), dilations.end());
}
int64_t ZendnnQConv::GetGroup(ZendnnNode &node) {
    auto attr = node.Attributes().find("group");
    if (attr != node.Attributes().end()) {
        return attr->second().i();
    }
    return 1;
}

float ZendnnQConv::GetAlpha(ZendnnNode &node) {
    auto attr = node.Attributes().find("alpha");
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return 1.0f;
}

float ZendnnQConv::GetMin(ZendnnNode &node, float default_min) {
    auto attr = node.Attributes().find("min");
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return default_min;
}

float ZendnnQConv::GetMax(ZendnnNode &node, float default_max) {
    auto attr = node.Attributes().find("max");
    if (attr != node.Attributes().end()) {
        return attr->second().f();
    }
    return default_max;
}

std::vector<int64_t> ZendnnQConv::GetKernelShape(ZendnnNode &node) {
    auto attr = node.Attributes().find("kernel_shape");
    std::vector<int64_t> kernel_shape;
    if (attr != node.Attributes().end()) {
        kernel_shape.reserve(attr->second().ints_size());
        for (int i = 0; i < attr->second().ints_size(); ++i) {
            kernel_shape.push_back(attr->second().ints(i));
        }
        return kernel_shape;
    }
    // Infer the Kernel shape from the input weights
    auto weight_dims = node.Input(IN_W).Dim();
    kernel_shape = std::vector<int64_t>(weight_dims.begin() + 2, weight_dims.end());
    return kernel_shape;
}

std::vector<int64_t> ZendnnQConv::GetPads(ZendnnNode &node) {
    auto attr = node.Attributes().find("pads");
    if (attr != node.Attributes().end()) {
        std::vector<int64_t> pads;
        pads.reserve(attr->second().ints_size());
        for (int i = 0; i < attr->second().ints_size(); ++i) {
            pads.push_back(attr->second().ints(i));
        }
        return pads;
    }
    return {};
}

zendnn::memory::dims ZendnnQConv::GetStrides(ZendnnNode &node,
        ConvShape shape) {
    if (node.GetStridesOpt() == 1) {
        return std::vector<int64_t> {1, 1};
    }
    else if (node.GetStridesOpt() == 2) {
        return std::vector<int64_t> {2, 2};
    }
    auto attr = node.Attributes().find("strides");
    std::vector<int64_t> strides;
    if (attr != node.Attributes().end()) {
        strides.reserve(attr->second().ints_size());
        for (int i = 0; i < attr->second().ints_size(); ++i) {
            strides.push_back(attr->second().ints(i));
        }
    }
    else {
        strides.resize(shape, 1);
    }
    return zendnn::memory::dims(strides.begin(), strides.end());
}

// ComputePad is copy/paste of a the ComputePad found in core/providers/common.h
// With some minor modifications.
// ComputePad is not exposed to the shared library so this copy is used instead.
bool ZendnnQConv::ComputePad(const int64_t in_dim,
                             const int64_t stride,
                             const int64_t kernel,
                             const int64_t dilation,
                             AutoPadType pad_type,
                             int64_t &pad_head, /* output param */
                             int64_t &pad_tail, /* output param */
                             bool force_symmetric_auto_padding /*= false*/) {
    pad_head = 0;
    pad_tail = 0;
    switch (pad_type) {
    case AutoPadType::NOTSET:
        break;
    case AutoPadType::VALID:
        break;
    case AutoPadType::SAME_UPPER:
    //[[fallthrough]] //fallthrough attribute requires C++17
    case AutoPadType::SAME_LOWER: {
        if (1 != dilation) {
            LOGS_DEFAULT(ERROR) <<
                                "Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.";
            return false;
        }

        // The ONNX spec says if `auto_pad` attribute is set, pad until the `legacy_target_size`
        // is `ceil (in_dim / stride)`. The following line of code is essentially just that and
        // is retained as is
        int64_t legacy_target_size = (in_dim + stride - 1) / stride;
        int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_dim;
        // make sure padding is symmetric
        if (force_symmetric_auto_padding) {
            // Inlining math::roundUpPow2() from util/math.h to avoid bringing in the transitive dependencies.
            pad_needed = (pad_needed + 1) & ~1;
        }

        if (pad_type == AutoPadType::SAME_LOWER) {
            pad_head = (pad_needed + 1) / 2;
        }
        else {
            pad_head = pad_needed / 2;
        }
        pad_tail = pad_needed - pad_head;
    }
    break;
    default:
        LOGS_DEFAULT(ERROR) << "ComputePad: pad_type attribute not supported.";
        return false;
    }
    return true;
}

zendnn::memory::dims ZendnnQConv::InferOutputShape(ZendnnNode &node,
        const zendnn::memory::dims &src_dims,
        const zendnn::memory::dims &weight_dims,
        const std::vector<int64_t> &kernel_shape,
        const zendnn::memory::dims &strides,
        const zendnn::memory::dims &dilations,
        const std::vector<int64_t> &pads) {
    auto pad_type = GetAutoPad(node);
    ConvShape shape = static_cast<ConvShape>(kernel_shape.size());
    zendnn::memory::dims output_shape;

    output_shape.push_back(src_dims[0]);
    output_shape.push_back(weight_dims[0]);
    for (size_t dim = 0; dim < shape; ++dim) {
        if (dim >= strides.size() || dim >= kernel_shape.size() ||
                dim >= dilations.size() || dim >= pads.size() ||
                shape + dim >= pads.size()) {
            LOGS_DEFAULT(ERROR) << "Out of bound access to array";
            return {};
        }
        int64_t dkernel = (dilations[dim] + 1) * (kernel_shape[dim] - 1) + 1;
        switch (pad_type) {
        case onnxruntime::AutoPadType::NOTSET: {
            output_shape.push_back(static_cast<int64_t>(static_cast<float>
                                   (src_dims[dim + 2] + pads[dim] + pads[dim + shape] - dkernel) / strides[dim] +
                                   1));
        }
        break;
        case onnxruntime::AutoPadType::VALID: {
            output_shape.push_back((src_dims[dim + 2] - dkernel) / strides[dim] + 1);
        }
        break;
        case onnxruntime::AutoPadType::SAME_UPPER: {
            if (dilations[dim] != 0) {
                LOGS_DEFAULT(ERROR) <<
                                    "Dilation not supported for AutoPadType::SAME_UPPER or AutoPadType::SAME_LOWER.";
                return {};
            }
            int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) /
                                         strides[dim];
            int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim]
                                 - src_dims[dim + 2];
            output_shape.push_back((src_dims[dim + 2] + pad_needed - dkernel) / strides[dim]
                                   + 1);
        }
        break;
        case onnxruntime::AutoPadType::SAME_LOWER: {
            int64_t legacy_target_size = (src_dims[dim + 2] + strides[dim] - 1) /
                                         strides[dim];
            int64_t pad_needed = (legacy_target_size - 1) * strides[dim] + kernel_shape[dim]
                                 - src_dims[dim + 2];
            output_shape.push_back((src_dims[dim + 2] + pad_needed - dkernel) / strides[dim]
                                   + 1);
        }
        break;
        default:
            break;
        }
    }
    return output_shape;
}

}  // namespace ort_zendnn
}  // namespace onnxruntime
