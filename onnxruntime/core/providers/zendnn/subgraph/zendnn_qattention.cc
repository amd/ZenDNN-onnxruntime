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

#include "zendnn_qattention.h"
#include "zendnn_subgraph.h"
#include "zendnn_subgraph_primitive.h"
#include <cmath>

namespace onnxruntime {
namespace ort_zendnn {

ZendnnQAttention::ZendnnQAttention() {}

//compute a total scale memory from input and weight scale
zendnn::memory ZendnnQAttention::ComputeTotalScale(ZendnnSubgraphPrimitive &sp,
        ZendnnNode &node) {
    auto eng = sp.GetEngine();
    bool has_input_scale = node.Input(INPUT_SCALE).Exists();
    bool has_weights_scale = node.Input(WEIGHTS_SCALE).Exists();
    auto input_scale_mem = has_input_scale ? sp.GetMemory(node.Input(
                               INPUT_SCALE)) : zendnn::memory();
    auto weights_scale_mem = has_weights_scale ? sp.GetMemory(node.Input(
                                 WEIGHTS_SCALE)) : zendnn::memory();

    if (input_scale_mem && weights_scale_mem) {
        //force descriptor to be 1 dim, will fail if product of dims not equal 1
        auto src_0_md = input_scale_mem.get_desc().reshape({1});
        auto src_1_md = weights_scale_mem.get_desc().reshape({1});
        auto dst_md = src_1_md;
        auto binary_d = zendnn::binary::desc(zendnn::algorithm::binary_mul, src_0_md,
                                             src_1_md, dst_md);
        auto binary_pd = zendnn::binary::primitive_desc(binary_d, eng);

        auto binary_src0_mem = sp.GetMemoryAndReshape(node.Input(INPUT_SCALE),
                               binary_pd.src0_desc(), eng);
        auto binary_src1_mem = sp.GetMemoryAndReshape(node.Input(WEIGHTS_SCALE),
                               binary_pd.src1_desc(), eng);
        auto binary_dst_mem = zendnn::memory(binary_pd.dst_desc(), eng);
        auto binary_prim = zendnn::binary(binary_pd);
        sp.AddPrimitive(binary_prim, {{ZENDNN_ARG_SRC_0, binary_src0_mem},
            {ZENDNN_ARG_SRC_1, binary_src1_mem},
            {ZENDNN_ARG_DST, binary_dst_mem}
        });
        return binary_dst_mem;
    }
    else if (input_scale_mem) {
        return sp.GetMemoryAndReshape(node.Input(INPUT_SCALE),
                                      input_scale_mem.get_desc().reshape({1}), eng);
    }
    else if (weights_scale_mem) {
        return sp.GetMemoryAndReshape(node.Input(WEIGHTS_SCALE),
                                      weights_scale_mem.get_desc().reshape({1}), eng);
    }
    else {
        return zendnn::memory();
    }
}

/*
   input_tensor            weight_tensor
         \                       /
          \                     /
           \                   /
            \                 /
               matmulinteger
        with input and weight zero point,
        input and weight scale and bias
                    |
                    |
                    | QKV
                    |
                  slice
                /   |   \
               /    |    \
              /     |     \
             /      |      \
            |Q      |K      |V
            |       |       |
         reshape  reshape  reshape
            |       |       |
         permute  permute  permute
            |       |       |
            |    transpose  |
             \      |       |
              \     |       |
               \    |       |
                \   |       |
                  matmul    |
                    |       |
                    |       |
   sqrt(head_dim)   |       |
                \   |       |
                 \  |       |
                  \ |       |
                   div      |
                    |       |
                  (mask)   /
                    |     /
                 softmax /
                    |   /
                  matmul
                    |
                  permute
                    |
                  reshape
                    |
                  output
*/
/*
limitations
  scalar input zp
  scalar weight zp
  scalar input scale
  scalar weight scale
  2D raw mask
  no past and present input
*/
void ZendnnQAttention::CreatePrimitive(ZendnnSubgraphPrimitive &sp,
                                       ZendnnNode &node) {
    auto eng = sp.GetEngine();

    zendnn::memory QKV_mem;
    {
        //prepare zero points for int8 matmul primitive
        zendnn::primitive_attr matmul_attr;
        bool has_input_zero_point = node.Input(INPUT_ZP).Exists();
        bool has_weights_zero_point = node.Input(WEIGHTS_ZP).Exists();

        // (input-input_zero_point)*(weight-weight_zero_point)
        {
            //set input zp
            if (has_input_zero_point) {
                matmul_attr.set_zero_points(ZENDNN_ARG_SRC, 0, {ZENDNN_RUNTIME_S32_VAL});
            }

            //set weight zp
            if (has_weights_zero_point) {
                matmul_attr.set_zero_points(ZENDNN_ARG_WEIGHTS, 0, {ZENDNN_RUNTIME_S32_VAL});
            }
        }

        zendnn::memory::desc input_md;
        zendnn::memory::desc weights_md;

        {
            auto input_md_ori = sp.GetMemory(node.Input(INPUT)).get_desc();
            auto weights_md_ori = sp.GetMemory(node.Input(WEIGHTS)).get_desc();

            auto weights_dims = weights_md_ori.dims();
            weights_dims.insert(weights_dims.begin(), 1);

            input_md = zendnn::memory::desc(input_md_ori.dims(), input_md_ori.data_type(),
                                            zendnn::memory::format_tag::any);
            weights_md = zendnn::memory::desc(weights_dims, weights_md_ori.data_type(),
                                              zendnn::memory::format_tag::any);
        }

        zendnn::memory::desc QKV_md;
        {
            //the output of int8 matmul is always 3 dims and consists of Q,K,V values
            auto QKV_dims = input_md.dims();
            QKV_dims[2] = weights_md.dims()[2];
            //use format any for optimization
            QKV_md = zendnn::memory::desc(QKV_dims, zendnn::memory::data_type::f32,
                                          zendnn::memory::format_tag::any);
        }

        auto matmul_d = zendnn::matmul::desc(input_md, weights_md, QKV_md);
        auto matmul_pd = zendnn::matmul::primitive_desc(matmul_d, matmul_attr, eng);
        // (input-input_zero_point)*(weight-weight_zero_point)
        auto matmul_prim = zendnn::matmul(matmul_pd);

        auto matmul_src_mem = sp.GetMemoryAndReshape(node.Input(INPUT),
                              matmul_pd.src_desc(), eng);
        auto matmul_weights_mem = sp.GetMemoryAndReshape(node.Input(WEIGHTS),
                                  matmul_pd.weights_desc(), eng);
        QKV_mem = zendnn::memory(matmul_pd.dst_desc(), eng);

        std::unordered_map<int, zendnn::memory> mem_map({{ZENDNN_ARG_SRC, matmul_src_mem},
            {ZENDNN_ARG_WEIGHTS, matmul_weights_mem},
            {ZENDNN_ARG_DST, QKV_mem}});

        if (has_input_zero_point) {
            auto zp_mem_desc = zendnn::memory::desc({1}, zendnn::memory::data_type::s32, {1});
            auto &tensor = node.Input(INPUT_ZP);
            auto zp_mem = sp.GetMemoryAndReshape(tensor, zp_mem_desc, eng);
            mem_map[ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_SRC] = zp_mem;
        }

        if (has_weights_zero_point) {
            auto zp_mem_desc = zendnn::memory::desc({1}, zendnn::memory::data_type::s32, {1});
            auto &tensor = node.Input(WEIGHTS_ZP);
            auto zp_mem = sp.GetMemoryAndReshape(tensor, zp_mem_desc, eng);
            mem_map[ZENDNN_ARG_ATTR_ZERO_POINTS | ZENDNN_ARG_WEIGHTS] = zp_mem;
        }

        //(input-input_zero_point)*(weight-weight_zero_point)
        sp.AddPrimitive(matmul_prim, mem_map);
    }

    //in place binary add with source0 scaling
    {
        //compute a total scale from input scale and weight scale
        //i_scale * w_scale if both exist
        auto total_scale_mem = ComputeTotalScale(sp, node);

        auto bias_md = sp.GetMemory(node.Input(BIAS)).get_desc();
        bias_md = bias_md.reshape({1, 1, bias_md.dims()[0]});
        auto QKV_desc = QKV_mem.get_desc();

        //always broadcast from bias to QKV
        auto binary_d = zendnn::binary::desc(zendnn::algorithm::binary_add, QKV_desc,
                                             bias_md, QKV_desc);

        zendnn::primitive_attr binary_attr;
        //scale source 0, matmul output
        if (total_scale_mem) {
            binary_attr.set_scales(ZENDNN_ARG_SRC_0, 0, {ZENDNN_RUNTIME_F32_VAL});
        }

        auto binary_pd = zendnn::binary::primitive_desc(binary_d, binary_attr, eng);
        auto binary_prim = zendnn::binary(binary_pd);

        auto bias_mem = sp.GetMemoryAndReshape(node.Input(BIAS), binary_pd.src1_desc(),
                                               eng);

        std::unordered_map<int, zendnn::memory> binary_mem_map({{ZENDNN_ARG_SRC_0, QKV_mem},
            {ZENDNN_ARG_SRC_1, bias_mem},
            {ZENDNN_ARG_DST, QKV_mem}});

        if (total_scale_mem) {
            binary_mem_map[ZENDNN_ARG_ATTR_INPUT_SCALES | ZENDNN_ARG_SRC_0] =
                total_scale_mem;
        }

        sp.AddPrimitive(binary_prim, binary_mem_map);
    }

    //parse some dim information for permute and reshape
    //eg, 8,512,2034 = 8,512,(3,12,64)
    auto batch_size = QKV_mem.get_desc().dims()[0];
    auto sequence_length = QKV_mem.get_desc().dims()[1];
    auto num_heads = GetNumHeads(node);
    auto hidden_size = QKV_mem.get_desc().dims()[2] / 3;
    auto head_size = hidden_size / num_heads;

    //slice QKV into submemories
    auto Q_md = QKV_mem.get_desc().submemory_desc({batch_size, sequence_length, hidden_size}, {0, 0, 0});
    auto K_md = QKV_mem.get_desc().submemory_desc({batch_size, sequence_length, hidden_size}, {0, 0, hidden_size});
    auto V_md = QKV_mem.get_desc().submemory_desc({batch_size, sequence_length, hidden_size}, {0, 0, hidden_size * 2});

    //split QKV last dim to num_heads, hidden_dim
    Q_md = Q_md.reshape({batch_size, sequence_length, num_heads, head_size});
    K_md = K_md.reshape({batch_size, sequence_length, num_heads, head_size});
    V_md = V_md.reshape({batch_size, sequence_length, num_heads, head_size});

    //permute K and QV
    Q_md = Q_md.permute_axes({0, 2, 1, 3});
    //K is different as it needs to be tranposed
    K_md = K_md.permute_axes({0, 3, 1, 2});
    V_md = V_md.permute_axes({0, 2, 1, 3});

    bool has_mask_index = node.Input(MASK_INDEX).Exists();
    auto mask_index_mem =
        zendnn::memory();  // mask_index will reside in this memory

    //prepare mask for calculating attention probs
    //mask size has to be 2D batch_size, max_sequence_length
    //if tensor has values 50,60,70,80 and the mask is 1,1,0,0
    //mask will be converted to 0,0,-10000,-10000 (10000*x-10000)
    //the resulted "masked" tensor will be 50,60,-9930,-9920 (softmax will then evalute large magnitude negative number to be 0)
    //need a reorder of data type from s32 to f32 to let mask to have the same data type as QK result
    if (has_mask_index) {
        auto mask_index_mem_desc = sp.GetMemory(node.Input(MASK_INDEX)).get_desc();

        auto linear_d = zendnn::eltwise_forward::desc(
                            zendnn::prop_kind::forward_inference, zendnn::algorithm::eltwise_linear,
                            mask_index_mem_desc, 10000.0f, -10000.0f);
        auto linear_pd = zendnn::eltwise_forward::primitive_desc(linear_d, eng);

        auto mask_index_ori_mem = sp.GetMemoryAndReshape(node.Input(MASK_INDEX),
                                  linear_pd.src_desc(), eng);
        assert(linear_pd.dst_desc().data_type() == zendnn::memory::data_type::s32);
        auto mask_index_mem_unbroadcasted = zendnn::memory(linear_pd.dst_desc(), eng);

        auto linear_prim = zendnn::eltwise_forward(linear_pd);
        //mask = 10000*mask-10000
        sp.AddPrimitive(linear_prim, {{ZENDNN_ARG_SRC, mask_index_ori_mem}, {ZENDNN_ARG_DST, mask_index_mem_unbroadcasted}});

        zendnn::memory mask_index_mem_unbroadcasted_f32;
        {
            auto md = mask_index_mem_unbroadcasted.get_desc();
            auto dims = md.dims();
            auto strides = md.data.format_desc.blocking.strides;
            zendnn::memory::dims strides_vec;
            for (size_t i = 0; i < dims.size(); i++) {
                strides_vec.push_back(strides[i]);
            }
            auto mask_index_md_unbroadcasted_f32 = zendnn::memory::desc(dims,
                                                   zendnn::memory::data_type::f32, strides_vec);
            mask_index_mem_unbroadcasted_f32 = zendnn::memory(
                                                   mask_index_md_unbroadcasted_f32, eng);
            sp.AddPrimitive(zendnn::reorder(mask_index_mem_unbroadcasted,
            mask_index_mem_unbroadcasted_f32), {
                {ZENDNN_ARG_FROM, mask_index_mem_unbroadcasted},
                {ZENDNN_ARG_TO, mask_index_mem_unbroadcasted_f32}
            });
        }

        //unsqueeze the mem for broadcasting
        auto mask_index_dims = mask_index_mem_unbroadcasted_f32.get_desc().dims();
        //not symetric, simply broadcasting
        //eg 8,512 -> 8,1,1,512
        //eg 8,1,1,512 -> 8,12,512,512
        mask_index_dims.insert(mask_index_dims.begin() + 1, 1);
        mask_index_dims.insert(mask_index_dims.begin() + 2, 1);
        auto mask_index_broadcasted_md =
            mask_index_mem_unbroadcasted_f32.get_desc().reshape(mask_index_dims);
        //set mask_index_mem
        mask_index_mem = zendnn::memory(mask_index_broadcasted_md, eng, nullptr);
        zendnn::stream s(eng);
        mask_index_mem.set_data_handle(
            mask_index_mem_unbroadcasted_f32.get_data_handle(), s);
        s.wait();
    }

    zendnn::memory QK_mem;
    //matmul Q and K (transpose) with mask as binary post op followed by binary div (normalization) and in place softmax.
    //softmax((Q * K(T) + MASK ) / sqrt(head_dim))
    {
        zendnn::primitive_attr QK_attr;
        {
            auto scales = std::vector<float>({float(1 / std::sqrt(head_size))});
            QK_attr.set_output_scales(0, scales);

            if (mask_index_mem) {
                zendnn::post_ops add_bias;
                add_bias.append_binary(zendnn::algorithm::binary_add,
                                       mask_index_mem.get_desc());
                QK_attr.set_post_ops(add_bias);
            }
        }

        auto QK_md = zendnn::memory::desc({batch_size, num_heads, sequence_length, sequence_length},
                                          zendnn::memory::data_type::f32, zendnn::memory::format_tag::any);
        auto QK_d = zendnn::matmul::desc(Q_md, K_md, QK_md);
        auto QK_pd = zendnn::matmul::primitive_desc(QK_d, QK_attr, eng);
        auto QK_prim = zendnn::matmul(QK_pd);

        QK_mem = zendnn::memory(QK_pd.dst_desc(), eng);
        {
            //QKV_mem is used as both input and weight but since matmul is defined on submemory, computation will be applied to correct submemory
            std::unordered_map<int, zendnn::memory> QK_mem_map({{ZENDNN_ARG_SRC, QKV_mem},
                {ZENDNN_ARG_WEIGHTS, QKV_mem},
                {ZENDNN_ARG_DST, QK_mem}});
            if (mask_index_mem) {
                QK_mem_map[ZENDNN_ARG_ATTR_MULTIPLE_POST_OP(0) | ZENDNN_ARG_SRC_1] =
                    mask_index_mem;
            }
            sp.AddPrimitive(QK_prim, QK_mem_map);
        }

        //apply softmax in place to produce attention prob
        {
            auto softmax_desc = zendnn::softmax_forward::desc(
                                    zendnn::prop_kind::forward_inference, QK_mem.get_desc(), 3);
            auto softmax_pd = zendnn::softmax_forward::primitive_desc(softmax_desc, eng);
            auto softmax_prim = zendnn::softmax_forward::primitive(softmax_pd);

            //QK = softmax(QK) in place
            sp.AddPrimitive(softmax_prim, {{ZENDNN_ARG_SRC, QK_mem}, {ZENDNN_ARG_DST, QK_mem}});
        }
    }

    //matmul attention_probs with V to produce the final attended result
    zendnn::memory QAttention_dst_mem;
    {
        //format acbd in order to work with subsequent permute and merge to produce ort format memory
        auto QAttention_dst_md = zendnn::memory::desc({batch_size, num_heads, sequence_length, head_size},
                                 zendnn::memory::data_type::f32, zendnn::memory::format_tag::acbd);

        auto Prob_V_d = zendnn::matmul::desc(QK_mem.get_desc(), V_md,
                                             QAttention_dst_md);
        auto Prob_V_pd = zendnn::matmul::primitive_desc(Prob_V_d, eng);
        auto Prob_V_prim = zendnn::matmul(Prob_V_pd);

        QAttention_dst_mem = zendnn::memory(Prob_V_pd.dst_desc(), eng);
        std::unordered_map<int, zendnn::memory> Prob_V_mem_map({{ZENDNN_ARG_SRC, QK_mem},
            {ZENDNN_ARG_WEIGHTS, QKV_mem},
            {ZENDNN_ARG_DST, QAttention_dst_mem}});
        //prob * V
        sp.AddPrimitive(Prob_V_prim, Prob_V_mem_map);
    }

    //permute and merge axes through reshape
    {
        auto QAttention_dst_md_BNSH = QAttention_dst_mem.get_desc();
        //swap axes
        auto QAttention_dst_md_BSNH = QAttention_dst_md_BNSH.permute_axes({0, 2, 1, 3});
        //merge axes
        auto QAttention_dst_md_BSH = QAttention_dst_md_BSNH.reshape({batch_size, sequence_length, hidden_size});
        auto QAttention_dst_mem_correct_shape = zendnn::memory(QAttention_dst_md_BSH,
                                                eng, nullptr);
        sp.AddReshape(QAttention_dst_mem, QAttention_dst_mem_correct_shape);
        //needs to copy if it outputs for subgraph
        sp.SetMemory(node.Output(OUTPUT), QAttention_dst_mem_correct_shape, true);
    }
}


//obtain the number of heads for qattention node
zendnn::memory::dim ZendnnQAttention::GetNumHeads(ZendnnNode &node) {
    auto attr = node.Attributes().find("num_heads");
    if (attr != node.Attributes().end()) {
        return attr->second().i();
    }
    else {
        //num_heads should always exists as an attribute in qattention
        ORT_THROW("NUM_HEADS NOT EXIST");
    }
}


}  // namespace ort_zendnn
}  // namespace onnxruntime
