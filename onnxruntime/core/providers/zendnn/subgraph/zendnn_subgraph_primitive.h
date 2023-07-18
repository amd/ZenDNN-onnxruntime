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
#include "zendnn.hpp"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
namespace ort_zendnn {

//borrow from coreml ep's data structures to organize data handle, shape and data type
struct OnnxTensorInfo {
    const int32_t data_type;  // Uses TensorProto::DataType
    const std::vector<int64_t> shape;
};

struct OnnxTensorData {
    OnnxTensorInfo tensor_info;
    void *buffer{nullptr};
};

struct PrimitiveMemInfo {
    bool is_dynamic = false;
    int variable_inputs = 0;
    int ref_count;
    zendnn::memory::desc mem_desc;
};

class ZendnnSubgraphPrimitive {
  public:
    ZendnnSubgraphPrimitive(ort_zendnn::ZendnnSubgraph &zendnn_subgraph,
                            AllocatorPtr alloc_ptr);
    ~ZendnnSubgraphPrimitive() = default;

    //compile subgraph primitive with runtime input information
    void Compile(const std::unordered_map<std::string, OnnxTensorData> &inputs);
    void AddInitializers();
    void AddOutputs();
    void AddKernels();

    //run inference
    onnxruntime::common::Status Predict(const
                                        std::unordered_map<std::string, OnnxTensorData> &inputs,
                                        const std::unordered_map<std::string, OnnxTensorData> &outputs);

    void SetOrderedInputs(std::vector<std::string> &&inputs);
    void SetOrderedOutputs(std::vector<std::string> &&outputs);
    const std::vector<std::string> &GetOrderedInputs() const;
    const std::vector<std::string> &GetOrderedOutputs() const;

    //get corresponding ZENDNN format from dim size in onnxruntime
    zendnn::memory::format_tag GetZendnnFormat(size_t dim_size);
    zendnn::engine GetCPUEngine();
    zendnn::engine GetEngine();
    zendnn::stream GetStream();

    //obtain a zendnn::memory with specified name, memory descriptor and engine, will perform extra reorder/reshape if necessary before returning
    zendnn::memory GetMemoryAndReshape(const ZendnnTensor &tensor,
                                       zendnn::memory::desc mem_desc, zendnn::engine eng, bool transpose = false);
    /* This API is part of reorder optimization. s8(-128, 127)->relu = u8(0, 127)*/
    zendnn::memory GetMemoryAndReshapeByHandle(const ZendnnTensor &tensor,
            zendnn::memory::desc mem_desc, zendnn::engine eng, bool transpose = false);
    //add zendnn primitive and memory map to subgraph primitive
    //when you add primitive, you can optionally specify a vector of indexes to be printed in runtime for debug purpose
    //eg, sp.AddPrimitve(prim,mem_map,{ZENDNN_ARG_SRC})
    void AddPrimitive(zendnn::primitive prim,
                      std::unordered_map<int, zendnn::memory> mem_map,
                      PrimitiveMemInfo mem_info = {false, 0, 0, zendnn::memory::desc()},
                      std::vector<int> items_to_print = {});
    //add a reshape (e.g. squeeze, unsqueeze) to subgraph primitive
    void AddReshape(zendnn::memory src, zendnn::memory dst);
    bool HasMemory(std::string memory_name, zendnn::memory::desc mem_desc,
                   zendnn::engine eng);
    zendnn::memory GetMemory(const ZendnnTensor &tensor);
    zendnn::memory GetMemory(const ZendnnTensor &tensor,
                             zendnn::memory::desc mem_desc, zendnn::engine eng);
    //set memory to a tensor (output)
    //if always_copy_output is true a copy of the memory will be made when the output is leaving the subgraph.
    //is_scalar is true to indicate a scalar output in order to allocate the correct onnxruntime output buffer
    void SetMemory(const ZendnnTensor &tensor, zendnn::memory mem,
                   bool always_copy_output = false, bool is_scalar = false);
    void SetMemory(std::string memory_name, zendnn::memory mem);
    void SetInitializer(std::string memory_name, zendnn::memory mem);
    zendnn::memory::desc GetOutputInfo(std::string name);
    bool IsScalarOutput(const std::string &name);
    bool IsDynamic();
    // All Scalar inputs are automatically converterted to a one dimentional tensor when used in ZenDNN
    // If the input being a scalar affects the operator this function can be used to determine if the
    // original input from ORT was a scalar.
    bool IsScalar(const ZendnnTensor &tensor);
    OrtMutex &GetMutex() {
        return mutex_;
    }

    //GetMemory in OrtFormat if the memory is not in the OrtFormat this will reorder the memory.
    //All memory will be moved to the zendnn_engine even if it is already in OrtFormat.
    zendnn::memory GetMemoryInOrtFormat(const ZendnnTensor &tensor,
                                        const zendnn::engine &eng);
    bool IsMemoryInExpectedOrtFormat(const zendnn::memory::desc &desc) const;
    void IncMemoryRefCount(zendnn::memory &mem);
    bool UseCPUAllocator();

  private:
    std::string shape_key_;

    std::unordered_map<std::string, std::vector<zendnn::memory>> intermediates_;

    std::unordered_map<std::string, zendnn::memory> inputs_;
    std::unordered_map<std::string, zendnn::memory::desc> inputs_md_;
    std::unordered_set<std::string> input_is_scalar_;


    std::unordered_map<std::string, zendnn::memory> outputs_;
    std::unordered_map<std::string, zendnn::memory::desc> outputs_md_;
    std::unordered_set<std::string> outputs_are_always_copied_;

    //initializer should not be dynamic
    std::unordered_map<std::string, std::vector<zendnn::memory>> initializers_;

    std::vector<zendnn::primitive> net_;
    std::vector<std::unordered_map<int, zendnn::memory>> net_args_;
    std::vector<PrimitiveMemInfo> net_prim_mem_info_;

    std::vector<std::pair<zendnn::memory, zendnn::memory>> reshapes_;
    std::unordered_set<std::string> scalar_outputs_;

    ort_zendnn::ZendnnSubgraph *subgraph_;

    std::vector<std::string> ordered_inputs_;
    std::vector<std::string> ordered_outputs_;

    zendnn::engine cpu_engine_;
    zendnn::engine gpu_engine_;

    OrtMutex mutex_;
    AllocatorPtr alloc_ptr;
    bool cpu_mem_alloc_ = false;

    //for memory debug purpose
    std::vector<std::pair<int,int>> items_to_print_;
    void PrintMemory(const zendnn::memory &mem);

};

}  // namespace ort_zendnn

inline std::ostream &operator<<(std::ostream &os,
                                const zendnn::memory::dims &dims) {
    std::copy(dims.begin(), dims.end(),
              std::ostream_iterator<zendnn::memory::dim>(os, " "));
    return os;
}

inline std::ostream &operator<<(std::ostream &os,
                                const gsl::span<const int64_t> &span) {
    std::copy(span.begin(), span.end(), std::ostream_iterator<int64_t>(os, " "));
    return os;
}

inline std::ostream &operator<<(std::ostream &os,
                                const gsl::span<int64_t> &span) {
    std::copy(span.begin(), span.end(), std::ostream_iterator<int64_t>(os, " "));
    return os;
}

}  // namespace onnxruntime

