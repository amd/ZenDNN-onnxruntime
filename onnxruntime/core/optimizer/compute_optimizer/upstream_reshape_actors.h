// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The optimization here ideally applies to both training and inference,
// while so far we mainly validate training during cooking the optimization.
#ifdef ENABLE_TRAINING_CORE
#pragma once

#include "core/optimizer/compute_optimizer/shared_utils.h"

namespace onnxruntime::optimizer::compute_optimizer {

/**
 * @brief Struct to hold the information of the reshape operations.
 *
 * Initially, an instance of this class for the entry node is created, as the Reshape op propagates
 * to the entry node's inputs, more instances of this class are created. The propagation stops when
 * all inputs are not supported to be reshaped.
 */
struct ReshapeInfo : public UpstreamOperatorInfoBase {
 private:
  static constexpr int kDataInputIndex_ = 0;
  static constexpr int kReshapeOutputIndex_ = 0;

 public:
  ReshapeInfo(const Graph& graph, Node* reshape_node,
              bool is_entry_node_ptr = false)
      : UpstreamOperatorInfoBase(reshape_node, is_entry_node_ptr) {
    const NodeArg* input = reshape_node->InputDefs()[kDataInputIndex_];
    ORT_ENFORCE(input->Shape()->dim_size() == 3, "Only support data of 3D");
    const NodeArg* output = node_ptr->OutputDefs()[kReshapeOutputIndex_];
    last_dim = output->Shape()->dim(0);

    if (is_entry_node_ptr) {
      entry_reshape_arg_name = output->Name();
    }

    const Node* producer = graph.GetProducerNode(input->Name());
    if (producer) {
      // Allow the data input to be graph input or initializer, but this won't be passed through further.
      data_producer_output_index_ = optimizer_utils::IndexOfNodeOutput(*producer, *input);
    }
  }

  int GetDataInputIndex() const {
    return kDataInputIndex_;
  }

  int GetOutputIndex() const {
    return kReshapeOutputIndex_;
  }

  int GetDataProducerOutputIndex() const {
    ORT_ENFORCE(data_producer_output_index_ >= 0, "Data producer output index is not set");
    return data_producer_output_index_;
  }

  std::string entry_reshape_arg_name;

  // The dimension of the output tensor after merging the two leading dims.
  ONNX_NAMESPACE::TensorShapeProto_Dimension last_dim;

 private:
  int data_producer_output_index_{-1};
};

/**
 * @brief Base class for all passthrough actors.
 *
 * Each actor defines rules to determine whether a node can be passed through, and post
 * process after passthrough.
 * PreCheck is the interface to check whether a node can be passed through.
 * The passthrough is done transparently, without any interface required to implement.
 * PostProcess is the interface to do some adaptor work after the passthrough.
 */
class UpStreamReshapeOperatorActorBase : public UpStreamOperatorActorBase {
 public:
  UpStreamReshapeOperatorActorBase() = default;
  virtual ~UpStreamReshapeOperatorActorBase() = default;

  /**
   * @brief Check whether a node can be passed through.
   *  At this point, graph modification is not started, once we see any clues that this node
   *  cannot be passed through, we should return false immediately.
   *
   * @param current_node The node to be checked.
   * @param info The reshape info of the Reshape node.
   * that are allowed to pass through.
   * @param all_input_cmp_rets: Used as a return value - a map of input index to the input's dim compare result.
   *  The key is an integer, which is the index of the input of the current_node.
   *  The value is a vector of DimCompare.
   * @param shape_update_func: Used as a return value - a function to update the shape of the current_node.
   */
  virtual bool PreCheck(const Node& current_node, const ReshapeInfo& info,
                        const logging::Logger& logger,
                        std::vector<int>& propagate_input_indices,
                        std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                        std::function<void(Node& node)>& shape_update_func) = 0;

  /**
   * @brief After Reshape op pass through all inputs, do some post-process work.
   *
   * @param graph The graph that the node belongs to.
   * @param current_node The node that has been passed through.
   * @param info_without_node The reshape info of the Reshape node. BUT AT THIS POINT, the node
   *   pointer is invalid, usage for it is not allowed.
   * @param propagate_input_indices: a vector of input index to do propagation, generated by PreCheck().
   * @param all_input_cmp_rets: a map of shape compare result for ALL input of current_node, generated by PreCheck()
   * @param new_reshape_infos new gather infos that are generated during the pass through for current_node's inputs.
   * @return
   * So far, we don't have requirements to override PostProcess function.

   */
  bool PostProcess(Graph& /* graph */, Node& /* current_node */, const ReshapeInfo& /* info_without_node */,
                   const logging::Logger& /* logger */,
                   std::vector<int>& /* propagate_input_indices */,
                   const std::unordered_map<int, std::vector<DimCompare>>& /* all_input_cmp_rets */,
                   const std::unordered_map<int, ReshapeInfo>& /* new_reshape_infos */) {
    return true;
  }
};

// The inputs are broad-cast-able. The outputs should have the same shape (fully broadcasted shape)
// If an operator cannot meet these requirements, we need to add specialized actor for it.
template <bool AreAllOutputShapesEqual>
class SimplePointwiseReshapeActor : public UpStreamReshapeOperatorActorBase {
 public:
  SimplePointwiseReshapeActor() = default;
  ~SimplePointwiseReshapeActor() = default;

  bool PreCheck(const Node& current_node, const ReshapeInfo& info,
                const logging::Logger& logger,
                std::vector<int>& propagate_input_indices,
                std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                std::function<void(Node& node)>& shape_update_func) override;
};

class MatMulReshapeActor : public UpStreamReshapeOperatorActorBase {
 public:
  MatMulReshapeActor() = default;
  ~MatMulReshapeActor() = default;

  bool PreCheck(const Node& current_node, const ReshapeInfo& info,
                const logging::Logger& logger,
                std::vector<int>& propagate_input_indices,
                std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                std::function<void(Node& node)>& shape_update_func) override;
};

class LayerNormalizationReshapeActor : public UpStreamReshapeOperatorActorBase {
 public:
  LayerNormalizationReshapeActor() = default;
  ~LayerNormalizationReshapeActor() = default;

  bool PreCheck(const Node& current_node, const ReshapeInfo& info,
                const logging::Logger& logger,
                std::vector<int>& propagate_input_indices,
                std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                std::function<void(Node& node)>& shape_update_func) override;
};

/**
 * @brief From the given TensorShape, update the specified dimension with the given value.
 * If no new_dim is provided, the dimension will be removed.
 *
 * @param shape TensorShape used as base shape to modify.
 * @param new_dim The new dimension value.
 * @return TensorShapeProto A copy of "shape" after modification.
 */
ONNX_NAMESPACE::TensorShapeProto CreateNewShapeWithMergedTwoLeadingDims(
    const ONNX_NAMESPACE::TensorShapeProto* shape,
    const ONNX_NAMESPACE::TensorShapeProto_Dimension& new_dim);

}  // namespace onnxruntime::optimizer::compute_optimizer

#endif
