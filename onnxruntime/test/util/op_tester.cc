// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/util/include/op_tester.h"

namespace onnxruntime {
namespace test {

void OpTester::AddInitializers(onnxruntime::Graph& graph) {
  for (auto index : InitializerIndexes()) {
    auto& data = input_data_[index];
    auto& tensor = data.data_.Get<Tensor>();
    ONNX_NAMESPACE::TensorProto tensor_proto;

    // 1. set dimension
    auto& shape = tensor.Shape();
    for (auto& dim : shape.GetDims()) {
      tensor_proto.add_dims(dim);
    }

    // 2. set type
    tensor_proto.set_data_type(data.def_.TypeAsProto()->tensor_type().elem_type());

    // 3. data
    if (data.def_.TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
      const std::string* string_data = tensor.Data<std::string>();
      for (auto i = 0; i < shape.Size(); i++) {
        tensor_proto.add_string_data(string_data[i]);
      }
    } else {
      auto buffer_size = tensor.DataType()->Size() * shape.Size();
      tensor_proto.set_raw_data(tensor.DataRaw(), buffer_size);
    }

    // 4. name
    tensor_proto.set_name(data.def_.Name());
    graph.AddInitializedTensor(tensor_proto);
  }
}

void OpTester::AddNodes(onnxruntime::Graph& graph,
                        std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                        std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                        std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) {
  // default behavior is to create a single Node for the op being tested, with
  // node inputs/outputs
  // being 1:1 with graph inputs/outputs.
  auto& node = graph.AddNode("node1", op_, op_, graph_input_defs, graph_output_defs, nullptr, Domain());

  // Add the attributes if any
  for (auto& add_attribute_fn : add_attribute_funcs)
    add_attribute_fn(node);
}

std::unique_ptr<onnxruntime::Model>
OpTester::BuildGraph(const std::unordered_map<std::string, int>& extra_domain_to_version,
                     const ModelOptions& model_options) {
  // Generate the input & output def lists
  std::vector<onnxruntime::NodeArg*> node_input_defs;
  std::vector<onnxruntime::NodeArg*> output_defs;

  for (size_t i = 0; i < input_data_.size(); ++i) {
    node_input_defs.push_back(&input_data_[i].def_);
  }

  for (auto& data : output_data_) {
    output_defs.push_back(&data.def_);
  }

  // Create a simple model
  std::unordered_map<std::string, int> domain_to_version(extra_domain_to_version);
  const auto& domain = Domain();
  if (domain_to_version.count(domain) == 0) {
    domain_to_version.insert({domain, Opset()});
  } else {
    auto key_val = extra_domain_to_version.find(domain);

    ORT_ENFORCE(key_val->second <= Opset());

    if (key_val->second < Opset()) {
      domain_to_version[domain] = Opset();
    }
  }

  auto p_model = std::make_unique<onnxruntime::Model>(
      "test", false, ModelMetaData(), PathString(), CustomSchemaRegistries(),
      domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>{},
      DefaultLoggingManager().DefaultLogger(),
      model_options);

  onnxruntime::Graph& graph = p_model->MainGraph();
  AddNodes(graph, node_input_defs, output_defs, add_attribute_funcs_);

  // Add Initializer
  AddInitializers(graph);
  return p_model;
}

const onnxruntime::Model& OpTester::GetModel() {
  if (model_) {
    return *model_;
  }

  // IsAllowReleasedONNXOpsetsOnlySet() checks for the appropriate env var in the process (i.e.) process-wide
  // `IsAllowReleasedONNXOpsetsOnlySetForThisTest()` is for this specific OpTester instance
  // We will only support released opsets iff IsAllowReleasedONNXOpsetsOnlySet() and `IsAllowReleasedONNXOpsetsOnlySetForThisTest()`
  // are both true
  auto allow_released_onnx_opset_only =
      IsAllowReleasedONNXOpsetsOnlySetForThisTest() && model_load_utils::IsAllowReleasedONNXOpsetsOnlySet();

  if (allow_released_onnx_opset_only) {
    auto& onnx_released_versions =
        ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().LastReleaseVersionMap();
    auto it = onnx_released_versions.find(domain_);
    if (it != onnx_released_versions.end() && opset_version_ > it->second) {
      LOGS_DEFAULT(WARNING) << "Encountered model with opset version greater than released onnx opset version. "
                            << "Skipping this test. To run this test set environment variable ALLOW_RELEASED_ONNX_OPSET_ONLY to \"0\". "
                            << "Opset version of current model is " << opset_version_
                            << ", the latest released onnx opset version is " << it->second << ".";
      GTEST_SKIP();
    }
  }

  const bool strict_shape_type_inference = ctx_.session_options.config_options.GetConfigOrDefault(
                                               kOrtSessionOptionsConfigStrictShapeTypeInference, "1") == "1";
  const ModelOptions model_options(allow_released_onnx_opset_only,
                                   strict_shape_type_inference);

  auto model = BuildGraph({}, model_options);

  if (add_shape_to_tensor_data_ &&
      ctx_.expect_result == ExpectResult::kExpectFailure) {
    // capture possible exceptions from shape inference for invalid testcase
    ORT_TRY {
      status = graph.Resolve(ctx_.resolve_options);
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
      });
    }
  } else {
    status = graph.Resolve(ctx_.resolve_options);
  }

  if (!status.IsOK()) {
    if (ctx_.expect_result == ExpectResult::kExpectFailure) {
      EXPECT_TRUE(!status.IsOK());
      EXPECT_THAT(status.ErrorMessage(),
                  testing::HasSubstr(ctx_.expected_failure_string));
    } else {
      LOGS_DEFAULT(ERROR) << "Resolve failed with status: "
                          << status.ErrorMessage();
      EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
    }
  }

  if (!status.IsOK()) {
    return;
  }
}

auto& graph = p_model->MainGraph();

Status status = Status::OK();
if (!cache_enabled) {
  if (add_shape_to_tensor_data_ &&
      ctx_.expect_result == ExpectResult::kExpectFailure) {
    // capture possible exceptions from shape inference for invalid testcase
    ORT_TRY {
      status = graph.Resolve(ctx_.resolve_options);
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
      });
    }
  } else {
    status = graph.Resolve(ctx_.resolve_options);
  }

  if (!status.IsOK()) {
    if (ctx_.expect_result == ExpectResult::kExpectFailure) {
      EXPECT_TRUE(!status.IsOK());
      EXPECT_THAT(status.ErrorMessage(),
                  testing::HasSubstr(ctx_.expected_failure_string));
    } else {
      LOGS_DEFAULT(ERROR) << "Resolve failed with status: "
                          << status.ErrorMessage();
      EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
    }
  }

  if (!status.IsOK()) {
    return;
  }
}
}  // namespace test
}  // namespace onnxruntime
