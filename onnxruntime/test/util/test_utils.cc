// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/util/include/test_utils.h"

#include "core/common/narrow.h"
#include "core/common/span_utils.h"
#include "core/framework/ort_value.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/framework/tensorprotoutils.h"

#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gmock/gmock.h"

namespace onnxruntime {
namespace test {
namespace {

template <typename T>
Tensor copy_sort(const Tensor& src, const AllocatorPtr& allocator) {
  Tensor result(src.DataType(), src.Shape(), allocator);
  memcpy(result.MutableDataRaw(), src.DataRaw(), src.SizeInBytes());
  auto dst_span = gsl::make_span(result.MutableData<T>(), result.MutableData<T>() + result.Shape().Size());
  std::sort(dst_span.begin(), dst_span.end());
  return result;
}

// Check functions for tensor types
template <typename T>
void sort_expected_and_actual_buffers(const Tensor& expected, Tensor& expected_sorted,
                                      const Tensor& actual, Tensor& actual_sorted) {
  auto allocator = TestCPUExecutionProvider()->GetAllocator(OrtMemTypeDefault);
  expected_sorted = copy_sort<T>(expected, allocator);
  actual_sorted = copy_sort<T>(actual, allocator);
}

// Check functions for tensor types
template <typename T>
void sort_expected_and_actual_buffers(std::vector<T>& expected,
                                      std::vector<T>& actual) {
  ORT_ENFORCE(expected.size() == actual.size(),
              "The 2 containers contain different number of elements");
  std::sort(expected.begin(), expected.end());
  std::sort(actual.begin(), actual.end());
}

// The default implementation compares for equality, specialized versions for
// other types are below
template <typename T>
struct TensorCheck {
  void operator()(const Tensor& expected_tensor, const Tensor& output_tensor,
                  const std::string& provider_type, const ValidateOutputParams& params) const {
    Tensor expected_sorted, output_sorted;
    const T* expected;
    const T* output;
    const auto size = output_tensor.Shape().Size();
    if (params.sort_output_) {
      // if order can be jumbled in the output of an operator, sort both the
      // expected and output buffers prior to
      // comparison this is a "best-effort" algo and should satisfy the
      // requirement for the few ops that do require this
      // support without investing in a more sophisticated infrastructure for the
      // same
      sort_expected_and_actual_buffers<T>(expected_tensor, expected_sorted, output_tensor, output_sorted);
      expected = expected_sorted.Data<T>();
      output = output_sorted.Data<T>();
    } else {
      expected = expected_tensor.Data<T>();
      output = output_tensor.Data<T>();
    }

    for (int i = 0; i < size; ++i) {
      EXPECT_EQ(expected[i], output[i]) << "i:" << i
                                        << ", provider_type: " << provider_type;
    }
  }
};

template <>
struct TensorCheck<uint8_t> {
  void operator()(const Tensor& expected_tensor,
                  const Tensor& output_tensor,
                  const std::string& provider_type, const ValidateOutputParams& params) const {
    const bool has_abs_err = params.absolute_error_.has_value();
    const bool has_rel_err = params.relative_error_.has_value();

    Tensor expected_sorted, output_sorted;
    const uint8_t* expected;
    const uint8_t* output;
    const auto size = output_tensor.Shape().Size();
    if (params.sort_output_) {
      // if order can be jumbled in the output of an operator, sort both the
      // expected and output buffers prior to
      // comparison this is a "best-effort" algo and should satisfy the
      // requirement for the few ops that do require this
      // support without investing in a more sophisticated infrastructure for the
      // same
      sort_expected_and_actual_buffers<uint8_t>(expected_tensor, expected_sorted, output_tensor, output_sorted);
      expected = expected_sorted.Data<uint8_t>();
      output = output_sorted.Data<uint8_t>();
    } else {
      expected = expected_tensor.Data<uint8_t>();
      output = output_tensor.Data<uint8_t>();
    }

    // For uint8_t results, we only allow NNAPI/XNNPACK EP to have an error tolerance, see below for the reason
    // XNNPACK EP will always round to larger. For example, 0.1 will be rounded to 1.0
    // For any other EPs, we still expect an exact match for the results
    // TODO: Verify if DML can possibly have a ROUNDING_MODE parameter and conform to the other EPs #41968513
    if ((provider_type == kNnapiExecutionProvider || provider_type == kDmlExecutionProvider ||
         provider_type == kXnnpackExecutionProvider) &&
        (has_abs_err || has_rel_err)) {
      double threshold = has_abs_err
                             ? *(params.absolute_error_)
                             : 0.0;

      for (int i = 0; i < size; ++i) {
        if (has_rel_err) {
          EXPECT_NEAR(expected[i], output[i],
                      *(params.relative_error_) * expected[i])  // expected[i] is unsigned, can't be negative
              << "i:" << i << ", provider_type: " << provider_type;
        } else {  // has_abs_err
          EXPECT_NEAR(expected[i], output[i], threshold)
              << "i:" << i << ", provider_type: " << provider_type;
        }
      }
    } else {
      for (int i = 0; i < size; ++i) {
        EXPECT_EQ(expected[i], output[i]) << "i:" << i
                                          << ", provider_type: " << provider_type;
      }
    }
  }
};

template <>
struct TensorCheck<int8_t> {
  void operator()(const Tensor& expected_tensor,
                  const Tensor& output_tensor,
                  const std::string& provider_type,
                  const ValidateOutputParams& params) const {
    Tensor expected_sorted, output_sorted;
    const int8_t* expected;
    const int8_t* output;
    const auto size = output_tensor.Shape().Size();
    if (params.sort_output_) {
      // if order can be jumbled in the output of an operator, sort both the
      // expected and output buffers prior to
      // comparison this is a "best-effort" algo and should satisfy the
      // requirement for the few ops that do require this
      // support without investing in a more sophisticated infrastructure for the
      // same
      sort_expected_and_actual_buffers<int8_t>(expected_tensor, expected_sorted, output_tensor, output_sorted);
      expected = expected_sorted.Data<int8_t>();
      output = output_sorted.Data<int8_t>();
    } else {
      expected = expected_tensor.template Data<int8_t>();
      output = output_tensor.template Data<int8_t>();
    }

    const bool has_abs_err = params.absolute_error_.has_value();
    if (has_abs_err) {
      double threshold = *(params.absolute_error_);

      for (int i = 0; i < size; ++i) {
        EXPECT_NEAR(expected[i], output[i], threshold)
            << "i:" << i << ", provider_type: " << provider_type;
      }
    } else {
      for (int i = 0; i < size; ++i) {
        EXPECT_EQ(expected[i], output[i])
            << "i:" << i << ", provider_type: " << provider_type;
      }
    }
  }
};

template <>
struct TensorCheck<double> {
  void operator()(const Tensor& expected_tensor,
                  const Tensor& output_tensor,
                  const std::string& provider_type,
                  const ValidateOutputParams& params) const {
    auto size = output_tensor.Shape().Size();

    bool has_abs_err = params.absolute_error_.has_value();
    bool has_rel_err = params.relative_error_.has_value();

    // deal with rare cases in which order of output data from a kernel MAY be
    // undefined
    Tensor expected_sorted, output_sorted;
    const double* expected;
    const double* output;
    if (params.sort_output_) {
      sort_expected_and_actual_buffers<double>(expected_tensor, expected_sorted, output_tensor, output_sorted);
      expected = expected_sorted.Data<double>();
      output = output_sorted.Data<double>();
    } else {
      expected = expected_tensor.Data<double>();
      output = output_tensor.Data<double>();
    }

    double threshold = 0.001;
#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DML)
    threshold = 0.005;
#endif

    for (int i = 0; i < size; ++i) {
      // NOTE: Check isnan first to work around MSVC linker bug when /LTCG:incremental is specified.
      // If the isinf check is first the isnan check and branch gets omitted
      if (std::isnan(expected[i])) {
        ASSERT_TRUE(std::isnan(output[i])) << "Expected NaN. i:" << i << ", provider_type: " << provider_type;
      } else if (std::isinf(expected[i])) {  // Test infinity for equality
        ASSERT_EQ(expected[i], output[i]) << "Expected infinity. i:" << i << ", provider_type: " << provider_type;
      } else {
        if (!has_abs_err && !has_rel_err) {
          // the default for existing tests
          ASSERT_NEAR(expected[i], output[i], threshold)
              << "i:" << i << ", provider_type: " << provider_type;
        } else {
          if (has_abs_err) {
            ASSERT_NEAR(expected[i], output[i],
                        *(params.absolute_error_))
                << "i:" << i << ", provider_type: " << provider_type;
          }
          if (has_rel_err) {
            ASSERT_NEAR(expected[i], output[i],
                        *(params.relative_error_) *
                            std::abs(expected[i]))
                << "i:" << i << ", provider_type: " << provider_type;
          }
        }
      }
    }
  }
};

template <typename TypeToCheck>
void InternalNumericalCheck(const Tensor& expected_tensor,
                            const Tensor& output_tensor,
                            const std::string& provider_type,
                            const ValidateOutputParams& params) {
  const bool has_abs_err = params.absolute_error_.has_value();
  const bool has_rel_err = params.relative_error_.has_value();

  // deal with rare cases in which order of output data from a kernel MAY be
  // undefined
  Tensor expected_sorted, output_sorted;
  const TypeToCheck* expected;
  const TypeToCheck* output;
  auto size = output_tensor.Shape().Size();
  if (params.sort_output_) {
    sort_expected_and_actual_buffers<TypeToCheck>(expected_tensor, expected_sorted, output_tensor, output_sorted);
    expected = expected_sorted.Data<TypeToCheck>();
    output = output_sorted.Data<TypeToCheck>();
  } else {
    expected = expected_tensor.Data<TypeToCheck>();
    output = output_tensor.Data<TypeToCheck>();
  }

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DML)
  constexpr float threshold = 0.005f;
#else
  constexpr float threshold = 0.0001f;
#endif

  for (int i = 0; i < size; ++i) {
    // NOTE: Check isnan first to work around MSVC linker bug when /LTCG:incremental is specified.
    // If the isinf check is first the isnan check and branch gets omitted
    if (std::isnan(expected[i])) {
      ASSERT_TRUE(std::isnan(output[i])) << "Expected NaN. i:" << i << ", provider_type: " << provider_type;
    } else if (std::isinf(expected[i])) {  // Test infinity for equality
      ASSERT_EQ(expected[i], output[i]) << "Expected infinity. i:" << i << ", provider_type: " << provider_type;
    } else {
      if (!has_abs_err && !has_rel_err) {
        // the default for existing tests
        ASSERT_NEAR(expected[i], output[i], threshold)
            << "i:" << i << ", provider_type: " << provider_type;
      } else {
        if (has_abs_err) {
          ASSERT_NEAR(expected[i], output[i],
                      *(params.absolute_error_))
              << "i:" << i << ", provider_type: " << provider_type;
        }
        if (has_rel_err) {
          ASSERT_NEAR(expected[i], output[i],
                      *(params.relative_error_) *
                          std::abs(expected[i]))
              << "i:" << i << ", provider_type: " << provider_type;
        }
      }
    }
  }
}

template <>
struct TensorCheck<float> {
  void operator()(const Tensor& expected_tensor,
                  const Tensor& output_tensor,
                  const std::string& provider_type,
                  const ValidateOutputParams& params) const {
    InternalNumericalCheck<float>(expected_tensor, output_tensor, provider_type, params);
  }
};

template <>
struct TensorCheck<MLFloat16> {
  void operator()(const Tensor& expected_tensor,
                  const Tensor& output_tensor,
                  const std::string& provider_type,
                  const ValidateOutputParams& params) const {
    auto* expected = expected_tensor.Data<MLFloat16>();
    auto* output = output_tensor.Data<MLFloat16>();
    auto size = output_tensor.Shape().Size();

    std::vector<float> f_expected(size);
    std::vector<float> f_output(size);
    ConvertMLFloat16ToFloat(expected, f_expected.data(), static_cast<int>(size));
    ConvertMLFloat16ToFloat(output, f_output.data(), static_cast<int>(size));

    // deal with rare cases in which order of output data from a kernel MAY be
    // undefined
    if (params.sort_output_) {
      sort_expected_and_actual_buffers<float>(f_expected, f_output);
    }

    const bool has_abs_err = params.absolute_error_.has_value();
    const bool has_rel_err = params.relative_error_.has_value();

    float threshold = 0.001f;
#if defined(USE_TENSORRT) || defined(ENABLE_TRAINING_CORE) || defined(USE_CUDA) || defined(USE_ROCM)
    threshold = 0.005f;
#elif defined(USE_DML)
    threshold = 0.02f;
#endif
    for (int i = 0; i < size; ++i) {
      if (std::isnan(f_expected[i])) {
        EXPECT_TRUE(std::isnan(f_expected[i])) << "Expected NaN. i:" << i << ", provider_type: " << provider_type;
      } else if (std::isinf(f_expected[i])) {  // Test infinity for equality
        EXPECT_EQ(f_expected[i], f_output[i]) << "Expected infinity. i:" << i << ", provider_type: " << provider_type;
      } else {
        if (!has_abs_err && !has_rel_err) {
          // the default for existing tests
          EXPECT_NEAR(f_expected[i], f_output[i], threshold)
              << "i:" << i << ", provider_type: " << provider_type;
        } else {
          if (has_abs_err) {
            EXPECT_NEAR(f_expected[i], f_output[i],
                        *(params.absolute_error_))
                << "i:" << i << ", provider_type: " << provider_type;
          }
          if (has_rel_err) {
            EXPECT_NEAR(f_expected[i], f_output[i],
                        *(params.relative_error_) *
                            std::abs(expected[i]))
                << "i:" << i << ", provider_type: " << provider_type;
          }
        }
      }
    }
  }
};

template <>
struct TensorCheck<BFloat16> {
  void operator()(const Tensor& expected_tensor,
                  const Tensor& output_tensor,
                  const std::string& provider_type,
                  const ValidateOutputParams& params) const {
    auto* expected = expected_tensor.Data<BFloat16>();
    auto* output = output_tensor.Data<BFloat16>();
    auto size = output_tensor.Shape().Size();

    std::vector<float> f_expected(size);
    std::vector<float> f_output(size);
    BFloat16ToFloat(expected, f_expected.data(), static_cast<size_t>(size));
    BFloat16ToFloat(output, f_output.data(), static_cast<size_t>(size));

    // deal with rare cases in which order of output data from a kernel MAY be
    // undefined
    if (params.sort_output_) {
      sort_expected_and_actual_buffers<float>(f_expected, f_output);
    }

    /// XXX: May need to adjust threshold as BFloat is coarse
    float abs_threshold = 0.0001f;
    float threshold = 0.001f;
#if defined(USE_TENSORRT) || defined(ENABLE_TRAINING_CORE) || defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DML) || defined(USE_DNNL)
    threshold = 0.05f;  // expect at least 95% close
#endif

    for (int i = 0; i < size; ++i) {
      if (std::isnan(f_expected[i])) {
        EXPECT_TRUE(std::isnan(f_expected[i])) << "Expected NaN. i:" << i << ", provider_type: " << provider_type;
      } else if (std::isinf(f_expected[i])) {  // Test infinity for equality
        EXPECT_EQ(f_expected[i], f_output[i]) << "Expected infinity. i:" << i << ", provider_type: " << provider_type;
      } else {
        // the default for existing tests
        const float max_value = fmax(fabs(f_expected[i]), fabs(f_output[i]));
        if (max_value != 0) {  // max_value = 0 means output and expected are 0s.
          const float abs_error = fabs(f_expected[i] - f_output[i]);
          if (abs_error <= abs_threshold) {
            // if the absolute error is small enough, then no need to calculate realative error
            EXPECT_NEAR(0, abs_error, abs_threshold) << "provider_type: "
                                                     << provider_type;
          } else {
            // default for existing tests.
            const float rel_error = abs_error / max_value;
            EXPECT_NEAR(0, rel_error, threshold) << "provider_type: "
                                                 << provider_type;
          }
        }
      }
    }
  }
};

// Check for non tensor types

template <typename T>
void Check(const BaseTester::Data& expected_data, const T& run_output,
           const std::string& provider_type) {
  EXPECT_EQ(expected_data.data_.Get<T>(), run_output) << "provider_type: "
                                                      << provider_type;
}

template <>
void Check<TensorSeq>(const BaseTester::Data& expected_data,
                      const TensorSeq& output_seq,
                      const std::string& provider_type) {
  const auto& exp_seq = expected_data.data_.Get<TensorSeq>();

  // first ensure data types match
  EXPECT_EQ(exp_seq.DataType(), output_seq.DataType())
      << "Data types don't match: Expected: "
      << DataTypeImpl::ToString(exp_seq.DataType())
      << " Output: " << output_seq.DataType()
      << " provider_type: " << provider_type;

  // check num of contained tensors
  size_t expected_num_tensors = exp_seq.Size();
  size_t output_num_tensors = output_seq.Size();
  EXPECT_EQ(expected_num_tensors, output_num_tensors)
      << "Mismatch in number of tensors in the sequence"
      << " Expected: " << expected_num_tensors
      << " Output: " << output_num_tensors
      << " provider_type: " << provider_type;

  // now check the contents of the tensors
  CheckParams check_params = MakeCheckParams(expected_data);

  auto element_type = exp_seq.DataType()->AsPrimitiveDataType()->GetDataType();
  utils::MLTypeCallDispatcher<bool, float, double, uint8_t, uint16_t, uint32_t, uint64_t,
                              int8_t, int16_t, int32_t, int64_t, std::string, MLFloat16,
                              BFloat16>
      t_disp(element_type);

  for (size_t i = 0; i < output_num_tensors; ++i) {
    t_disp.Invoke<TensorCheck>(exp_seq.Get(i), output_seq.Get(i), provider_type, check_params);
  }
}

template <typename Type>
void CheckDispatch(MLDataType type, const BaseTester::Data& expected_data,
                   OrtValue& ort_value, const std::string& provider_type) {
  if (type == DataTypeImpl::GetType<Type>())
    Check<Type>(expected_data, ort_value.Get<Type>(), provider_type);
  else
    ORT_THROW("OpTester:Check() not implemented for output tensor type of ", type);
}

template <typename Type, typename Next, typename... Types>
void CheckDispatch(MLDataType type, const BaseTester::Data& expected_data,
                   OrtValue& ort_value, const std::string& provider_type) {
  if (type == DataTypeImpl::GetType<Type>())
    Check<Type>(expected_data, ort_value.Get<Type>(), provider_type);
  else
    CheckDispatch<Next, Types...>(type, expected_data, ort_value,
                                  provider_type);
}

}  // namespace

void DebugTrap() {
#if _MSC_VER
  __debugbreak();
#else
  raise(SIGTRAP);
#endif
}

void ValidateOutput(const std::string_view output_name, const Tensor& expected, const Tensor& output,
                    const std::string& provider_type, ValidateOutputParams params) {
  ORT_ENFORCE(expected.Shape() == output.Shape(),
              "Expected output shape [", expected.Shape(), "] did not match run output shape [",
              output.Shape(), "] for ", output_name);

  utils::MLTypeCallDispatcher<bool, float, double, uint8_t, uint16_t, uint32_t, uint64_t,
                              int8_t, int16_t, int32_t, int64_t, std::string, MLFloat16,
                              BFloat16>
      t_disp(output.GetElementType());

  t_disp.Invoke<TensorCheck>(expected, output, provider_type, params);
}

// TODO: Could/should this use ValidateOutputTensor or vice-versa so we don't have two different implementations
static void VerifyOutputs(const std::vector<std::string>& output_names,
                          const std::vector<OrtValue>& expected_fetches,
                          const std::vector<OrtValue>& fetches,
                          const EPVerificationParams& params) {
  ASSERT_EQ(expected_fetches.size(), fetches.size());

  for (size_t i = 0, end = expected_fetches.size(); i < end; ++i) {
    auto& ltensor = expected_fetches[i].Get<Tensor>();
    auto& rtensor = fetches[i].Get<Tensor>();
    ASSERT_TRUE(SpanEq(ltensor.Shape().GetDims(), rtensor.Shape().GetDims()));
    auto element_type = ltensor.GetElementType();
    switch (element_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_INT32:
        EXPECT_TRUE(SpanEq(ltensor.DataAsSpan<int32_t>(), rtensor.DataAsSpan<int32_t>()))
            << " mismatch for " << output_names[i];
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT64:
        EXPECT_TRUE(SpanEq(ltensor.DataAsSpan<int64_t>(), rtensor.DataAsSpan<int64_t>()))
            << " mismatch for " << output_names[i];
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
        EXPECT_TRUE(SpanEq(ltensor.DataAsSpan<uint8_t>(), rtensor.DataAsSpan<uint8_t>()))
            << " mismatch for " << output_names[i];
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_INT8:
        EXPECT_TRUE(SpanEq(ltensor.DataAsSpan<int8_t>(), rtensor.DataAsSpan<int8_t>()))
            << " mismatch for " << output_names[i];
        break;
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
        EXPECT_THAT(ltensor.DataAsSpan<float>(),
                    ::testing::Pointwise(::testing::FloatNear(params.fp32_abs_err), rtensor.DataAsSpan<float>()));
        break;
      }
      default:
        ORT_THROW("Unhandled data type. Please add 'case' statement for ", element_type);
    }
  }
}

int CountAssignedNodes(const Graph& current_graph, const std::string& ep_type) {
  int count = 0;

  for (const auto& node : current_graph.Nodes()) {
    if (node.GetExecutionProviderType() == ep_type) {
      ++count;
    }

    if (node.ContainsSubgraph()) {
      for (const auto& entry : node.GetSubgraphs()) {
        count += CountAssignedNodes(*entry, ep_type);
      }
    }
  }

  return count;
}

void RunAndVerifyOutputsWithEP(const ORTCHAR_T* model_path, const char* log_id,
                               std::unique_ptr<IExecutionProvider> execution_provider,
                               const NameMLValMap& feeds,
                               const EPVerificationParams& params) {
  // read raw data from model provided by the model_path
  std::ifstream stream(model_path, std::ios::in | std::ios::binary);
  std::string model_data((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  RunAndVerifyOutputsWithEP(model_data, log_id, std::move(execution_provider), feeds, params);
}

void RunAndVerifyOutputsWithEP(const std::string& model_data, const char* log_id,
                               std::unique_ptr<IExecutionProvider> execution_provider,
                               const NameMLValMap& feeds,
                               const EPVerificationParams& params) {
  SessionOptions so;
  so.session_logid = log_id;
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  //
  // get expected output from CPU EP
  //
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_data.data(), static_cast<int>(model_data.size())));
  ASSERT_STATUS_OK(session_object.Initialize());

  const auto& graph = session_object.GetGraph();
  const auto& outputs = graph.GetOutputs();

  // fetch all outputs
  std::vector<std::string> output_names;
  output_names.reserve(outputs.size());
  for (const auto* node_arg : outputs) {
    if (node_arg->Exists()) {
      output_names.push_back(node_arg->Name());
    }
  }

  std::vector<OrtValue> expected_fetches;
  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &expected_fetches));

  auto provider_type = execution_provider->Type();  // copy string so the std::move doesn't affect us

  //
  // get output with EP enabled
  //
  InferenceSessionWrapper session_object2{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object2.RegisterExecutionProvider(std::move(execution_provider)));
  ASSERT_STATUS_OK(session_object2.Load(model_data.data(), static_cast<int>(model_data.size())));
  ASSERT_STATUS_OK(session_object2.Initialize());

  // make sure that some nodes are assigned to the EP, otherwise this test is pointless...
  const auto& graph2 = session_object2.GetGraph();
  auto ep_nodes = CountAssignedNodes(graph2, provider_type);
  if (params.ep_node_assignment == ExpectedEPNodeAssignment::All) {
    // Verify the entire graph is assigned to the EP
    ASSERT_EQ(ep_nodes, graph2.NumberOfNodes()) << "Not all nodes were assigned to " << provider_type;
  } else if (params.ep_node_assignment == ExpectedEPNodeAssignment::None) {
    // Check if expected failure path is correctly handled by ep. (only used in NNAPI EP QDQ model test case for now)
    ASSERT_EQ(ep_nodes, 0) << "No nodes are supposed to be assigned to " << provider_type;
  } else {
    ASSERT_GT(ep_nodes, 0) << "No nodes were assigned to " << provider_type;
  }

  // Run with EP and verify the result
  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session_object2.Run(run_options, feeds, output_names, &fetches));
  VerifyOutputs(output_names, expected_fetches, fetches, params);

  if (params.graph_verifier) {
    (*params.graph_verifier)(graph2);
  }
}

void CheckShapeEquality(const ONNX_NAMESPACE::TensorShapeProto* shape1,
                        const ONNX_NAMESPACE::TensorShapeProto* shape2) {
  EXPECT_NE(shape1, nullptr);
  EXPECT_NE(shape2, nullptr);
  EXPECT_EQ(shape1->dim_size(), shape2->dim_size()) << "Shapes do not have same rank";
  auto min_dims = std::min(shape1->dim_size(), shape2->dim_size());
  for (int i = 0; i < min_dims; ++i) {
    auto dim1 = shape1->dim(i);
    auto dim2 = shape2->dim(i);
    EXPECT_EQ(dim1.has_dim_value(), dim2.has_dim_value());
    if (dim1.has_dim_value()) {
      EXPECT_EQ(dim1.dim_value(), dim2.dim_value());
    }
    EXPECT_EQ(dim1.has_dim_param(), dim2.has_dim_param());
    if (dim1.has_dim_param()) {
      EXPECT_EQ(dim1.dim_param(), dim2.dim_param());
    }
  }
}

#if !defined(DISABLE_SPARSE_TENSORS)
void SparseIndicesChecker(const ONNX_NAMESPACE::TensorProto& indices_proto, gsl::span<const int64_t> expected_indicies) {
  using namespace ONNX_NAMESPACE;
  Path model_path;
  std::vector<uint8_t> unpack_buffer;
  gsl::span<const int64_t> ind_span;
  std::vector<int64_t> converted_indices;
  TensorShape ind_shape(indices_proto.dims().data(), indices_proto.dims().size());
  const auto elements = narrow<size_t>(ind_shape.Size());
  const bool has_raw_data = indices_proto.has_raw_data();
  switch (indices_proto.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      if (has_raw_data) {
        const auto& rd = indices_proto.raw_data();
        ASSERT_EQ(rd.size(), elements * sizeof(int64_t));
        ASSERT_STATUS_OK(utils::UnpackInitializerData(indices_proto, model_path, unpack_buffer));
        ind_span = ReinterpretAsSpan<const int64_t>(gsl::make_span(unpack_buffer));
      } else {
        ind_span = gsl::make_span(indices_proto.int64_data().data(), indices_proto.int64_data_size());
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      if (has_raw_data) {
        const auto& rd = indices_proto.raw_data();
        ASSERT_EQ(rd.size(), elements * sizeof(int32_t));
        ASSERT_STATUS_OK(utils::UnpackInitializerData(indices_proto, model_path, unpack_buffer));
        auto int32_span = ReinterpretAsSpan<const int32_t>(gsl::make_span(unpack_buffer));
        converted_indices.insert(converted_indices.cend(), int32_span.begin(), int32_span.end());
      } else {
        converted_indices.insert(converted_indices.cend(), indices_proto.int32_data().cbegin(), indices_proto.int32_data().cend());
      }
      ind_span = gsl::make_span(converted_indices);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
      ASSERT_TRUE(has_raw_data);
      const auto& rd = indices_proto.raw_data();
      ASSERT_EQ(rd.size(), elements * sizeof(int16_t));
      ASSERT_STATUS_OK(utils::UnpackInitializerData(indices_proto, model_path, unpack_buffer));
      auto int16_span = ReinterpretAsSpan<const int16_t>(gsl::make_span(unpack_buffer));
      converted_indices.insert(converted_indices.cend(), int16_span.begin(), int16_span.end());
      ind_span = gsl::make_span(converted_indices);
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      ASSERT_TRUE(has_raw_data);
      const auto& rd = indices_proto.raw_data();
      ASSERT_EQ(rd.size(), elements);
      ASSERT_STATUS_OK(utils::UnpackInitializerData(indices_proto, model_path, unpack_buffer));
      auto int8_span = ReinterpretAsSpan<const int8_t>(gsl::make_span(unpack_buffer));
      converted_indices.insert(converted_indices.cend(), int8_span.begin(), int8_span.end());
      ind_span = gsl::make_span(converted_indices);
      break;
    }
    default:
      ASSERT_TRUE(false);
  }
  ASSERT_TRUE(SpanEq(ind_span, expected_indicies));
}

#endif  // DISABLE_SPARSE_TENSORS

}  // namespace test
}  // namespace onnxruntime
