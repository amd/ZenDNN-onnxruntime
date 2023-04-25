// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorReduce : public DmlOperator, public ReduceHelperBase
{
public:
    DmlOperatorReduce(
        const MLOperatorKernelCreationContext& kernelInfo,
        DML_REDUCE_FUNCTION function,
        uint32_t opsetVersion
        )
    :   DmlOperator(kernelInfo),
        ReduceHelperBase(
            kernelInfo,
            kernelInfo.GetTensorShapeDescription(),
            (function != DML_REDUCE_FUNCTION_ARGMAX && function != DML_REDUCE_FUNCTION_ARGMIN),
            opsetVersion
        )
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() >= 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);
        std::vector<std::optional<uint32_t>> inputIndices = { 0 };
        std::vector<std::optional<uint32_t>> outputIndices = { 0 };
        DmlOperator::Initialize(kernelInfo, inputIndices, outputIndices, std::nullopt, std::nullopt, 1u);

        std::vector<uint32_t> dmlAxes;
        std::vector<DimensionType> reducedDims = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(0);
        for (auto& dim : m_axes)
        {
            // Replace all reduced axes with 1 for their size.
            assert(dim < static_cast<int32_t>(reducedDims.size())); // ReduceHelperBase already validated this.
            reducedDims[dim] = 1;
            dmlAxes.push_back(static_cast<uint32_t>(dim)); // Signed to unsigned which DML expects.
        }

        if (!m_keepDims)
        {
            // DML expects the input and output tensors to have identical counts and doesn't know about
            // ONNX's 'keepdims' attribute, keeping all dimensions anyway rather removing those of size 1.
            // So if m_keepDims is false, the ONNX output dim is different than DML tensor desc dim.
            //
            // ReduceSum example:
            //     input dims: {3, 2, 2}
            //     axes: 1
            //     keepDims: 0
            //
            // The ONNX output expects output dims of {3, 2},
            // while DML expect the output tensor desc of {3, 1, 2}.

            m_outputTensorDescs[0] = CreateTensorDescFromOutput(
                kernelInfo,
                0,
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                reducedDims,
                1 // minimumDimensionCount
            );
        }

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        // Zero the output tensor's memory for ArgMin & ArgMax, which produce INT64 output.
        if (function == DML_REDUCE_FUNCTION_ARGMAX)
        {
            DML_ARGMAX_OPERATOR_DESC argmaxDesc;
            argmaxDesc.AxisDirection = static_cast<DML_AXIS_DIRECTION>(m_selectLastIndex);
            argmaxDesc.InputTensor = inputDescs.data();
            argmaxDesc.OutputTensor = outputDescs.data();
            argmaxDesc.Axes = dmlAxes.data();
            argmaxDesc.AxisCount = gsl::narrow_cast<uint32_t>(dmlAxes.size());

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ARGMAX, &argmaxDesc };
            SetDmlOperatorDesc(opDesc, kernelInfo);
        }
        else if (function == DML_REDUCE_FUNCTION_ARGMIN)
        {
            DML_ARGMIN_OPERATOR_DESC argminDesc;
            argminDesc.AxisDirection = static_cast<DML_AXIS_DIRECTION>(m_selectLastIndex);
            argminDesc.InputTensor = inputDescs.data();
            argminDesc.OutputTensor = outputDescs.data();
            argminDesc.Axes = dmlAxes.data();
            argminDesc.AxisCount = gsl::narrow_cast<uint32_t>(dmlAxes.size());

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ARGMIN, &argminDesc };
            SetDmlOperatorDesc(opDesc, kernelInfo);
        }
        else
        {
            DML_REDUCE_OPERATOR_DESC reduceDesc = {};
            reduceDesc.InputTensor = inputDescs.data();
            reduceDesc.OutputTensor = outputDescs.data();
            reduceDesc.Function = function;
            reduceDesc.Axes = dmlAxes.data();
            reduceDesc.AxisCount = gsl::narrow_cast<uint32_t>(dmlAxes.size());

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_REDUCE, &reduceDesc };
            SetDmlOperatorDesc(opDesc, kernelInfo);
        }
    }

    void Compute(const MLOperatorKernelContext& kernelContext) override
    {
        std::vector<IMLOperatorTensor*> inputTensors = GetInputTensorsForExecute(kernelContext);
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensorsForExecute(kernelContext);

        ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
            m_compiledOperator.Get(),
            m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
            gsl::make_span(inputTensors),
            gsl::make_span(outputTensors)));
    }
};

// A specific type of operation for registration.
template <DML_REDUCE_FUNCTION Function>
class DmlOperatorReduceTemplate : public DmlOperatorReduce
{
public:
    DmlOperatorReduceTemplate(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperatorReduce(kernelInfo, Function, OpsetVersion)
    {
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(ReduceSum13,       VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_SUM>, 13>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceMean13,      VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_AVERAGE>, 13>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceProd13,      VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_MULTIPLY>, 13>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceLogSum13,    VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_LOG_SUM>, 13>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceLogSumExp13, VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_LOG_SUM_EXP>, 13>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceSumSquare13, VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_SUM_SQUARE>, 13>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceL113,        VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_L1>, 13>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceL213,        VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_L2>, 13>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceMax13,       VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_MAX>, 13>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceMin13,       VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_MIN>, 13>);
DML_OP_DEFINE_CREATION_FUNCTION(ArgMax13,          VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_ARGMAX>, 13>);
DML_OP_DEFINE_CREATION_FUNCTION(ArgMin13,          VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_ARGMIN>, 13>);

DML_OP_DEFINE_CREATION_FUNCTION(ReduceMean18,      VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_AVERAGE>, 18>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceProd18,      VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_MULTIPLY>, 18>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceLogSum18,    VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_LOG_SUM>, 18>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceLogSumExp18, VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_LOG_SUM_EXP>, 18>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceSumSquare18, VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_SUM_SQUARE>, 18>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceL118,        VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_L1>, 18>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceL218,        VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_L2>, 18>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceMax18,       VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_MAX>, 18>);
DML_OP_DEFINE_CREATION_FUNCTION(ReduceMin18,       VersionedKernel<DmlOperatorReduceTemplate<DML_REDUCE_FUNCTION_MIN>, 18>);


} // namespace Dml
