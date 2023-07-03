﻿using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Xunit;

namespace Microsoft.ML.OnnxRuntime.Tests
{
    [Collection("OrtValueTests")]
    public class OrtValueTests
    {
        public OrtValueTests()
        {
        }

        [Fact(DisplayName = "PopulateAndReadStringTensor")]
        public void PopulateAndReadStringTensor()
        {
            OrtEnv.Instance();

            string[] strsRom = { "HelloR", "OrtR", "WorldR" };
            string[] strs = { "Hello", "Ort", "World" };
            long[] shape = { 1, 1, 3 };
            var elementsNum = ArrayUtilities.GetSizeForShape(shape);
            Assert.Equal(elementsNum, strs.Length);
            Assert.Equal(elementsNum, strsRom.Length);

            using (var strTensor = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, shape))
            {
                Assert.True(strTensor.IsTensor);
                Assert.False(strTensor.IsSparseTensor);
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, strTensor.OnnxType);
                var typeShape = strTensor.GetTensorTypeAndShape();
                {
                    Assert.True(typeShape.IsString);
                    Assert.Equal(shape.Length, typeShape.DimensionsCount);
                    var fetchedShape = typeShape.Shape;
                    Assert.Equal(shape.Length, fetchedShape.Length);
                    Assert.Equal(shape, fetchedShape);
                    Assert.Equal(elementsNum, typeShape.ElementCount);
                }

                using (var memInfo = strTensor.GetTensorMemoryInfo())
                {
                    Assert.Equal("Cpu", memInfo.Name);
                    Assert.Equal(OrtMemType.Default, memInfo.GetMemoryType());
                    Assert.Equal(OrtAllocatorType.DeviceAllocator, memInfo.GetAllocatorType());
                }

                // Verify that everything is empty now.
                for (int i = 0; i < elementsNum; ++i)
                {
                    var str = strTensor.GetStringElement(i);
                    Assert.Empty(str);

                    var rom = strTensor.GetStringElementAsMemory(i);
                    Assert.Equal(0, rom.Length);

                    var bytes = strTensor.GetStringElementAsSpan(i);
                    Assert.Equal(0, bytes.Length);
                }

                // Let's populate the tensor with strings.
                for (int i = 0; i < elementsNum; ++i)
                {
                    // First populate via ROM
                    strTensor.FillStringTensorElement(strsRom[i].AsMemory(), i);
                    Assert.Equal(strsRom[i], strTensor.GetStringElement(i));
                    Assert.Equal(strsRom[i], strTensor.GetStringElementAsMemory(i).ToString());
                    Assert.Equal(Encoding.UTF8.GetBytes(strsRom[i]), strTensor.GetStringElementAsSpan(i).ToArray());

                    // Fill via Span
                    strTensor.FillStringTensorElement(strs[i].AsSpan(), i);
                    Assert.Equal(strs[i], strTensor.GetStringElement(i));
                    Assert.Equal(strs[i], strTensor.GetStringElementAsMemory(i).ToString());
                    Assert.Equal(Encoding.UTF8.GetBytes(strs[i]), strTensor.GetStringElementAsSpan(i).ToArray());
                }
            }
        }

        [Fact(DisplayName = "PopulateAndReadStringTensorViaTensor")]
        public void PopulateAndReadStringTensorViaTensor()
        {
            OrtEnv.Instance();

            string[] strs = { "Hello", "Ort", "World" };
            int[] shape = { 1, 1, 3 };

            var tensor = new DenseTensor<string>(strs, shape);

            using (var strTensor = OrtValue.CreateFromStringTensor(tensor))
            {
                Assert.True(strTensor.IsTensor);
                Assert.False(strTensor.IsSparseTensor);
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, strTensor.OnnxType);
                var typeShape = strTensor.GetTensorTypeAndShape();
                {
                    Assert.True(typeShape.IsString);
                    Assert.Equal(shape.Length, typeShape.DimensionsCount);
                    var fetchedShape = typeShape.Shape;
                    Assert.Equal(shape.Length, fetchedShape.Length);
                    Assert.Equal(strs.Length, typeShape.ElementCount);
                }

                using (var memInfo = strTensor.GetTensorMemoryInfo())
                {
                    Assert.Equal("Cpu", memInfo.Name);
                    Assert.Equal(OrtMemType.Default, memInfo.GetMemoryType());
                    Assert.Equal(OrtAllocatorType.DeviceAllocator, memInfo.GetAllocatorType());
                }

                for (int i = 0; i < strs.Length; ++i)
                {
                    // Fill via Span
                    Assert.Equal(strs[i], strTensor.GetStringElement(i));
                    Assert.Equal(strs[i], strTensor.GetStringElementAsMemory(i).ToString());
                    Assert.Equal(Encoding.UTF8.GetBytes(strs[i]), strTensor.GetStringElementAsSpan(i).ToArray());
                }
            }
        }
        static void VerifyTensorCreateWithData<T>(OrtValue tensor, TensorElementType dataType, long[] shape,
            ReadOnlySpan<T> originalData) where T : struct
        {
            // Verify invocation
            var dataTypeInfo = TensorBase.GetTypeInfo(typeof(T));
            Assert.NotNull(dataTypeInfo);
            Assert.Equal(dataType, dataTypeInfo.ElementType);

            var elementsNum = ArrayUtilities.GetSizeForShape(shape);

            Assert.True(tensor.IsTensor);
            Assert.False(tensor.IsSparseTensor);
            Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, tensor.OnnxType);

            var typeInfo = tensor.GetTypeInfo();
            {
                Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, typeInfo.OnnxType);
                var typeShape = typeInfo.TensorTypeAndShapeInfo;
                Assert.Equal(shape.Length, typeShape.DimensionsCount);

                var fetchedShape = typeShape.Shape;
                Assert.Equal(shape.Length, fetchedShape.Length);
                Assert.Equal(shape, fetchedShape);
                Assert.Equal(elementsNum, typeShape.ElementCount);
            }

            using (var memInfo = tensor.GetTensorMemoryInfo())
            {
                Assert.Equal("Cpu", memInfo.Name);
                Assert.Equal(OrtMemType.CpuOutput, memInfo.GetMemoryType());
                Assert.Equal(OrtAllocatorType.DeviceAllocator, memInfo.GetAllocatorType());
            }

            // Verify contained data
            Assert.Equal(originalData.ToArray(), tensor.GetTensorDataAsSpan<T>().ToArray());
        }

        [Fact(DisplayName = "CreateTensorOverManagedBuffer")]
        public void CreateTensorOverManagedBuffer()
        {
            int[] data = { 1, 2, 3 };
            var mem = new Memory<int>(data);
            long[] shape = { 1, 1, 3 };
            var elementsNum = ArrayUtilities.GetSizeForShape(shape);
            Assert.Equal(elementsNum, data.Length);


            var typeInfo = TensorBase.GetElementTypeInfo(TensorElementType.Int32);
            Assert.NotNull(typeInfo);

            // The tensor will be created on top of the managed memory. No copy is made.
            // The memory should stay pinned until the OrtValue instance is disposed. This means
            // stayed pinned until the end of Run() method when you are actually running inference.
            using(var tensor = OrtValue.CreateTensorValueFromMemory(data, shape))
            {
                VerifyTensorCreateWithData<int>(tensor, TensorElementType.Int32, shape, data);
            }
        }

        // One can do create an OrtValue over a device memory and used as input.
        // Just make sure that OrtMemoryInfo is created for GPU.
        [Fact(DisplayName = "CreateTensorOverUnManagedBuffer")]
        public void CreateTensorOverUnmangedBuffer()
        {
            const int Elements = 3;
            // One can use stackalloc as well
            var bufferLen = Elements * sizeof(int);
            var dataPtr = Marshal.AllocHGlobal(bufferLen);
            try
            {
                // Use span to populate chunk of native memory
                Span<int> data;
                unsafe
                {
                    data = new Span<int>(dataPtr.ToPointer(), Elements);
                }
                data[0] = 1;
                data[1] = 2;
                data[2] = 3;

                long[] shape = { 1, 1, 3 };
                var elementsNum = ArrayUtilities.GetSizeForShape(shape);
                Assert.Equal(elementsNum, Elements);

                using (var tensor = OrtValue.CreateTensorValueWithData(OrtMemoryInfo.DefaultInstance, TensorElementType.Int32,
                        shape, dataPtr, bufferLen))
                {
                    VerifyTensorCreateWithData<int>(tensor, TensorElementType.Int32, shape, data);
                }
            }
            finally
            {
                Marshal.FreeHGlobal(dataPtr);
            }
        }

        private static void PopulateAndCheck<T>(T[] data) where T : struct
        {
            var typeInfo = TensorBase.GetTypeInfo(typeof(T));
            Assert.NotNull(typeInfo);

            long[] shape = { data.LongLength };

            using (var ortValue = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance,
                typeInfo.ElementType, shape))
            {
                var dst = ortValue.GetTensorMutableDataAsSpan<T>();
                Assert.Equal(data.Length, dst.Length);

                var src = new Span<T>(data);
                src.CopyTo(dst);
                Assert.Equal(data, ortValue.GetTensorDataAsSpan<T>().ToArray());
            }
        }

        // Create Tensor with allocated memory so we can test copying of the data
        [Fact(DisplayName = "CreateAllocatedTensor")]
        public void CreateAllocatedTensor()
        {
            float[] float_data = { 1, 2, 3, 4, 5, 6, 7, 8 };
            int[] int_data = { 1, 2, 3, 4, 5, 6, 7, 8 };
            ushort[] ushort_data = { 1, 2, 3, 4, 5, 6, 7, 8 };
            double[] dbl_data = { 1, 2, 3, 4, 5, 6, 7, 8 };
            Float16[] fl16_data = { 1, 2, 3, 4, 5, 6, 7, 8 };

            PopulateAndCheck(float_data);
            PopulateAndCheck(int_data);
            PopulateAndCheck(ushort_data);
            PopulateAndCheck(dbl_data);
            PopulateAndCheck(fl16_data);
        }

        private static readonly long[] ml_data_1 = { 1, 2 };
        private static readonly long[] ml_data_2 = { 3, 4 };

        // Use this utility method to create two tensors for Map and Sequence tests
        private static Tuple<OrtValue, OrtValue> CreateTwoTensors(IList<IDisposable> cleanup)
        {
            const int ml_data_dim = 2;
            // For map tensors they must be single dimensional
            long[] shape = { ml_data_dim };

            unsafe
            {
                var ortValue_1 = OrtValue.CreateTensorValueFromMemory(ml_data_1, shape);
                cleanup.Add(ortValue_1);
                var ortValue_2 = OrtValue.CreateTensorValueFromMemory(ml_data_2, shape);
                cleanup.Add(ortValue_2);
                return Tuple.Create(ortValue_1, ortValue_2);
            }
        }

        [Fact(DisplayName = "CreateMap")]
        public void CreateMap()
        {
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var valTuple = CreateTwoTensors(cleanUp);
                using (var map = OrtValue.CreateMap(valTuple.Item1, valTuple.Item2))
                {
                    Assert.Equal(OnnxValueType.ONNX_TYPE_MAP, map.OnnxType);
                    var typeInfo = map.GetTypeInfo();
                    var mapInfo = typeInfo.MapTypeInfo;
                    Assert.Equal(TensorElementType.Int64, mapInfo.KeyType);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, mapInfo.ValueType.OnnxType);

                    // Must return always 2 for map since we have two ort values
                    Assert.Equal(2, map.GetValueCount());

                    var keys = map.GetValue(0, OrtAllocator.DefaultInstance);
                    cleanUp.Add(keys);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, keys.OnnxType);
                    Assert.Equal(ml_data_1, keys.GetTensorDataAsSpan<long>().ToArray());

                    var vals = map.GetValue(1, OrtAllocator.DefaultInstance);
                    cleanUp.Add(vals);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, vals.OnnxType);
                    Assert.Equal(ml_data_2, vals.GetTensorDataAsSpan<long>().ToArray());
                }
            }
        }

        [Fact(DisplayName = "CreateSequence")]
        public void CreateSequence()
        {
            using (var cleanUp = new DisposableListTest<IDisposable>())
            {
                var valTuple = CreateTwoTensors(cleanUp);
                OrtValue[] seqVals = { valTuple.Item1, valTuple.Item2 };
                using (var seq = OrtValue.CreateSequence(seqVals))
                {
                    Assert.Equal(OnnxValueType.ONNX_TYPE_SEQUENCE, seq.OnnxType);
                    var typeInfo = seq.GetTypeInfo();
                    var seqInfo = typeInfo.SequenceTypeInfo;
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, seqInfo.ElementType.OnnxType);

                    // Will return 2 because we put 2 values in the sequence
                    Assert.Equal(2, seq.GetValueCount());

                    var item_0 = seq.GetValue(0, OrtAllocator.DefaultInstance);
                    cleanUp.Add(item_0);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, item_0.OnnxType);
                    Assert.Equal(ml_data_1, item_0.GetTensorDataAsSpan<long>().ToArray());

                    var item_1 = seq.GetValue(1, OrtAllocator.DefaultInstance);
                    cleanUp.Add(item_1);
                    Assert.Equal(OnnxValueType.ONNX_TYPE_TENSOR, item_1.OnnxType);
                    Assert.Equal(ml_data_2, item_1.GetTensorDataAsSpan<long>().ToArray());
                }
            }
        }
    }
}