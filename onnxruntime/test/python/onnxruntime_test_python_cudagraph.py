# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import gc
import os
import pickle
import sys
import threading
import time

# -*- coding: UTF-8 -*-
import unittest
from typing import Dict, List

import numpy as np
from helper import get_name

import onnxruntime as onnxrt
from onnxruntime.capi.onnxruntime_pybind11_state import Fail


class CudaGraphHelper:
    def __init__(
        self,
        ort_session: onnxrt.InferenceSession,
        io_shape: Dict[str, List[int]],
        device_id: int = 0,
    ):
        self.input_names = [input.name for input in ort_session.get_inputs()]
        self.output_names = [output.name for output in ort_session.get_outputs()]

        self.io_shape = io_shape
        self.io_numpy_type = self.get_io_numpy_type_map(ort_session)
        self.io_binding = ort_session.io_binding()
        self.io_ort_value = {}

        for name in self.input_names + self.output_names:
            ort_value = onnxrt.OrtValue.ortvalue_from_shape_and_type(
                io_shape[name], self.io_numpy_type[name], "cuda", device_id
            )
            self.io_ort_value[name] = ort_value
            if name in self.input_names:
                self.io_binding.bind_ortvalue_input(name, ort_value)
            else:
                self.io_binding.bind_ortvalue_output(name, ort_value)

    def get_io_numpy_type_map(self, ort_session: onnxrt.InferenceSession):
        ort_type_to_numpy_type = {
            "tensor(int64)": np.longlong,
            "tensor(int32)": np.intc,
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
        }

        name_to_numpy_type = {}
        for input in ort_session.get_inputs():
            name_to_numpy_type[input.name] = ort_type_to_numpy_type[input.type]

        for output in ort_session.get_outputs():
            name_to_numpy_type[output.name] = ort_type_to_numpy_type[output.type]

        return name_to_numpy_type

    def update_inputs(self, inputs: Dict[str, np.ndarray]):
        for input_name in self.input_names:
            self.io_ort_value[input_name].update_inplace(inputs[input_name])

    def get_output(self, output_name: str):
        return self.io_ort_value[output_name].numpy()


class TestInferenceSessionWithCudaGraph(unittest.TestCase):
    def testOrtValueUpdateInPlace(self):
        x0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        ortvalue_cpu = onnxrt.OrtValue.ortvalue_from_numpy(x0)
        np.testing.assert_allclose(x0, ortvalue_cpu.numpy())

        x1 = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
        ortvalue_cpu.update_inplace(x1)
        np.testing.assert_allclose(x1, ortvalue_cpu.numpy())

        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            ortvalue_gpu = onnxrt.OrtValue.ortvalue_from_numpy(x0, "cuda", 0)
            np.testing.assert_allclose(x0, ortvalue_gpu.numpy())

            ortvalue_gpu.update_inplace(x1)
            np.testing.assert_allclose(x1, ortvalue_gpu.numpy())

    def testRunModelWithCudaGraph(self):
        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            providers = [("CUDAExecutionProvider", {"enable_cuda_graph": True})]
            INPUT_SIZE = 1280
            x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]] * INPUT_SIZE, dtype=np.float32)
            y = np.array([[0.0], [0.0], [0.0]] * INPUT_SIZE, dtype=np.float32)
            x_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
            y_ortvalue = onnxrt.OrtValue.ortvalue_from_numpy(y, "cuda", 0)

            session = onnxrt.InferenceSession(get_name("matmul_2.onnx"), providers=providers)
            io_binding = session.io_binding()

            # Bind the input and output
            io_binding.bind_ortvalue_input("X", x_ortvalue)
            io_binding.bind_ortvalue_output("Y", y_ortvalue)

            # Two regular runs for the necessary memory allocation and cuda graph capturing
            session.run_with_iobinding(io_binding)
            session.run_with_iobinding(io_binding)
            expected_y = np.array([[5.0], [11.0], [17.0]] * INPUT_SIZE, dtype=np.float32)
            np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

            # After capturing, CUDA graph replay happens from this Run onwards
            session.run_with_iobinding(io_binding)
            np.testing.assert_allclose(expected_y, y_ortvalue.numpy(), rtol=1e-05, atol=1e-05)

            # Update input and then replay CUDA graph
            x_ortvalue.update_inplace(
                np.array(
                    [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]] * INPUT_SIZE,
                    dtype=np.float32,
                )
            )
            session.run_with_iobinding(io_binding)
            np.testing.assert_allclose(
                np.array([[50.0], [110.0], [170.0]] * INPUT_SIZE, dtype=np.float32),
                y_ortvalue.numpy(),
                rtol=1e-05,
                atol=1e-05,
            )

    def testRunUnetModelWithCudaGraph(self):
        if "CUDAExecutionProvider" in onnxrt.get_available_providers():
            providers = [("CUDAExecutionProvider", {"enable_cuda_graph": True})]

            test_model_path = get_name("tiny_unet.onnx")
            with open(get_name("tiny_unet_test_data_1.pickle"), "rb") as file:
                test_data_1 = pickle.load(file)
            with open(get_name("tiny_unet_test_data_2.pickle"), "rb") as file:
                test_data_2 = pickle.load(file)

            io_shape = {
                "sample": list(test_data_1["sample"].shape),
                "timestep": list(test_data_1["timestep"].shape),
                "encoder_hidden_states": list(test_data_1["encoder_hidden_states"].shape),
                "out_sample": list(test_data_1["out_sample"].shape),
            }

            session = onnxrt.InferenceSession(test_model_path, providers=providers)
            cuda_graph_helper = CudaGraphHelper(session, io_shape)
            inputs = {key: test_data_1[key] for key in cuda_graph_helper.input_names}
            cuda_graph_helper.update_inputs(inputs)

            # Two regular runs for the necessary memory allocation and cuda graph capturing
            io_binding = cuda_graph_helper.io_binding
            session.run_with_iobinding(io_binding)
            session.run_with_iobinding(io_binding)
            output = cuda_graph_helper.get_output("out_sample")
            expected_output = test_data_1["out_sample"]
            np.testing.assert_allclose(expected_output[0], output[0], rtol=3e-01, atol=3e-01)

            # After capturing, CUDA graph replay happens from this Run onwards
            session.run_with_iobinding(io_binding)
            output = cuda_graph_helper.get_output("out_sample")
            np.testing.assert_allclose(expected_output, output, rtol=3e-01, atol=3e-01)

            # Update input and then replay CUDA graph
            inputs = {key: test_data_2[key] for key in cuda_graph_helper.input_names}
            cuda_graph_helper.update_inputs(inputs)

            session.run_with_iobinding(io_binding)
            output = cuda_graph_helper.get_output("out_sample")
            expected_output = test_data_2["out_sample"]
            self.assertLess(np.median(expected_output - output), 1e-3)
            np.testing.assert_allclose(expected_output, output, rtol=3e-01, atol=3e-01)


if __name__ == "__main__":
    unittest.main()
