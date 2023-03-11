# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import time
from onnxruntime import InferenceSession, SessionOptions, OrtValue
from onnxruntime.transformers.io_binding_helper import TypeHelper
import numpy as np
import onnxruntime as ort
import onnx
from contextlib import redirect_stdout

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from onnx_model import OnnxModel

class OrtModelBinding:
    def __init__(self, ort_session, io_shape, device_id=0):
        for input in ort_session.get_inputs():
            assert input.name in io_shape
        # for output in ort_session.get_outputs():
        #     assert output.name in io_shape
        input_names = [input.name for input in ort_session.get_inputs()]
        output_names = [output.name for output in ort_session.get_outputs()]

        self.io_shape = io_shape
        self.io_numpy_type = TypeHelper.get_io_numpy_type_map(ort_session)
        self.io_binding = ort_session.io_binding()
        self.io_ort_value = {}

        for name in input_names:
            ort_value = OrtValue.ortvalue_from_shape_and_type(
                io_shape[name], self.io_numpy_type[name], "cuda", device_id
            )
            self.io_ort_value[name] = ort_value
            self.io_binding.bind_ortvalue_input(name, ort_value)

        for name in output_names:
            if name in io_shape:
                ort_value = OrtValue.ortvalue_from_shape_and_type(
                    io_shape[name], self.io_numpy_type[name], "cuda", device_id
                )
                self.io_ort_value[name] = ort_value
                self.io_binding.bind_ortvalue_output(name, ort_value)
            else:
                self.io_binding.bind_output(name, 'cuda')

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        required=False,
        type=str,
        default='unet.onnx',
        help="path of unet onnx model.",
    )

    parser.add_argument(
        "-o",
        "--output",
        required=False,
        type=str,
        default='test_result.txt',
        help="path of test result.",
    )

    args = parser.parse_args()
    return args

def test(input_model, enable_cuda_graph, output_shape = None):
    options = SessionOptions()
    options.enable_mem_pattern = True
    options.enable_mem_reuse = False

    session = InferenceSession(input_model, options, providers=[("CUDAExecutionProvider",
                                                                {'enable_cuda_graph': enable_cuda_graph}), "CPUExecutionProvider"])
    # Create dummy inputs
    batch_size = 1
    num_images_per_prompt = 1
    height = 512 // 8
    width = 512 // 8
    batch_size = batch_size * num_images_per_prompt
    channels = 4
    sequence_length = 77
    hidden_size = 768  # TODO: parse from graph input shape

    sample = np.random.normal(0, 0.01, size=(2 * batch_size, channels, height, width)).astype('float16')
    timestep = np.ones((1), dtype=np.float16)
    encoder_hidden_states = np.random.normal(0, 0.01, size=(2 * batch_size, sequence_length, hidden_size)).astype('float16')

    io_shape = {
        "sample": list(sample.shape),
        "timestep": list(timestep.shape),
        "encoder_hidden_states": list(encoder_hidden_states.shape),
        #"out_sample": [2 * batch_size, 4, height, width],
    }

    if output_shape:
        io_shape["out_sample"] = output_shape

    model_bindings = OrtModelBinding(session, io_shape)

    model_bindings.io_ort_value["encoder_hidden_states"].update_inplace(encoder_hidden_states)
    model_bindings.io_ort_value["sample"].update_inplace(sample)
    model_bindings.io_ort_value["timestep"].update_inplace(timestep)
    session.run_with_iobinding(model_bindings.io_binding)

    if "out_sample" in io_shape:
        output = model_bindings.io_ort_value["out_sample"].numpy()
    else:
        output = model_bindings.io_binding.get_outputs()[0].numpy()

    return output

def run(onnx_model, last_node_name, i):
    for node in onnx_model.model.graph.node:
        if node.name == last_node_name:
            first_output_name = onnx_model.model.graph.output[0].name
            onnx_model.replace_output_of_all_nodes(first_output_name, "dummy_output")
            onnx_model.replace_input_of_all_nodes(node.output[0], first_output_name)
            node.output[0] = first_output_name
            onnx_model.prune_graph(allow_remove_graph_inputs=False)
            print("remaining nodes", len(onnx_model.model.graph.node))

            model_proto_bytes = onnx._serialize(onnx_model.model)
            #onnx.save(onnx_model.model, f"onnx_{i}.onnx")

            try:
                output = test(model_proto_bytes, False)
                output = test(model_proto_bytes, True, output.shape)
                is_ok = isinstance(output, np.ndarray)
            except Exception as e:
                print(e)
                is_ok = False

            result = {"name": last_node_name, "ok": is_ok}
            return result

    result = {"i": i, "name": last_node_name, "ok": "NA"}
    return result

def main():
    args = parse_arguments()

    good_nodes = []

    candidate_nodes = []
    with open(args.input, "rb") as input_file:
        s = input_file.read()
        model = onnx.load_model_from_string(s)
        node_names = [node.name for node in model.graph.node]
        onnx_model = OnnxModel(model)
        # for node in model.graph.node:
        #     if node.name == "SkipLayerNorm_AddBias_17":
        #         candidate_nodes.append(node.name)
        #         good_nodes = onnx_model.get_parent_subgraph_nodes(node, [])

        for node in onnx_model.nodes():
            if node in good_nodes:
                continue

            graph_inputs = onnx_model.get_graph_inputs(node, recursive=True)
            if graph_inputs and len(graph_inputs) == 3:
                candidate_nodes.append(node.name)

    print("nodes:", len(model.graph.node))
    print("good_nodes:", len(good_nodes))
    print("candidate_nodes:", len(candidate_nodes))

    with open(args.output, 'w') as f:
        results = []
        for i, last_node_name in enumerate(candidate_nodes):
            print("last node", last_node_name)
            result = run(onnx_model, last_node_name, i)
            results.append(result)

            print(f"{i}/{len(candidate_nodes)}", result)

            f.write(str(result))
            f.write('\n')
            f.flush()

            # Reset onnx model
            model = onnx.load_model_from_string(s)
            onnx_model = OnnxModel(model)

        print(results)

if __name__ == "__main__":
    main()
