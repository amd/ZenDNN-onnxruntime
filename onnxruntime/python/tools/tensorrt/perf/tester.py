import argparse
import coloredlogs
import csv
import json
import logging
import tempfile
import time
import timeit
import os
import re
import subprocess

from datetime import datetime

import numpy as np
import onnxruntime
import onnx
from onnx import numpy_helper  # isort:skip

logger = logging.getLogger("")

NUM_RUNS = 3
INFERENCES_PER_RUN = 4

# Backends
CPU_BE = "ORT-CPUFp32"
CUDA_BE = "ORT-CUDAFp32"
CUDA_FP16_BE = "ORT-CUDAFp16"
TRT_BE = "ORT-TRTFp32"
TRT_FP16_BE = "ORT-TRTFp16"
STANDALONE_TRT_BE = "TRTFp32"
STANDALONE_TRT_FP16_BE = "TRTFp16"

#BACKENDS = [CPU_BE, CUDA_BE, CUDA_FP16_BE, TRT_BE, TRT_FP16_BE, STANDALONE_TRT_BE, STANDALONE_TRT_FP16_BE]
#BACKENDS = [CPU_BE, CUDA_BE, CUDA_FP16_BE, TRT_BE, TRT_FP16_BE]
BACKENDS = [ CUDA_BE]

# ORT Execution Providers
CPU_EP = "CPUExecutionProvider"
CUDA_EP = "CUDAExecutionProvider"
TRT_EP = "TensorrtExecutionProvider"

BE_TO_EPS = {
    CPU_BE: [CPU_EP],
    CUDA_BE: [CUDA_EP],
    CUDA_FP16_BE: [CUDA_EP],
    TRT_BE: [TRT_EP, CUDA_EP],
    TRT_FP16_BE: [TRT_EP, CUDA_EP],
}


def setup_logger(verbose):
    if verbose:
        coloredlogs.install(
            level="DEBUG",
            fmt="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s",
        )
    else:
        coloredlogs.install(fmt="%(message)s")
        logging.getLogger("transformers").setLevel(logging.WARNING)


def resolve_trtexec_path(workspace):
    trtexec_options = get_output(["find", workspace, "-name", "trtexec"])
    trtexec_path = re.search(r".*/bin/trtexec", trtexec_options).group(0)
    return trtexec_path


def is_ort_backend(backend):
    return backend in [CPU_BE, CUDA_BE, CUDA_FP16_BE, TRT_BE, TRT_FP16_BE]


def get_provider_options(providers, trt_ep_options, cuda_ep_options):
    provider_options = []

    for ep in providers:
        if ep == TRT_EP:
            provider_options.append(trt_ep_options)
        elif ep == CUDA_EP:
            provider_options.append(cuda_ep_options)
        else:
            provider_options.append({})

    return provider_options


def get_output(command):
    p = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    output = p.stdout.decode("ascii").strip()
    return output


def split_and_sort_output(string_list):
    string_list = string_list.split("\n")
    string_list.sort()
    return string_list


def find_files(path, name_pattern, are_dirs=False):
    files = []

    cmd = ["find", path, "-name", name_pattern]
    if are_dirs:
        cmd += ["-type", "d"]

    files_str = get_output(cmd)
    if files_str:
        files = files_str.split("\n")

    return files


def parse_models(models_filepath):
    models = {}

    with open(models_filepath) as file_h:
        model_entries = json.load(file_h)
        models = {entry["model_name"] : entry for entry in model_entries}

    return models


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--models_file",
        required=True,
        help="JSON file listing models to test.",
    )

    return parser.parse_args()

def convert_model_from_float_to_float16(model_path, new_model_dir):
    from float16 import convert_float_to_float16
    from onnxmltools.utils import load_model, save_model

    new_model_path = os.path.join(new_model_dir, "new_fp16_model_by_trt_perf.onnx")

    if not os.path.exists(new_model_path):
        onnx_model = load_model(model_path)
        new_onnx_model = convert_float_to_float16(onnx_model)
        save_model(new_onnx_model, new_model_path)

    return new_model_path


def load_onnx_model_zoo_test_data(path, all_inputs_shape, fp16):
    output = get_output(["find", path, "-name", "test_data*", "-type", "d"])
    test_data_set_dir = split_and_sort_output(output)

    inputs = []
    outputs = []

    shape_flag = False
    # if not empty means input shape has been parsed before.
    if len(all_inputs_shape) > 0:
        shape_flag = True

    # find test data path
    for test_data_dir in test_data_set_dir:
        pwd = os.getcwd()
        os.chdir(test_data_dir)

        # load inputs and create bindings
        output = get_output(["find", ".", "-name", "input*"])
        input_data = split_and_sort_output(output)

        input_data_pb = []
        i = 0
        for data in input_data:
            with open(data, "rb") as file_handle:
                tensor = onnx.TensorProto()
                tensor.ParseFromString(file_handle.read())
                tensor_to_array = numpy_helper.to_array(tensor)
                if fp16 and tensor_to_array.dtype == np.dtype(np.float32):
                    tensor_to_array = tensor_to_array.astype(np.float16)
                input_data_pb.append(tensor_to_array)
                if not shape_flag:
                    all_inputs_shape.append(input_data_pb[-1].shape)
        inputs.append(input_data_pb)

        # load outputs
        output = get_output(["find", ".", "-name", "output*"])
        output_data = split_and_sort_output(output)

        if len(output_data) > 0 and output_data[0] != "":
            output_data_pb = []
            for data in output_data:
                tensor = onnx.TensorProto()
                with open(data, "rb") as file_handle:
                    tensor.ParseFromString(file_handle.read())

                    tensor_to_array = numpy_helper.to_array(tensor)

                    if fp16 and tensor_to_array.dtype == np.dtype(np.float32):
                        tensor_to_array = tensor_to_array.astype(np.float16)
                    output_data_pb.append(tensor_to_array)

            outputs.append(output_data_pb)

        os.chdir(pwd)

    return inputs, outputs


def get_test_data(convert_io_fp16, test_data_dir, all_inputs_shape):
    inputs = []
    ref_outputs = []
    inputs, ref_outputs = load_onnx_model_zoo_test_data(test_data_dir, all_inputs_shape, convert_io_fp16)
    return inputs, ref_outputs


def run_symbolic_shape_inference(model_path, new_model_path):
    import onnxruntime.tools.symbolic_shape_infer as symbolic_shape_infer

    try:
        out = symbolic_shape_infer.SymbolicShapeInference.infer_shapes(onnx.load(model_path), auto_merge=True)
        onnx.save(out, new_model_path)
        return True, None
    except Exception as e:
        return False, "Symbolic shape inference error"


def create_ort_session(model_path, providers, provider_options, session_options):
    try:
        return onnxruntime.InferenceSession(
            model_path,
            providers=providers,
            provider_options=provider_options,
            sess_options=session_options,
        )
    except Exception as e:
        # shape inference required on model
        if "shape inference" in str(e):
            new_model_path = model_path[:].replace(".onnx", "_new_by_trt_perf.onnx")
            if not os.path.exists(new_model_path):
                status = run_symbolic_shape_inference(model_path, new_model_path)
                if not status[0]:  # symbolic shape inference error
                    e = status[1]
                    raise Exception(e)
            return onnxruntime.InferenceSession(
                new_model_path,
                providers=providers,
                provider_options=provider_options,
                sess_options=session_options,
            )
        else:
            raise Exception(e)


def get_ort_session_inputs_and_outputs(name, session, ort_input):

    sess_inputs = {}
    sess_outputs = None

    if "bert_squad" in name.lower() or "bert-squad" in name.lower():
        unique_ids_raw_output = ort_input[0]
        input_ids = ort_input[1]
        input_mask = ort_input[2]
        segment_ids = ort_input[3]

        sess_inputs = {
            "unique_ids_raw_output___9:0": unique_ids_raw_output,
            "input_ids:0": input_ids[0:1],
            "input_mask:0": input_mask[0:1],
            "segment_ids:0": segment_ids[0:1],
        }
        sess_outputs = ["unique_ids:0", "unstack:0", "unstack:1"]

    elif "bidaf" in name.lower():
        sess_inputs = {
            "context_word": ort_input[0],
            "context_char": ort_input[2],
            "query_word": ort_input[1],
            "query_char": ort_input[3],
        }
        sess_outputs = ["start_pos", "end_pos"]

    elif "yolov4" in name.lower():
        sess_inputs[session.get_inputs()[0].name] = ort_input[0]
        sess_outputs = ["Identity:0"]

    else:
        sess_inputs = {}
        sess_outputs = []
        for i in range(len(session.get_inputs())):
            sess_inputs[session.get_inputs()[i].name] = ort_input[i]
        for i in range(len(session.get_outputs())):
            sess_outputs.append(session.get_outputs()[i].name)
    return (sess_inputs, sess_outputs)


def get_runtimes_stats(runtimes):
    values = [rtime * 1000.0 for rtime in runtimes]
    avg = np.mean(values)
    std = np.std(values, ddof=1)
    return (avg, std, (std / avg) * 100.0)


def get_avgs(values, window_size):
    result = []

    rsum = 0
    idx = 0

    for val in values:
        if idx == window_size:
            result.append(rsum / window_size)
            rsum = 0
            idx = 0

        rsum += val
        idx += 1

    result.append(rsum / window_size)

    return result


def get_mins(values, window_size):
    result = []

    m = values[0]
    idx = 0

    for val in values:
        if idx == window_size:
            result.append(m)
            m = val
            idx = 0

        if val < m:
            m = val

        idx += 1

    result.append(m)

    return result


def benchmark_model_be(temp_dir, init_dir, comb_idx, num_combs, model_name, model_info, backend):
    logger.info(f"{comb_idx + 1}/{num_combs}: Benchmarking {model_name} on {backend}")

    all_inputs_shape = []  # used for standalone trt
    model_work_dir = os.path.normpath(os.path.join(init_dir, model_info["working_directory"]))
    model_path = os.path.normpath(os.path.join(model_work_dir, model_info["model_path"]))
    test_data_dir = os.path.normpath(os.path.join(model_work_dir, model_info["test_data_path"]))

    result = {}
    os.chdir(temp_dir)

    # Set TRT EP options.
    trt_ep_options = {
        "trt_engine_cache_enable": "True",
        "trt_max_workspace_size": "4294967296",
    }

    if "ORT-TRT" in backend:
        trt_ep_options["trt_fp16_enable"] = "True" if "Fp16" in backend else "False"

        # Create/set a directory to store TRT engine caches.
        engine_cache_path = os.path.normpath(os.path.join(temp_dir, "engine_cache"))
        if not os.path.exists(engine_cache_path):
            os.makedirs(engine_cache_path)

        trt_ep_options["trt_engine_cache_path"] = engine_cache_path

    # Get input data (and convert model to fp16 if necessary).
    convert_io_fp16 = False

    if backend == CUDA_FP16_BE:
        if "model_path_fp16" in model_info:
            model_path = os.path.normpath(os.path.join(model_work_dir, model_info["model_path_fp16"]))
        else:
            try:
                model_path = convert_model_from_float_to_float16(model_path, model_work_dir)
                convert_io_fp16 = True
            except Exception:
                return (False, result)

        if "test_data_path_fp16" in model_info:
            test_data_dir = os.path.normpath(os.path.join(model_work_dir, model_info["test_data_path_fp16"]))
            convert_io_fp16 = False
    elif backend == STANDALONE_TRT_FP16_BE:
        convert_io_fp16 = True

    inputs, ref_outputs = get_test_data(convert_io_fp16, test_data_dir, all_inputs_shape)

    if is_ort_backend(backend):

        # Create ORT session.
        sess_opts = onnxruntime.SessionOptions()
        sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        ort_eps = BE_TO_EPS[backend]
        ort_ep_opts = get_provider_options(ort_eps, trt_ep_options, {})

        try:
            sess = create_ort_session(model_path, ort_eps, ort_ep_opts, sess_opts)
        except Exception:
            return (False, result)

        sess_inputs, sess_outputs = get_ort_session_inputs_and_outputs(model_name, sess, inputs[0])

        # First warmup/validation run.
        try:
            sess.run(sess_outputs, sess_inputs)
        except Exception:
            return (False, result)

        # Time multiple runs.
        rtimes = timeit.repeat(
            lambda: sess.run(sess_outputs, sess_inputs),
            number=1,
            repeat=NUM_RUNS * INFERENCES_PER_RUN
        )

        avgs = get_avgs(rtimes, INFERENCES_PER_RUN)
        (avgs_avg, avgs_std, avgs_cv) = get_runtimes_stats(avgs)

        mins = get_mins(rtimes, INFERENCES_PER_RUN)
        (mins_avg, mins_std, mins_cv) = get_runtimes_stats(mins)

        result["avg"] = {
            "avg" : avgs_avg,
            "std" : avgs_std,
            "cv"  : avgs_cv,
        }

        result["min"] = {
            "avg" : mins_avg,
            "std" : mins_std,
            "cv"  : mins_cv,
        }

        logger.info(result)

    else:
        pass

    return (True, result)


def main():
    setup_logger(True)

    clock_info = time.get_clock_info("perf_counter")
    logger.info("Performance clock info:")
    logger.info(clock_info)
    logger.info("\n")

    args = parse_args()
    models = parse_models(args.models_file)

    num_combs = len(models) * len(BACKENDS)
    comb_idx = 0

    init_dir = os.getcwd()

    status = {}
    results = {}

    start = datetime.now()
    for backend in BACKENDS:

        # Ensure ORT supports backend EP.
        if is_ort_backend(backend):
            main_ep = BE_TO_EPS[backend][0]
            if main_ep not in onnxruntime.get_available_providers():
                comb_idx += 1
                continue

        status[backend] = {}
        results[backend] = {}

        for model_name, model_info in models.items():
            with tempfile.TemporaryDirectory() as temp_dir:
                success, result = benchmark_model_be(temp_dir, init_dir, comb_idx, num_combs, model_name, model_info, backend)
                status[backend][model_name] = success

                if success:
                    results[backend][model_name] = result

            comb_idx += 1

    end = datetime.now()
    logger.info("\nElapsed time: %f\n", (end - start).total_seconds())
    logger.info(json.dumps(status))

    with open(os.path.join(init_dir, "tester.csv"), mode="w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["backend", "model", "avg_avg", "avg_cv", "min_avg", "min_cv"])

        for backend, model_info in results.items():
            for model, stats in model_info.items():
                avg_stats = stats["avg"]
                min_stats = stats["min"]

                csv_writer.writerow([backend, model, avg_stats["avg"], avg_stats["cv"],
                                     min_stats["avg"], min_stats["cv"]])


if __name__ == "__main__":
    main()
