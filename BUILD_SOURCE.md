
**`Documentation`** |
------------------- |
To build ONNXRUNTIME with ZenDNN follow below steps.

## Build From Source
### Setup for Linux
Create and activate a conda environment
```
$ conda create -n onnxrt-v1.19.2-zendnn-5.0-rel-env python=3.8 -y
$ conda activate onnxrt-v1.19.2-zendnn-5.0-rel-env
```
Install necessary dependencies
```
$ pip install -U cmake onnx==1.16.1 pytest
$ pip install psutil coloredlogs torch==2.1.2 torchvision transformers==4.39.3 sympy
```

### Download the AMD ZenDNN ONNXRUNTIME source code
Location of AMD ZenDNN ONNXRUNTIME: [AMD ZenDNN ONNXRUNTIME](https://github.com/amd/ZenDNN-onnxruntime).

Checkout AMD ZenDNN ONNXRUNTIME
```
$ git clone https://github.com/amd/ZenDNN-onnxruntime.git
$ cd ZenDNN-onnxruntime
```

The repo defaults to the main development branch which doesn't have ZenDNN support. You need to check out a release branch to build, e.g. `rel-1.19.2_zendnn_rel`.
```
$ git checkout branch_name  # rel-1.19.2_zendnn_rel.
```

### Set environment variables
Set environment variables for optimum performance. Some of the environment variables are for housekeeping purposes and can be ignored.
```
$ source scripts/zendnn_ONNXRT_env_setup.sh
```

### Build and install the pip package
Ensure that you install the ONNX Runtime-ZenDNN package corresponding to the Python version with which you created the conda environment.
```
$ ./build.sh --config Release --build_shared_lib --parallel --use_zendnn --build_wheel --use_openmp --skip_onnx_tests
$ pip install --force-reinstall ./build/Linux/Release/dist/<*.whl>
```

### Quick verification of Build. You should see ZendnnExecutionProvider and ONNXRuntime version
### Change directory to parent folder
```
$ cd ../
$ python -c 'import onnxruntime as ort; print("\nONNXRuntime version : ", ort.__version__); print("\nAvailable Execution Providers : ", ort.get_available_providers())'
```

