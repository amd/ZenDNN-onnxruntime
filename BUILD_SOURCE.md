
**`Documentation`** |
------------------- |
To build ONNXRUNTIME with ZenDNN follow below steps.

## Build From Source
### Setup for Linux
Create and activate a conda environment and install the following dependencies
```
Install dependencies to models
$ pip install -U cmake numpy==1.24.4 protobuf==3.20.2 onnx==1.14.0 pytest

Install dependencies to models
$ pip install psutil coloredlogs torch torchvision transformers sympy
```


### Download the AMD ZenDNN ONNXRUNTIME source code
Location of AMD ZenDNN ONNXRUNTIME: [AMD ZenDNN ONNXRUNTIME](https://github.com/amd/ZenDNN-onnxruntime).

Checkout AMD ZenDNN ONNXRUNTIME
```
$ git clone https://github.com/amd/ZenDNN-onnxruntime.git
$ cd ZenDNN-onnxruntime
```

The repo defaults to the main development branch which doesn't have ZenDNN support. You need to check out a release branch to build, e.g. `release/rel-1.15.1_zendnn_rel`.
```
$ git checkout branch_name  # release/rel-1.15.1_zendnn_rel.
```

### Set environment variables
Set environment variables for optimum performance. Some of the environment variables are for housekeeping purposes and can be ignored.
```
$ source scripts/zendnn_ONNXRT_env_setup.sh
```

### Build and install the pip package
```
$ ./build.sh --config Release --build_shared_lib --parallel --use_zendnn --build_wheel --use_openmp --skip_onnx_tests
$ pip install --force-reinstall ./build/Linux/Release/dist/<*.whl>
$ pip install --force protobuf==3.20.2
```

### Quick verification of Build. You should see ZendnnExecutionProvider and ONNXRuntime version
### Change directory to parent folder
```
$ cd ../
$ python -c 'import onnxruntime as ort; print("\nONNXRuntime version : ", ort.__version__); print("\nAvailable Execution Providers : ", ort.get_available_providers())'
```
