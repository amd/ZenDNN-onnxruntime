## tools/ci_build/github/azure-pipelines/c-api-noopenmp-packaging-pipelines.yml
- Line 209 (replaced): [`buildparameter: --use_tensorrt --use_tensorrt_builtin_parser --tensorrt_home="C:\local\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8" --cuda_version=11.8 --cuda_home="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"  --enable_onnx_tests --enable_wcos --build_java --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;80"`](https://github.com/microsoft/onnxruntime/tree/main/tools/ci_build/github/azure-pipelines/c-api-noopenmp-packaging-pipelines.yml#L209)

## tools/ci_build/github/azure-pipelines/linux-gpu-tensorrt-daily-perf-pipeline.yml
- Line 11 (replaced): [`default: 8.6.1.6`](https://github.com/microsoft/onnxruntime/tree/main/tools/ci_build/github/azure-pipelines/linux-gpu-tensorrt-daily-perf-pipeline.yml#L11)
- Line 15 (replaced): [`- 8.6.1.6`](https://github.com/microsoft/onnxruntime/tree/main/tools/ci_build/github/azure-pipelines/linux-gpu-tensorrt-daily-perf-pipeline.yml#L15)

## tools/ci_build/github/azure-pipelines/win-gpu-tensorrt-ci-pipeline.yml
- Line 57 (replaced): [`arguments: '--config $(BuildConfig) --build_dir $(Build.BinariesDirectory) --skip_submodule_sync --build_shared_lib --update --cmake_generator "Visual Studio 16 2019" --build_wheel --enable_onnx_tests --use_tensorrt --tensorrt_home="C:\local\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8" --cuda_version=11.6 --cuda_home="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6" --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=75'`](https://github.com/microsoft/onnxruntime/tree/main/tools/ci_build/github/azure-pipelines/win-gpu-tensorrt-ci-pipeline.yml#L57)
- Line 87 (replaced): [`python $(Build.SourcesDirectory)\tools\ci_build\build.py --config $(BuildConfig) --build_dir $(Build.BinariesDirectory) --skip_submodule_sync --build_shared_lib --test --cmake_generator "Visual Studio 16 2019" --build_wheel --enable_onnx_tests --use_tensorrt --tensorrt_home="C:\local\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8" --cuda_version=11.6 --cuda_home="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6" --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=75`](https://github.com/microsoft/onnxruntime/tree/main/tools/ci_build/github/azure-pipelines/win-gpu-tensorrt-ci-pipeline.yml#L87)

## tools/ci_build/github/azure-pipelines/templates/py-packaging-selectable-stage.yml
- Line 395 (replaced): [`EpBuildFlags: --use_tensorrt --use_tensorrt_builtin_parser --tensorrt_home="C:\local\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8" --cuda_version=$(CUDA_VERSION) --cuda_home="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$(CUDA_VERSION)" --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=37;50;52;60;61;70;75;80"`](https://github.com/microsoft/onnxruntime/tree/main/tools/ci_build/github/azure-pipelines/templates/py-packaging-selectable-stage.yml#L395)

## tools/ci_build/github/azure-pipelines/templates/py-packaging-stage.yml
- Line 275 (replaced): [`EP_BUILD_FLAGS: --use_tensorrt --use_tensorrt_builtin_parser --tensorrt_home="C:\local\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8" --cuda_version=11.6 --cuda_home="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"  --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;80"`](https://github.com/microsoft/onnxruntime/tree/main/tools/ci_build/github/azure-pipelines/templates/py-packaging-stage.yml#L275)
- Line 283 (replaced): [`EP_BUILD_FLAGS: --use_tensorrt --use_tensorrt_builtin_parser --tensorrt_home="C:\local\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8" --cuda_version=11.6 --cuda_home="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"  --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;80"`](https://github.com/microsoft/onnxruntime/tree/main/tools/ci_build/github/azure-pipelines/templates/py-packaging-stage.yml#L283)
- Line 291 (replaced): [`EP_BUILD_FLAGS: --use_tensorrt --use_tensorrt_builtin_parser --tensorrt_home="C:\local\TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8" --cuda_version=11.6 --cuda_home="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"  --cmake_extra_defines "CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;80"`](https://github.com/microsoft/onnxruntime/tree/main/tools/ci_build/github/azure-pipelines/templates/py-packaging-stage.yml#L291)

## tools/ci_build/github/linux/docker/Dockerfile.manylinux2014_cuda11_8_tensorrt8_6
- Line 168 (replaced): [`#Install TensorRT 8.6.1.6`](https://github.com/microsoft/onnxruntime/tree/main/tools/ci_build/github/linux/docker/Dockerfile.manylinux2014_cuda11_8_tensorrt8_6#L168)
- Line 170 (replaced): [`RUN v="8.6.1.6-1.cuda11.8" &&\`](https://github.com/microsoft/onnxruntime/tree/main/tools/ci_build/github/linux/docker/Dockerfile.manylinux2014_cuda11_8_tensorrt8_6#L170)

## tools/ci_build/github/linux/docker/Dockerfile.ubuntu_cuda11_8_tensorrt8_6
- Line 34 (replaced): [`RUN v="8.6.1.6-1+cuda11.8" &&\`](https://github.com/microsoft/onnxruntime/tree/main/tools/ci_build/github/linux/docker/Dockerfile.ubuntu_cuda11_8_tensorrt8_6#L34)

