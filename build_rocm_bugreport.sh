#!/bin/bash

usage() { echo "Usage $0 [--no-config]" 1>&2; exit 1; }

config_cmake=true
use_ckcompiler=false

while getopts ":h-:" optchar; do
    case "$optchar" in
        -)  case "$OPTARG" in
                no-config) config_cmake=false ;;
                use-ckcompiler) use_ckcompiler=true ;;
            esac;;
        *) echo "config"; usage ;;
    esac
done

THIS_DIR=$(dirname $(realpath $0))

set -ex

build_dir_suffix=""
hip_clang_path=/opt/rocm/llvm/bin/clang++
hip_flags=""
if $use_ckcompiler; then
    build_dir_suffix="_ckcompiler"
    hip_clang_path=/usr/local/llvm-git/bin/clang++
    hip_flags="-Wno-deprecated-builtins"
    # export LIBRARY_PATH=${LLVM_PATH}/lib/clang/13.0.0/lib/linux
fi

build_dir="build_rocm${build_dir_suffix}"
# config="RelWithDebInfo"
config="Release"

rocm_home="/opt/rocm"

if $config_cmake; then
    rm -f  ${THIS_DIR}/${build_dir}/${config}/*.so
    # rm -fr ${THIS_DIR}/${build_dir}/${config}/build/lib

    ${THIS_DIR}/build.sh \
        --build_dir ${THIS_DIR}/${build_dir} \
        --config ${config} \
        --cmake_generator Ninja \
        --cmake_extra_defines \
            CMAKE_HIP_COMPILER=${hip_clang_path} \
            CMAKE_HIP_ARCHITECTURES=gfx908,gfx90a \
            CMAKE_HIP_FLAGS="${hip_flags} -DCK_EXPERIMENTAL_INTER_WAVE_SCHEDULING=1" \
            CMAKE_EXPORT_COMPILE_COMMANDS=ON \
            onnxruntime_BUILD_KERNEL_EXPLORER=ON \
        --use_rocm \
        --rocm_home /opt/rocm --nccl_home=/opt/rocm \
        --enable_rocm_profiling \
        --enable_training \
        --enable_training_torch_interop \
        --build_wheel \
        --skip_submodule_sync --skip_tests

        # --use_migraphx \

    ${THIS_DIR}/create-fake-dist-info.sh ${THIS_DIR}/${build_dir}/${config}/build/lib/
else
    printf "\n\tSkipping config\n\n"
    cmake --build ${THIS_DIR}/${build_dir}/${config}
fi
