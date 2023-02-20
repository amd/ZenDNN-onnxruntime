set(composable_kernel_URL https://github.com/ROCmSoftwarePlatform/composable_kernel.git)
set(composable_kernel_TAG bef0cb20dba0d9b315df46899310478a81c21852) # 2023-02-16 11:54:08 -0800

set(PATCH ${PROJECT_SOURCE_DIR}/patches/composable_kernel/Fix_Clang_Build.patch)

include(FetchContent)
FetchContent_Declare(composable_kernel
  GIT_REPOSITORY ${composable_kernel_URL}
  GIT_TAG        ${composable_kernel_TAG}
  PATCH_COMMAND  git apply --reverse --check ${PATCH} || git apply --ignore-space-change --ignore-whitespace ${PATCH}
)

FetchContent_GetProperties(composable_kernel)
if(NOT composable_kernel_POPULATED)
  FetchContent_Populate(composable_kernel)
  set(BUILD_DEV OFF CACHE BOOL "Disable -Weverything, otherwise, error: 'constexpr' specifier is incompatible with C++98 [-Werror,-Wc++98-compat]" FORCE)
  add_subdirectory(${composable_kernel_SOURCE_DIR} ${composable_kernel_BINARY_DIR} EXCLUDE_FROM_ALL)

  add_library(onnxruntime_composable_kernel_includes INTERFACE)
  target_include_directories(onnxruntime_composable_kernel_includes INTERFACE
    ${composable_kernel_SOURCE_DIR}/include
    ${composable_kernel_SOURCE_DIR}/library/include)

  file(GLOB_RECURSE all_ck_srcs "${composable_kernel_SOURCE_DIR}/*.cpp")
  file(GLOB_RECURSE all_ck_utility_srcs "${composable_kernel_SOURCE_DIR}/library/src/utility/*.cpp")
  set_source_files_properties(${all_ck_srcs} PROPERTIES LANGUAGE HIP)

  # build client example with ort builtin ck
  set(client_gemm_01_gemm_srcs ${composable_kernel_SOURCE_DIR}/client_example/01_gemm/gemm.cpp)
  add_executable(client_gemm_01_gemm ${client_gemm_01_gemm_srcs})
  target_link_libraries(client_gemm_01_gemm PRIVATE onnxruntime_composable_kernel_includes device_gemm_instance)

  # build profiler with ort builtin ck
  set(example_gemm_dl_fp16_srcs ${composable_kernel_SOURCE_DIR}/example/01_gemm/gemm_dl_fp16.cpp ${all_ck_utility_srcs})
  add_executable(example_gemm_dl_fp16 ${example_gemm_dl_fp16_srcs})
  target_link_libraries(example_gemm_dl_fp16 PRIVATE onnxruntime_composable_kernel_includes device_gemm_instance)
endif()
