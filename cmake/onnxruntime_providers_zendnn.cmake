#**************************************************************************************
# Modifications Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#**************************************************************************************

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#**************************************************************************************
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#**************************************************************************************

file(GLOB_RECURSE onnxruntime_providers_zendnn_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/zendnn/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/zendnn/*.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
)

source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_zendnn_cc_srcs})
onnxruntime_add_shared_library_module(onnxruntime_providers_zendnn ${onnxruntime_providers_zendnn_cc_srcs})
target_link_directories(onnxruntime_providers_zendnn PRIVATE ${ZENDNN_LIB_DIR})
if (MSVC AND onnxruntime_ENABLE_STATIC_ANALYSIS)
  # zendnn_convgrad.cc(47,0): Warning C6262: Function uses '38816' bytes of stack:  exceeds /analyze:stacksize '16384'.  Consider moving some data to heap.
  target_compile_options(onnxruntime_providers_zendnn PRIVATE  "/analyze:stacksize 131072")
endif()

add_dependencies(onnxruntime_providers_zendnn onnxruntime_providers_shared project_zendnn ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime_providers_zendnn PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS} ${ZENDNN_SOURCE} ${ZENDNN_INCLUDE_DIR} $ENV{ZENDNN_BLIS_PATH}/include ${ZENDNN_OCL_INCLUDE_DIR})
target_link_libraries(onnxruntime_providers_zendnn PRIVATE amdZenDNN ${ZENDNN_BLIS_LIB} ${ONNXRUNTIME_PROVIDERS_SHARED} Boost::mp11 ${ABSEIL_LIBS} ${GSL_TARGET} safeint_interface)

set_target_properties(onnxruntime_providers_zendnn PROPERTIES FOLDER "ONNXRuntime")
set_target_properties(onnxruntime_providers_zendnn PROPERTIES LINKER_LANGUAGE CXX)

# Needed for the provider interface, as it includes training headers when training is enabled
if (onnxruntime_ENABLE_TRAINING_OPS)
  target_include_directories(onnxruntime_providers_zendnn PRIVATE ${ORTTRAINING_ROOT})
endif()

# Needed for threadpool handling
if(onnxruntime_BUILD_JAVA)
  add_compile_definitions(ZENDNN_JAVA)
endif()

if(UNIX)
  set_property(TARGET onnxruntime_providers_zendnn APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/zendnn/version_script.lds -Xlinker --gc-sections -Xlinker -rpath=\$ORIGIN")
  target_link_libraries(onnxruntime_providers_zendnn PRIVATE nsync::nsync_cpp)
elseif(WIN32)
  set_property(TARGET onnxruntime_providers_zendnn APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/zendnn/symbols.def")
else()
  message(FATAL_ERROR "onnxruntime_providers_zendnn unknown platform, need to specify shared library exports for it")
endif()

install(TARGETS onnxruntime_providers_zendnn
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})