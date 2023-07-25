#*******************************************************************************
# Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
#*******************************************************************************

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

#*******************************************************************************
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
#*******************************************************************************/

include (ExternalProject)

set(ZENDNN_URL https://github.com/amd/ZenDNN.git)
#  Mention ZENDNN_TAG version to dowload.
set(ZENDNN_TAG v4.0)

set(BLIS_URL https://github.com/amd/blis.git)
set(BLIS_TAG ${BLIS_VERSION})

if (onnxruntime_USE_ZENDNN)
  if(UNIX)
    set(ZENDNN_SHARED_LIB     libamdZenDNN.so)
    set(ZENDNN_SOURCE         ${CMAKE_CURRENT_BINARY_DIR}/zendnn/src/zendnn/src)
    set(ZENDNN_INCLUDE_DIR    ${ZENDNN_SOURCE}/inc)
    set(ZENDNN_LIB_DIR        ${ZENDNN_SOURCE}/_out/lib)
    set(ZENDNN_DLL_PATH       ${ZENDNN_LIB_DIR}/${ZENDNN_SHARED_LIB})

    set(ZENDNN_AOCL_BLIS_LIB  libblis-mt.so.4)
    set(ZENDNN_BLIS_LIB       blis-mt)
    set(ZENDNN_BLIS_SOURCE    ${CMAKE_CURRENT_BINARY_DIR}/blis/src/blis/src)
    set(ZENDNN_BLIS_PATH      ${ZENDNN_BLIS_SOURCE}/build/blis_gcc_build/)
    set(ZENDNN_BLIS_LIB_DIR   ${ZENDNN_BLIS_SOURCE}/build/blis_gcc_build/lib)
    set(ZENDNN_BLIS_DLL_PATH  ${ZENDNN_BLIS_LIB_DIR}/${ZENDNN_AOCL_BLIS_LIB})

    include(ProcessorCount)
    ProcessorCount(N)

    if ($ENV{ZENDNN_ONNXRT_USE_LOCAL_ZENDNN})
      find_program(MAKE_EXE NAMES make REQUIRED)
      ExternalProject_Add(project_zendnn
        PREFIX zendnn
        DOWNLOAD_COMMAND ""
        SOURCE_DIR ${ZENDNN_SOURCE}
        CONFIGURE_COMMAND cd ${ZENDNN_SOURCE} && cp -rf "/$ENV{ZENDNN_LOCAL_PATH}/./" .
        BUILD_COMMAND cd ${ZENDNN_SOURCE} && make -j${N} ZENDNN_BLIS_PATH=${ZENDNN_BLIS_PATH} AOCC=0
        INSTALL_COMMAND ""
      )
    else()
      find_program(MAKE_EXE NAMES make REQUIRED)
      ExternalProject_Add(project_zendnn
        PREFIX zendnn
        GIT_REPOSITORY ${ZENDNN_URL}
        GIT_TAG ${ZENDNN_TAG}
        SOURCE_DIR ${ZENDNN_SOURCE}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND cd ${ZENDNN_SOURCE} && make -j${N} ZENDNN_BLIS_PATH=${ZENDNN_BLIS_PATH} AOCC=0
        INSTALL_COMMAND ""
      )
    endif()

    add_dependencies(project_zendnn project_blis)

    if ($ENV{ZENDNN_ONNXRT_USE_LOCAL_BLIS})
      find_program(MAKE_EXE NAMES make REQUIRED)
      ExternalProject_Add(project_blis
        PREFIX blis
        DOWNLOAD_COMMAND ""
        SOURCE_DIR ${ZENDNN_BLIS_SOURCE}
        CONFIGURE_COMMAND cd ${ZENDNN_BLIS_SOURCE} && cp -rf "/$ENV{BLIS_LOCAL_PATH}/./" . && ./configure --prefix=${ZENDNN_BLIS_PATH} --enable-threading=openmp --enable-cblas amdzen
        BUILD_COMMAND cd ${ZENDNN_BLIS_SOURCE} && make -j${N} install
        INSTALL_COMMAND ""
      )
    else ()
      find_program(MAKE_EXE NAMES make REQUIRED)
      ExternalProject_Add(project_blis
        PREFIX blis
        GIT_REPOSITORY ${BLIS_URL}
        GIT_TAG ${BLIS_TAG}
        SOURCE_DIR ${ZENDNN_BLIS_SOURCE}
        CONFIGURE_COMMAND cd ${ZENDNN_BLIS_SOURCE} && ./configure --prefix=${ZENDNN_BLIS_PATH} --enable-threading=openmp --enable-cblas amdzen
        BUILD_COMMAND cd ${ZENDNN_BLIS_SOURCE} && make -j${N} install
        INSTALL_COMMAND ""
      )
    endif()
  add_custom_command(
    TARGET project_blis POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy ${ZENDNN_BLIS_PATH}/include/blis/* ${ZENDNN_BLIS_PATH}/include
    COMMAND "${CMAKE_COMMAND}" -E copy ${ZENDNN_BLIS_PATH}/lib/* ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/capi/
    COMMENT "Copying to include directory. TODO: Remove direct copy to onnxruntime/capi and fix in build system"
  )
  elseif(WIN32)
    set(ZENDNN_SHARED_LIB     amdZenDNN.dll)
    set(ZENDNN_SOURCE         ${CMAKE_CURRENT_BINARY_DIR}/zendnn/src/zendnn/src)
    set(ZENDNN_INCLUDE_DIR    ${ZENDNN_SOURCE}/inc)
    set(ZENDNN_LIB_DIR        ${ZENDNN_SOURCE}/build/src/Release)#Debug,Release
    set(ZENDNN_DLL_PATH       ${ZENDNN_LIB_DIR}/${ZENDNN_SHARED_LIB})

    set(ZENDNN_AOCL_BLIS_LIB  AOCL-LibBlis-Win-MT-dll.dll)
    set(ZENDNN_BLIS_LIB       AOCL-LibBlis-Win-MT-dll)
    set(ZENDNN_BLIS_SOURCE    ${CMAKE_CURRENT_BINARY_DIR}/blis/src/blis/src)
    set(ZENDNN_BLIS_PATH      ${ZENDNN_BLIS_SOURCE})
    set(ZENDNN_BLIS_LIB_DIR   ${ZENDNN_BLIS_SOURCE}/bin/Release)
    set(ZENDNN_BLIS_DLL_PATH  ${ZENDNN_BLIS_LIB_DIR}/${ZENDNN_AOCL_BLIS_LIB})
    if ($ENV{ZENDNN_ONNXRT_USE_LOCAL_ZENDNN})
      find_program(MAKE_EXE NAMES msbuild REQUIRED)
      file(REMOVE_RECURSE $ENV{ZENDNN_GIT_ROOT}/build/CMakeCache.txt)
      ExternalProject_Add(project_zendnn
        PREFIX zendnn
        DOWNLOAD_COMMAND COMMAND ${CMAKE_COMMAND} -E copy_directory $ENV{ZENDNN_GIT_ROOT} ${ZENDNN_SOURCE}
        SOURCE_DIR ${ZENDNN_SOURCE}
        BINARY_DIR ${ZENDNN_SOURCE}/build
        CMAKE_GENERATOR "Visual Studio 16 2019"
        CMAKE_GENERATOR_TOOLSET "clangcl"
        CMAKE_ARGS "-DCMAKE_BUILD_TYPE=Release -- -m"
        INSTALL_COMMAND ""
        )
    else()
      find_program(MAKE_EXE NAMES msbuild REQUIRED)
      ExternalProject_Add(project_zendnn
        PREFIX zendnn
        GIT_REPOSITORY ${ZENDNN_URL}
        GIT_TAG ${ZENDNN_TAG}
        SOURCE_DIR ${ZENDNN_SOURCE}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND cd ${ZENDNN_SOURCE} && msbuild WinSln/ZenDNN.sln /p:Configuration=Release -m
        INSTALL_COMMAND ""
      )
    endif()
    if ($ENV{ZENDNN_ONNXRT_USE_LOCAL_BLIS})
      set(ZENDNN_BLIS_LIB_DIR   $ENV{ZENDNN_BLIS_PATH}/lib/ILP64)
      set(ZENDNN_BLIS_DLL_PATH  ${ZENDNN_BLIS_LIB_DIR}/${ZENDNN_AOCL_BLIS_LIB})
    else()
      add_dependencies(project_zendnn project_blis)
      list(APPEND BLIS_CMAKE_ARGS -DCMAKE_BUILD_TYPE:STRING=Release -DAOCL_BLIS_FAMILY:STRING=amdzen -DBUILD_SHARED_LIBS:BOOL=ON)
      list(APPEND BLIS_CMAKE_ARGS -DENABLE_OPENMP:BOOL=ON -DENABLE_MULTITHREADING:BOOL=ON -DENABLE_COMPLEX_RETURN_INTEL:BOOL=ON -DENABLE_AOCL_DYNAMIC:BOOL=ON -- -m)
      ExternalProject_Add(project_blis
        PREFIX blis
        GIT_REPOSITORY ${BLIS_URL}
        GIT_TAG ${BLIS_TAG}
        SOURCE_DIR ${ZENDNN_BLIS_SOURCE}
        CMAKE_GENERATOR "Visual Studio 16 2019"
        CMAKE_GENERATOR_TOOLSET "clangcl"
        CMAKE_ARGS ""
        CMAKE_CACHE_ARGS ${BLIS_CMAKE_ARGS}
        INSTALL_COMMAND ""
      )
      add_custom_command(
        TARGET project_blis POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy ${ZENDNN_BLIS_PATH}/include/amdzen/ ${ZENDNN_BLIS_PATH}/include/
        COMMAND "${CMAKE_COMMAND}" -E copy ${ZENDNN_BLIS_PATH}/bin/Release/AOCL-LibBlis-Win-MT-dll.dll ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/capi/
        COMMAND "${CMAKE_COMMAND}" -E copy ${ZENDNN_BLIS_PATH}/bin/Release/AOCL-LibBlis-Win-MT-dll.lib ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime/capi/
        COMMENT "Copying to include directory. TODO: Remove direct copy to onnxruntime/capi and fix in build system"
      )
    endif()
  endif()
  link_directories(${ZENDNN_LIB_DIR})
  link_directories(${ZENDNN_BLIS_LIB_DIR})
endif()
