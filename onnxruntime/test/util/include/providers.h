/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*******************************************************************************
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
* LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
*******************************************************************************/

#pragma once
#include "core/providers/cpu/cpu_provider_factory.h"

#ifdef USE_DNNL
#include "core/providers/dnnl/dnnl_provider_factory.h"
#endif
#ifdef USE_ZENDNN
#include "core/providers/zendnn/zendnn_provider_factory.h"
#endif
#ifdef USE_TVM
#include "core/providers/tvm/tvm_provider_factory.h"
#endif
#ifdef USE_TENSORRT
#include "core/providers/tensorrt/tensorrt_provider_factory.h"
#endif
#ifdef USE_OPENVINO
#include "core/providers/openvino/openvino_provider_factory.h"
#endif
#ifdef USE_NNAPI
#include "core/providers/nnapi/nnapi_provider_factory.h"
#endif
#ifdef USE_COREML
#include "core/providers/coreml/coreml_provider_factory.h"
#endif
#ifdef USE_DML
#include "core/providers/dml/dml_provider_factory.h"
#endif
#ifdef USE_ACL
#include "core/providers/acl/acl_provider_factory.h"
#endif
#ifdef USE_ARMNN
#include "core/providers/armnn/armnn_provider_factory.h"
#endif
#ifdef USE_MIGRAPHX
#include "core/providers/migraphx/migraphx_provider_factory.h"
#endif
#ifdef USE_XNNPACK
#include "core/providers/xnnpack/xnnpack_provider_factory_creator.h"
#endif
#ifdef USE_CANN
#include "core/providers/cann/cann_provider_factory.h"
#endif
