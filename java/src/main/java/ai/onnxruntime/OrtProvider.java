/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

/*
 * Copyright (c) 2020, 2021, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */

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

package ai.onnxruntime;

import java.util.HashMap;
import java.util.Map;

/** The execution providers available through the Java API. */
public enum OrtProvider {
  CPU("CPUExecutionProvider"),
  CUDA("CUDAExecutionProvider"),
  DNNL("DnnlExecutionProvider"),
  ZENDNN("ZendnnExecutionProvider"),
  OPEN_VINO("OpenVINOExecutionProvider"),
  VITIS_AI("VitisAIExecutionProvider"),
  TENSOR_RT("TensorrtExecutionProvider"),
  NNAPI("NnapiExecutionProvider"),
  RK_NPU("RknpuExecutionProvider"),
  DIRECT_ML("DmlExecutionProvider"),
  MI_GRAPH_X("MIGraphXExecutionProvider"),
  ACL("ACLExecutionProvider"),
  ARM_NN("ArmNNExecutionProvider"),
  ROCM("ROCMExecutionProvider"),
  CORE_ML("CoreMLExecutionProvider"),
  XNNPACK("XnnpackExecutionProvider"),
  AZURE("AzureExecutionProvider");

  private static final Map<String, OrtProvider> valueMap = new HashMap<>(values().length);

  static {
    for (OrtProvider p : OrtProvider.values()) {
      valueMap.put(p.name, p);
    }
  }

  private final String name;

  OrtProvider(String name) {
    this.name = name;
  }

  /**
   * Accessor for the internal name of this provider.
   *
   * @return The internal provider name.
   */
  public String getName() {
    return name;
  }

  /**
   * Maps from the name string used by ONNX Runtime into the enum.
   *
   * @param name The provider name string.
   * @return The enum constant.
   */
  public static OrtProvider mapFromName(String name) {
    OrtProvider provider = valueMap.get(name);
    if (provider == null) {
      throw new IllegalArgumentException("Unknown execution provider - " + name);
    } else {
      return provider;
    }
  }
}
