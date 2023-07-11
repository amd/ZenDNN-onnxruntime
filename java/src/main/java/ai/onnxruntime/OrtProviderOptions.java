/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
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

import java.io.IOException;

/** An abstract base class for execution provider options classes. */
// Note this lives in ai.onnxruntime to allow subclasses to access the OnnxRuntime.ortApiHandle
// package private field.
public abstract class OrtProviderOptions implements AutoCloseable {
  static {
    try {
      OnnxRuntime.init();
    } catch (IOException e) {
      throw new RuntimeException("Failed to load onnx-runtime library", e);
    }
  }

  protected final long nativeHandle;

  /**
   * Constructs a OrtProviderOptions wrapped around a native pointer.
   *
   * @param nativeHandle The native pointer.
   */
  protected OrtProviderOptions(long nativeHandle) {
    this.nativeHandle = nativeHandle;
  }

  /**
   * Allow access to the api handle pointer for subclasses.
   *
   * @return The api handle.
   */
  protected static long getApiHandle() {
    return OnnxRuntime.ortApiHandle;
  }

  /**
   * Gets the provider enum for this options instance.
   *
   * @return The provider enum.
   */
  public abstract OrtProvider getProvider();

  @Override
  public void close() {
    close(OnnxRuntime.ortApiHandle, nativeHandle);
  }

  /**
   * Native close method.
   *
   * @param apiHandle The api pointer.
   * @param nativeHandle The native options pointer.
   */
  protected abstract void close(long apiHandle, long nativeHandle);

  /**
   * Loads the provider's shared library (if necessary) and calls the create provider function.
   *
   * @param provider The OrtProvider for this options.
   * @param createFunction The create function.
   * @return The pointer to the native provider options object.
   * @throws OrtException If either the library load or provider options create call failed.
   */
  protected static long loadLibraryAndCreate(
      OrtProvider provider, OrtProviderSupplier createFunction) throws OrtException {
    // Shared providers need their libraries loaded before options can be defined.
    switch (provider) {
      case CUDA:
        if (!OnnxRuntime.extractCUDA()) {
          throw new OrtException(
              OrtException.OrtErrorCode.ORT_EP_FAIL, "Failed to find CUDA shared provider");
        }
        break;
      case DNNL:
        if (!OnnxRuntime.extractDNNL()) {
          throw new OrtException(
              OrtException.OrtErrorCode.ORT_EP_FAIL, "Failed to find DNNL shared provider");
        }
        break;
      case ZENDNN:
        if (!OnnxRuntime.extractZENDNN()) {
          throw new OrtException(
              OrtException.OrtErrorCode.ORT_EP_FAIL, "Failed to find ZENDNN shared provider");
        }
        break;
      case OPEN_VINO:
        if (!OnnxRuntime.extractOpenVINO()) {
          throw new OrtException(
              OrtException.OrtErrorCode.ORT_EP_FAIL, "Failed to find OpenVINO shared provider");
        }
        break;
      case ROCM:
        if (!OnnxRuntime.extractROCM()) {
          throw new OrtException(
              OrtException.OrtErrorCode.ORT_EP_FAIL, "Failed to find ROCm shared provider");
        }
        break;
      case TENSOR_RT:
        if (!OnnxRuntime.extractTensorRT()) {
          throw new OrtException(
              OrtException.OrtErrorCode.ORT_EP_FAIL, "Failed to find TensorRT shared provider");
        }
        break;
    }

    return createFunction.create();
  }

  /** Functional interface mirroring a Java supplier, but can throw OrtException. */
  @FunctionalInterface
  public interface OrtProviderSupplier {
    /**
     * Calls the function to get the native pointer.
     *
     * @return The native pointer.
     * @throws OrtException If the create call failed.
     */
    public long create() throws OrtException;
  }
}
