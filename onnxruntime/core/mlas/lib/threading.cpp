/*******************************************************************************
* Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
*******************************************************************************/

/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    threading.cpp

Abstract:

    This module implements platform specific threading support.

--*/

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

#include "mlasi.h"

void
MlasExecuteThreaded(
    MLAS_THREADED_ROUTINE* ThreadedRoutine,
    void* Context,
    ptrdiff_t Iterations,
    MLAS_THREADPOOL* ThreadPool
    )
{
    //
    // Execute the routine directly if only one iteration is specified.
    //

    if (Iterations == 1) {
        ThreadedRoutine(Context, 0);
        return;
    }

#if defined(BUILD_MLAS_NO_ONNXRUNTIME)
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

    //
    // Fallback to OpenMP or a serialized implementation.
    //

    //
    // Execute the routine for the specified number of iterations.
    //
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (ptrdiff_t tid = 0; tid < Iterations; tid++) {
        ThreadedRoutine(Context, tid);
    }
#else
    //
    // Schedule the threaded iterations using the thread pool object.
    //

    MLAS_THREADPOOL::TrySimpleParallelFor(ThreadPool, Iterations, [&](ptrdiff_t tid) {
        ThreadedRoutine(Context, tid);
    });
#endif
}


void
MlasTrySimpleParallel(
    MLAS_THREADPOOL * ThreadPool,
    const std::ptrdiff_t Iterations,
    const std::function<void(std::ptrdiff_t tid)>& Work)
{
    //
    // Execute the routine directly if only one iteration is specified.
    //
    if (Iterations == 1) {
        Work(0);
        return;
    }

#if defined(BUILD_MLAS_NO_ONNXRUNTIME)
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);

    //
    // Fallback to OpenMP or a serialized implementation.
    //

    //
    // Execute the routine for the specified number of iterations.
    //
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (ptrdiff_t tid = 0; tid < Iterations; tid++) {
        Work(tid);
    }
#else
    //
    // Schedule the threaded iterations using the thread pool object.
    //

    MLAS_THREADPOOL::TrySimpleParallelFor(ThreadPool, Iterations, Work);
#endif
}
