// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "xnnpack_threadpool.h"
#include "pthreadpool.h"

namespace onnxruntime {
namespace concurrency {
using Task = std::function<void()>;

XnnpackThreadPool::XnnpackThreadPool(size_t thread_num) : ThreadPool(nullptr, {}, nullptr, 1, false),
                                                          thread_flag_(PTHREADPOOL_FLAG_DISABLE_DENORMALS) {
  if (thread_num > 1) {
    xnnpack_thread_pool_ = pthreadpool_create(thread_num);
  }
}

XnnpackThreadPool ::~XnnpackThreadPool() {
  pthreadpool_destroy(xnnpack_thread_pool_);
}

void XnnpackThreadPool::EnableSpinning() {
  thread_flag_ |= PTHREADPOOL_FLAG_YIELD_WORKERS;
}

void XnnpackThreadPool::DisableSpinning() {
  thread_flag_ &= ~PTHREADPOOL_FLAG_YIELD_WORKERS;
}

int XnnpackThreadPool::NumThreads() const {
  return static_cast<int>(pthreadpool_get_threads_count(xnnpack_thread_pool_));
}

ptrdiff_t CalculateParallelForBlock(const ptrdiff_t n, const TensorOpCost& c,
                                    std::function<ptrdiff_t(ptrdiff_t)> block_align, int num_threads);

void XnnpackThreadPool::ParallelFor(std::ptrdiff_t total, double cost_per_unit,
                                    const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn) {
  ParallelFor(total, TensorOpCost{0, 0, static_cast<double>(cost_per_unit)}, fn);
}

void XnnpackThreadPool::ParallelFor(std::ptrdiff_t total, const TensorOpCost& cost,
                                    const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn) {
  // Compute small problems directly in the caller thread.
  if (total <= 1 || NumThreads() == 1) {
    fn(0, total);
    return;
  }
  /*
    size_t tile = total / NumThreads() + ((total % NumThreads()) >= (NumThreads() / 2) ? 1 : 0);
    tile = std::max<size_t>(1, tile);
  */

  size_t block = CalculateParallelForBlock(total, cost, nullptr, NumThreads());

  pthreadpool_parallelize_1d_tile_1d(
      xnnpack_thread_pool_,
      [fn](std::ptrdiff_t start, std::ptrdiff_t range) { fn(start, start + range); },
      total,
      block,
      thread_flag_);
}

void XnnpackThreadPool::SimpleParallelFor(std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) {
  pthreadpool_parallelize_1d(
      xnnpack_thread_pool_,
      fn,
      total,
      thread_flag_);
}

void XnnpackThreadPool::Schedule(std::function<void()>) {
  ORT_ENFORCE(false, "XnnpackThreadPool::Schedule not implemented");
}

void XnnpackThreadPool::StartProfiling() {
}

std::string XnnpackThreadPool::StopProfiling() {
  return "";
}

}  // namespace concurrency
}  // namespace onnxruntime
