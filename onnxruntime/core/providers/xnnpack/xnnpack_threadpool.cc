// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "xnnpack_threadpool.h"
#include "pthreadpool.h"

namespace onnxruntime {
namespace concurrency {
using Task = std::function<void()>;

XnnpackThreadPool::XnnpackThreadPool(size_t thread_num) : ThreadPool(nullptr, {}, nullptr, 1, false) {
  if (thread_num > 1) {
    xnnpack_thread_pool_ = pthreadpool_create(thread_num);
  }
}

XnnpackThreadPool ::~XnnpackThreadPool() {
  pthreadpool_destroy(xnnpack_thread_pool_);
}

int XnnpackThreadPool::NumThreads() const {
  return static_cast<int>(pthreadpool_get_threads_count(xnnpack_thread_pool_));
}

// Calculates block size based on (1) the iteration cost and (2) parallel
// efficiency. We want blocks to be not too small to mitigate parallelization
// overheads; not too large to mitigate tail effect and potential load
// imbalance and we also want number of blocks to be evenly dividable across
// threads.
template <typename T>
static T divup(const T x, const T y) {
  return static_cast<T>((x + y - 1) / y);
}
// core/common/threadpool.cc
static ptrdiff_t CalculateParallelForBlock(const ptrdiff_t n, int num_threads) {
  const double block_size_f = 109.289;
  constexpr ptrdiff_t max_oversharding_factor = 4;
  ptrdiff_t block_size = std::min(
      n,
      std::max<ptrdiff_t>(divup<ptrdiff_t>(n, max_oversharding_factor * num_threads), static_cast<ptrdiff_t>(block_size_f)));
  const ptrdiff_t max_block_size = std::min(n, 2 * block_size);

  ptrdiff_t block_count = divup(n, block_size);

  // Calculate parallel efficiency as fraction of total CPU time used for
  // computations:
  double max_efficiency =
      static_cast<double>(block_count) / (divup<ptrdiff_t>(block_count, num_threads) * num_threads);

  // Now try to increase block size up to max_block_size as long as it
  // doesn't decrease parallel efficiency.
  for (ptrdiff_t prev_block_count = block_count; max_efficiency < 1.0 && prev_block_count > 1;) {
    // This is the next block size that divides size into a smaller number
    // of blocks than the current block_size.
    ptrdiff_t coarser_block_size = divup(n, prev_block_count - 1);

    if (coarser_block_size > max_block_size) {
      break;  // Reached max block size. Stop.
    }
    // Recalculate parallel efficiency.
    const ptrdiff_t coarser_block_count = divup(n, coarser_block_size);
    prev_block_count = coarser_block_count;
    const double coarser_efficiency =
        static_cast<double>(coarser_block_count) / (divup<ptrdiff_t>(coarser_block_count, num_threads) * num_threads);
    if (coarser_efficiency + 0.01 >= max_efficiency) {
      // Taking it.
      block_size = coarser_block_size;
      if (max_efficiency < coarser_efficiency) {
        max_efficiency = coarser_efficiency;
      }
    }
  }

  return block_size;
}

void XnnpackThreadPool::ParallelFor(std::ptrdiff_t total, double cost_per_unit,
                                    const std::function<void(std::ptrdiff_t first, std::ptrdiff_t last)>& fn) {
  ParallelFor(total, TensorOpCost{0, 0, static_cast<double>(cost_per_unit)}, fn);
}

void XnnpackThreadPool::ParallelFor(std::ptrdiff_t total, const TensorOpCost&,
                                    const std::function<void(std::ptrdiff_t first, std::ptrdiff_t)>& fn) {
  // Compute small problems directly in the caller thread.
  if (total <= 1 || NumThreads() == 1) {
    fn(0, total);
    return;
  }

  uint32_t flags = PTHREADPOOL_FLAG_DISABLE_DENORMALS;
  // flags |= PTHREADPOOL_FLAG_YIELD_WORKERS;

  // block size calculation 1:
  size_t tile = total / NumThreads() + ((total % NumThreads()) >= (NumThreads() / 2) ? 1 : 0);
  tile = std::max<size_t>(1, tile);

  // block size calculation 2:
  size_t block = CalculateParallelForBlock(total, NumThreads());
  if (block > tile) {
    block = tile;
  }
  pthreadpool_parallelize_1d_tile_1d(
      xnnpack_thread_pool_,
      [fn](std::ptrdiff_t start, std::ptrdiff_t range) { fn(start, start + range); },
      total,
      block,
      flags);
}

void XnnpackThreadPool::SimpleParallelFor(std::ptrdiff_t total, const std::function<void(std::ptrdiff_t)>& fn) {
  std::function<void(std::ptrdiff_t, std::ptrdiff_t)> fn_wrapper = [fn](std::ptrdiff_t from, std::ptrdiff_t to) {
    for (auto i = from; i < to; ++i) {
      fn(i);
    }
  };
  return ParallelFor(total, 0.0, fn_wrapper);
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
