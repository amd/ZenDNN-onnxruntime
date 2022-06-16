// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"
#include "core/framework/iexecutor.h"
#include "core/framework/framework_common.h"
#include "core/framework/ort_value.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {

class ExecutionFrame;

class ParallelExecutor : public IExecutor {
 public:
  ParallelExecutor(const SessionState& session_state, const bool& terminate_flag = false);

  common::Status Execute(const SessionState& session_state, const std::vector<int>& feed_mlvalue_idxs,
                         const std::vector<OrtValue>& feeds, const std::vector<int>& fetch_mlvalue_idxs,
                         std::vector<OrtValue>& fetches,
                         const std::unordered_map<size_t, CustomAllocator>& fetch_allocators,
                         const logging::Logger& logger) override;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ParallelExecutor);

  Status RunNodeAsync(size_t p_node_index, const SessionState& session_state, const logging::Logger& logger);

  void EnqueueNode(size_t p_node_index, const SessionState& session_state, const logging::Logger& logger);

  void FinishNodeRun(const Status& status) {

    bool finished = false;
    {
      std::lock_guard<OrtMutex> lock(complete_mutex_);
      finished = out_standings_.fetch_sub(1, std::memory_order_seq_cst) == 1;
      if (!status.IsOK()) {
        errors_.push_back(status);
        errors_num_.fetch_add(1, std::memory_order_relaxed);
      }
    }

    if (finished) {
      //std::cout << "all out standing nodes are completed." << std::endl;
      complete_cv_.notify_one();
    }
  }

  std::unique_ptr<ExecutionFrame> root_frame_;

  // Allow storing in the vector and align to cache line
  struct alignas(64) AtomicHolder {
    std::atomic_int64_t ref_{0};
    AtomicHolder() = default;
    AtomicHolder(int64_t v) : ref_(v) {}
    AtomicHolder(const AtomicHolder& o) noexcept {
      ref_.store(o.ref_.load());
    }
    AtomicHolder& operator=(const AtomicHolder& o) noexcept {
      ref_.store(o.ref_.load());
      return *this;
    }
    AtomicHolder(AtomicHolder&& o) noexcept {
      ref_.store(o.ref_.load());
    }
    AtomicHolder& operator=(AtomicHolder&& o) noexcept {
      ref_.store(o.ref_.load());
      return *this;
    }
  };

  std::vector<AtomicHolder> node_refs_;
  // OrtMutex ref_mutex_;
  std::atomic_int out_standings_;  //protected by complete_mutex_
  OrtMutex complete_mutex_;
  OrtCondVar complete_cv_;
  std::vector<Status> errors_;
  std::atomic_size_t errors_num_{0};

  const bool& terminate_flag_;
  // TODO: Temporary threadpool for the executor.  This is a costly way to handle the problem.
  onnxruntime::concurrency::ThreadPool* const executor_pool_{};
};
}  // namespace onnxruntime
