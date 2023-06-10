// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "QnnTypes.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/endian_utils.h"

#include "nlohmann/json.hpp"

#include <assert.h>
#include <cstring>
#include <functional>
#include <numeric>
#include <vector>
#include <string>
#include <string_view>
#include <fstream>
#include <unordered_set>

namespace onnxruntime {
namespace qnn {
class QnnOpConfigWrapper;

namespace utils {
size_t GetElementSizeByType(const Qnn_DataType_t& data_type);

size_t GetElementSizeByType(ONNXTensorElementDataType elem_type);

// TODO: make these work with Wrappers?
std::ostream& operator<<(std::ostream& out, const Qnn_Param_t& qnn_param);
std::ostream& operator<<(std::ostream& out, const Qnn_Tensor_t& tensor);
std::ostream& operator<<(std::ostream& out, const QnnOpConfigWrapper& op_conf_wrapper);

class LittleEndianFileWriter {
 public:
  static constexpr size_t DEFAULT_BUF_SIZE = 4096;

  LittleEndianFileWriter() {}

  LittleEndianFileWriter(LittleEndianFileWriter&& other) = default;
  LittleEndianFileWriter& operator=(LittleEndianFileWriter&& other) = default;

  // Create a writer that allocates a buffer from the heap.
  common::Status Open(const char* filepath, size_t backing_buffer_size = DEFAULT_BUF_SIZE) {
    ofs_.open(filepath, std::ofstream::binary);
    ORT_RETURN_IF_NOT(ofs_.is_open(), "Failed to open file ", filepath, " for writing binary data");

    backing_heap_data_ = std::make_unique<unsigned char[]>(backing_buffer_size);
    ORT_RETURN_IF(backing_heap_data_ == nullptr, "Failed to allocate memory for LittleEndianFileWriter's buffer");

    buffer_ = gsl::make_span(backing_heap_data_.get(), backing_buffer_size);
    buffer_tail_ = 0;

    return Status::OK();
  }

  // Create a writer that uses the provided buffer.
  common::Status Open(const char* filepath, gsl::span<unsigned char> buffer) {
    ofs_.open(filepath, std::ofstream::binary);
    ORT_RETURN_IF_NOT(ofs_.is_open(), "Failed to open file ", filepath, " for writing binary data");

    buffer_ = buffer;
    buffer_tail_ = 0;

    return Status::OK();
  }

  int64_t GetFilePosition() {
    return static_cast<int64_t>(ofs_.tellp());
  }

  void Flush() {
    assert(buffer_tail_ <= buffer_.size());
    ofs_.write(reinterpret_cast<const char*>(buffer_.data()), buffer_tail_);
    buffer_tail_ = 0;
  }

  // Write a POD (e.g., int or a basic struct) into stream.
  template <typename T>
  common::Status WriteValue(const T& data) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    constexpr size_t data_size = sizeof(T);

    // Flush current buffer of bytes if writing this data would fill buffer or cause it to overflow.
    if ((buffer_tail_ + data_size) >= buffer_.size()) {
      Flush();
    }

    ORT_RETURN_IF(buffer_.size() - buffer_tail_ < data_size, "Not enough room to write value into buffer");

    // Copy data value into private buffer in little-endian byte order.
    ORT_RETURN_IF_ERROR(onnxruntime::utils::detail::CopyLittleEndian(
        data_size,
        gsl::make_span(reinterpret_cast<const unsigned char*>(&data), data_size),
        buffer_.subspan(buffer_tail_, data_size)));
    buffer_tail_ += data_size;

    return common::Status::OK();
  }

  template <typename T>
  common::Status WriteValues(gsl::span<const T> data) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    const size_t total_write_bytes = data.size_bytes();

    // Flush current buffer of bytes if writing this data would fill buffer or cause it to overflow.
    if ((buffer_tail_ + total_write_bytes) >= buffer_.size()) {
      Flush();
    }

    ORT_RETURN_IF(buffer_.size() - buffer_tail_ < total_write_bytes, "Not enough room to write values into buffer");

    // Copy data value into private buffer in little-endian byte order.
    ORT_RETURN_IF_ERROR(onnxruntime::utils::WriteLittleEndian(data, buffer_.subspan(buffer_tail_, total_write_bytes)));
    buffer_tail_ += total_write_bytes;

    return common::Status::OK();
  }

  common::Status WriteString(std::string_view str, bool write_null_char = false) {
    const size_t str_size = str.size();
    const size_t total_write_bytes = str_size + static_cast<size_t>(write_null_char);

    // If the string length is >= the buffer's size, then just
    // write it to the output file stream directly.
    if (total_write_bytes >= buffer_.size()) {
      Flush();
      ofs_.write(str.data(), str_size);
      if (write_null_char) {
        const char null_term = '\0';
        ofs_.write(&null_term, 1);
      }
      return common::Status::OK();
    }

    // Make room in the buffer (by flushing contents) if necessary.
    if ((buffer_tail_ + total_write_bytes) >= buffer_.size()) {
      Flush();
    }

    assert(buffer_.size() - buffer_tail_ >= total_write_bytes);

    std::memcpy(&buffer_[buffer_tail_], str.data(), str_size);
    buffer_tail_ += str_size;

    if (write_null_char) {
      buffer_[buffer_tail_++] = '\0';
    }

    return common::Status::OK();
  }

 private:
  std::ofstream ofs_;
  std::unique_ptr<unsigned char[]> backing_heap_data_;
  gsl::span<unsigned char> buffer_;
  size_t buffer_tail_ = 0;
};

// Class that allows building a JSON representation of a QNN graph.
// The JSON graph is built in a format that can be loaded with Qualcomm's QNN Netron visualizer.
class QnnJSONGraph {
 public:
  QnnJSONGraph();

  void AddGraphInput(const std::string& input_name);
  void AddGraphOutput(const std::string& output_name);

  /**
   * Add QNN operator to JSON graph.
   *
   * /param op_conf_wrapper QNN operator to add.
   */
  void AddOp(const QnnOpConfigWrapper& op_conf_wrapper, LittleEndianFileWriter& weights_writer);

  /**
   * Finalizes JSON graph (i.e., adds top-level graph metadata) and returns a reference
   * to the JSON object.
   *
   * /return A const reference to the finalized JSON graph object.
   */
  const nlohmann::json& Finalize();

 private:
  void AddOpTensors(gsl::span<const Qnn_Tensor_t> tensors, LittleEndianFileWriter& weights_writer);

  nlohmann::json json_;
  std::unordered_set<std::string> seen_tensors_;   // Tracks tensors already added to JSON graph.
  std::unordered_set<std::string> seen_op_types_;  // Tracks unique operator types.
  std::vector<std::string> graph_input_names_;
  std::vector<std::string> graph_output_names_;
};

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
