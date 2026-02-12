#pragma once

#include <vector>
#include <cstdint>
#include "databento/record.hpp"

namespace constellation::modules::replay::buffers {

/**
 * @brief InstrumentBuffer holds MboMsg objects for a specific instrument.
 *        This is used in the parallel replay approach to accumulate messages
 *        before processing them in a single batch pass.
 */
class InstrumentBuffer {
public:
  /**
   * @brief Reserve space up front if desired.
   */
  void Reserve(std::size_t capacity) {
    messages_.reserve(capacity);
  }

  /**
   * @brief Add a new MboMsg to this buffer.
   */
  void Add(const databento::MboMsg& msg) {
    messages_.push_back(msg);
  }

  /**
   * @brief Access the internal vector of MboMsg.
   */
  const std::vector<databento::MboMsg>& Messages() const {
    return messages_;
  }

  /**
   * @brief Clear the buffer (without deallocating).
   */
  void Clear() {
    messages_.clear();
  }

  /**
   * @brief Current number of messages stored.
   */
  std::size_t Size() const {
    return messages_.size();
  }

private:
  std::vector<databento::MboMsg> messages_;
};

} // end namespace constellation::modules::replay::buffers
