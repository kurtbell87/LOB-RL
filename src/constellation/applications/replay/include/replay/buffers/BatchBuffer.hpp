#pragma once

#include <deque>
#include <cstddef>
#include "databento/record.hpp"

namespace constellation::modules::replay::buffers {

/**
 * @brief A container for collecting DBN records before distributing them
 *        to instrument buffers. Uses std::deque to reduce copying cost
 *        on expansions, and a max_batch_size to prevent memory blowups.
 */
class BatchBuffer {
public:
  explicit BatchBuffer(std::size_t initial_capacity = 20000,
                       std::size_t max_batch_size = 50000)
    : max_batch_size_(max_batch_size)
  {
    records_.resize(initial_capacity); // optional pre-sizing
    // we can still push_back beyond this without reallocation, since it's a deque.
    used_size_ = 0;
  }

  /**
   * @brief Add a record pointer. If we exceed max_batch_size, user can check IsFull().
   */
  void Add(const databento::Record* rec) {
    if (used_size_ < records_.size()) {
      records_[used_size_] = rec;
      ++used_size_;
    } else {
      // need to push_back beyond our "initial capacity"
      records_.push_back(rec);
      ++used_size_;
    }
  }

  /**
   * @brief Access the stored records as a range [0..used_size_).
   */
  const std::deque<const databento::Record*>& Records() { return records_; }

  /**
   * @brief The current number of records stored.
   */
  std::size_t Size() const { return used_size_; }

  /**
   * @brief Check if the batch has reached or exceeded the max size.
   */
  bool IsFull() const { return (used_size_ >= max_batch_size_); }

  /**
   * @brief Clear the batch buffer for reuse. We do not shrink the deque, we just reset used_size_.
   */
  void Clear() {
    used_size_ = 0;
  }

private:
  std::deque<const databento::Record*> records_;
  std::size_t used_size_{0};
  std::size_t max_batch_size_;
};

} // end namespace constellation::modules::replay::buffers
