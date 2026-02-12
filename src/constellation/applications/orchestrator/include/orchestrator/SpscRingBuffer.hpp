#pragma once

#include <atomic>
#include <cstddef>
#include <stdexcept>

/**
 * @file SpscRingBuffer.hpp
 * @brief A single-producer, single-consumer lock-free ring buffer
 *        for transferring MBO messages from the Feed thread to
 *        the MarketBook Updater thread.
 *
 * Phase 2 Implementation for Constellation Orchestrator.
 */

namespace constellation::applications::orchestrator {

/**
 * @brief A simple single-producer single-consumer lock-free ring buffer.
 * @tparam T The type of elements stored (e.g. databento::MboMsg).
 */
template <typename T>
class SpscRingBuffer {
public:
  /**
   * @brief Constructor.
   * @param capacity The fixed capacity of the ring buffer. Must be > 0.
   */
  explicit SpscRingBuffer(std::size_t capacity)
    : capacity_(capacity),
      buffer_(new T[capacity]),
      head_(0),
      tail_(0)
  {
    if (capacity_ < 1) {
      throw std::runtime_error("SpscRingBuffer capacity must be >= 1");
    }
  }

  /**
   * @brief Destructor
   */
  ~SpscRingBuffer() {
    delete[] buffer_;
    buffer_ = nullptr;
  }

  // Non-copyable, non-movable
  SpscRingBuffer(const SpscRingBuffer&) = delete;
  SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;

  /**
   * @brief Try to push an element into the ring. Returns false if full.
   */
  bool TryPush(const T& item) {
    std::size_t head = head_.load(std::memory_order_relaxed);
    std::size_t nextHead = (head + 1) % capacity_;

    if (nextHead == tail_.load(std::memory_order_acquire)) {
      // buffer is full
      return false;
    }
    buffer_[head] = item;
    head_.store(nextHead, std::memory_order_release);
    return true;
  }

  /**
   * @brief Try to pop an element from the ring. Returns false if empty.
   */
  bool TryPop(T& out) {
    std::size_t tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_.load(std::memory_order_acquire)) {
      // buffer is empty
      return false;
    }
    out = buffer_[tail];
    std::size_t nextTail = (tail + 1) % capacity_;
    tail_.store(nextTail, std::memory_order_release);
    return true;
  }

private:
  const std::size_t capacity_;
  T* buffer_;
  // head_ is producer index, tail_ is consumer index
  std::atomic<std::size_t> head_;
  std::atomic<std::size_t> tail_;
};

} // end namespace constellation::applications::orchestrator
