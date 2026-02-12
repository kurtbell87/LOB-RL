#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

/**
 * @file IRecordBatch.hpp
 * @brief Defines an abstract interface for Batch record batches storing MBO-like data.
 *
 * The aggregator references these methods. By deriving from IRecordBatch,
 * a concrete RecordBatch class can be passed to Batch aggregator or strategy code
 * that expects an interface reference.
 */

namespace constellation {
namespace interfaces {
namespace batch {

/**
 * @class IRecordBatch
 * @brief A pure abstract interface that Batch aggregator and strategies can use
 *        without referencing concrete implementations.
 *
 * Typical usage:
 *   - Batch aggregator populates a concrete class that implements IRecordBatch
 *   - Strategy sees only IRecordBatch& from aggregator
 */
class IRecordBatch {
public:
  virtual ~IRecordBatch() = default;

  /**
   * @brief Remove all stored data. 
   */
  virtual void Clear() = 0;

  /**
   * @brief Pre-reserve space in internal vectors for 'capacity' records.
   */
  virtual void Reserve(std::size_t capacity) = 0;

  /**
   * @brief Append a new record with raw integer fields:
   *        timestamp, instrument_id, price, size, side, action, order_id.
   */
  virtual void Append(std::uint64_t timestamp,
                      std::uint32_t instrument_id,
                      std::int64_t  price,
                      std::uint32_t size,
                      int side,
                      int action,
                      std::uint64_t order_id) = 0;

  /**
   * @brief Return how many records are currently stored.
   */
  virtual std::size_t Size() const noexcept = 0;

  /**
   * @brief Return capacity for the 'timestamps' array. Batch aggregator uses it
   *        for memory usage checks.
   */
  virtual std::size_t TimestampsCapacity() const noexcept = 0;

  /**
   * @brief Return capacity for the 'instrument_ids' array.
   */
  virtual std::size_t InstrumentIdsCapacity() const noexcept = 0;

  /**
   * @brief Return capacity for the 'prices' array.
   */
  virtual std::size_t PricesCapacity() const noexcept = 0;

  /**
   * @brief Return capacity for the 'sizes' array.
   */
  virtual std::size_t SizesCapacity() const noexcept = 0;

  /**
   * @brief Return capacity for the 'sides' array.
   */
  virtual std::size_t SidesCapacity() const noexcept = 0;

  /**
   * @brief Return capacity for the 'actions' array.
   */
  virtual std::size_t ActionsCapacity() const noexcept = 0;

  /**
   * @brief Return capacity for the 'order_ids' array.
   */
  virtual std::size_t OrderIdsCapacity() const noexcept = 0;
};

} // end namespace batch
} // end namespace interfaces
} // end namespace constellation
