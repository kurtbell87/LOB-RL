#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "interfaces/batch/IRecordBatch.hpp"

/**
 * @file RecordBatch.hpp
 * @brief A concrete SoA-based implementation of IRecordBatch that stores MBO fields:
 *        timestamps, instrument_ids, prices, sizes, sides, actions, order_ids.
 *
 * Batch aggregator can reference it as IRecordBatch, or directly as RecordBatch.
 */

namespace constellation {
namespace interfaces {
namespace batch {

/**
 * @class RecordBatch
 * @brief A structure-of-arrays (SoA) storage implementing IRecordBatch.
 *        Batch aggregator relies on these capacity queries for memory usage checks,
 *        plus Clear, Reserve, Append, etc.
 */
class RecordBatch : public IRecordBatch {
public:
  RecordBatch();
  ~RecordBatch() override;

  // IRecordBatch interface implementations:
  void Clear() override;
  void Reserve(std::size_t capacity) override;
  void Append(std::uint64_t timestamp,
              std::uint32_t instrument_id,
              std::int64_t  price,
              std::uint32_t size,
              int side,
              int action,
              std::uint64_t order_id) override;
  std::size_t Size() const noexcept override;

  std::size_t TimestampsCapacity() const noexcept override;
  std::size_t InstrumentIdsCapacity() const noexcept override;
  std::size_t PricesCapacity() const noexcept override;
  std::size_t SizesCapacity() const noexcept override;
  std::size_t SidesCapacity() const noexcept override;
  std::size_t ActionsCapacity() const noexcept override;
  std::size_t OrderIdsCapacity() const noexcept override;

  // Direct array accessors for Batch bridging (C++ usage).
  const std::uint64_t* Timestamps()     const { return timestamps_.data(); }
  const std::uint32_t* InstrumentIds()  const { return instrument_ids_.data(); }
  const std::int64_t*  Prices()         const { return prices_.data(); }
  const std::uint32_t* Sizes()          const { return sizes_.data(); }
  const int*           Sides()          const { return sides_.data(); }
  const int*           Actions()        const { return actions_.data(); }
  const std::uint64_t* OrderIds()       const { return order_ids_.data(); }

private:
  // SoA vectors
  std::vector<std::uint64_t> timestamps_;
  std::vector<std::uint32_t> instrument_ids_;
  std::vector<std::int64_t>  prices_;
  std::vector<std::uint32_t> sizes_;
  std::vector<int>           sides_;
  std::vector<int>           actions_;
  std::vector<std::uint64_t> order_ids_;
};

} // end namespace batch
} // end namespace interfaces
} // end namespace constellation
