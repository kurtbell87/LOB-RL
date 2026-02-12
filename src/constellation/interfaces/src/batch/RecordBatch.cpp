#include "interfaces/batch/RecordBatch.hpp"

namespace constellation {
namespace interfaces {
namespace batch {

RecordBatch::RecordBatch() = default;
RecordBatch::~RecordBatch() = default;

void RecordBatch::Clear() {
  // sets the stored size to 0 by clearing each vector
  timestamps_.clear();
  instrument_ids_.clear();
  prices_.clear();
  sizes_.clear();
  sides_.clear();
  actions_.clear();
  order_ids_.clear();
}

void RecordBatch::Reserve(std::size_t capacity) {
  timestamps_.reserve(capacity);
  instrument_ids_.reserve(capacity);
  prices_.reserve(capacity);
  sizes_.reserve(capacity);
  sides_.reserve(capacity);
  actions_.reserve(capacity);
  order_ids_.reserve(capacity);
}

void RecordBatch::Append(std::uint64_t timestamp,
                         std::uint32_t instrument_id,
                         std::int64_t  price,
                         std::uint32_t size,
                         int side,
                         int action,
                         std::uint64_t order_id)
{
  timestamps_.push_back(timestamp);
  instrument_ids_.push_back(instrument_id);
  prices_.push_back(price);
  sizes_.push_back(size);
  sides_.push_back(side);
  actions_.push_back(action);
  order_ids_.push_back(order_id);
}

std::size_t RecordBatch::Size() const noexcept {
  return timestamps_.size();
}

std::size_t RecordBatch::TimestampsCapacity() const noexcept {
  return timestamps_.capacity();
}
std::size_t RecordBatch::InstrumentIdsCapacity() const noexcept {
  return instrument_ids_.capacity();
}
std::size_t RecordBatch::PricesCapacity() const noexcept {
  return prices_.capacity();
}
std::size_t RecordBatch::SizesCapacity() const noexcept {
  return sizes_.capacity();
}
std::size_t RecordBatch::SidesCapacity() const noexcept {
  return sides_.capacity();
}
std::size_t RecordBatch::ActionsCapacity() const noexcept {
  return actions_.capacity();
}
std::size_t RecordBatch::OrderIdsCapacity() const noexcept {
  return order_ids_.capacity();
}

} // end namespace batch
} // end namespace interfaces
} // end namespace constellation
