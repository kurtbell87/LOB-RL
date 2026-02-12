#pragma once

#include <functional>
#include <memory>
#include "databento/record.hpp"
#include "interfaces/common/InterfaceVersionInfo.hpp"

namespace constellation::interfaces::market_data {

/**
 * @brief Abstract interface for a market data ingestion feed,
 *        producing MBO or general DBN records.
 *
 * Implementations (e.g., DataBentoMboFeed) should:
 *   - Connect to a data source (historical or live).
 *   - Parse DBN records.
 *   - Invoke user callbacks with relevant MBO messages or raw records.
 */
class IIngestionFeed {
public:
  virtual ~IIngestionFeed() = default;

  /**
   * @brief Replaces the old InterfaceVersion with a major/minor approach.
   */
  virtual constellation::interfaces::common::InterfaceVersionInfo GetVersionInfo() const noexcept = 0;

  /**
   * @brief Register a callback to receive MBO (Market By Order) messages.
   *        The callback will be called for each MboMsg encountered.
   */
  virtual void SubscribeMboCallback(
      const std::function<void(const databento::MboMsg&)>& callback) = 0;

  /**
   * @brief (Optional) Register a callback to receive *all* DBN records,
   *        not just MBO. Useful for advanced usage (e.g., symbol mapping).
   */
  virtual void SubscribeRecordCallback(
      const std::function<void(const databento::Record&)>& callback) = 0;

  /**
   * @brief Start streaming or replaying data. This may spawn background threads.
   */
  virtual void Start() = 0;

  /**
   * @brief Gracefully stop the feed. Blocks until all internal threads exit.
   */
  virtual void Stop() = 0;
};

}  // end namespace constellation::modules::market_data
