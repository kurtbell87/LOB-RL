#pragma once

#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <unordered_map>

#include "replay/BatchAggregator.hpp"  // <-- Defines BatchAggregator + BatchAggregatorConfig
#include "interfaces/strategy/IStrategy.hpp"  // IStrategy is in constellation::interfaces::strategy
#include "interfaces/orders/IOrdersEngine.hpp"
#include "interfaces/orders/IOrderEvents.hpp"
#include "interfaces/orders/IOrderModels.hpp"
#include "interfaces/logging/ILogger.hpp"
#include "interfaces/features/MultiInstrumentFeatureManager.hpp"  // Phase 4: Multi-instrument feature support

namespace constellation {
namespace applications {
namespace replay {

/**
 * @brief A record of a fill event for Batch backtests, including chunk-based timestamp.
 *
 * Refactored to store fill_price as int64_t. 
 * If you are using 1 = 1e-9 scale, then a fill_price of 123456789 would mean 0.123456789.
 */
struct FillRecord {
  std::uint64_t timestamp{0};       // Batch aggregator chunk-based time
  std::uint64_t order_id{0};
  std::uint32_t instrument_id{0};
  std::int64_t  fill_price{0};      // int64-based price at nano scale
  std::uint32_t fill_qty{0};
  bool          is_buy{true};
};

/**
 * @brief IOrderEvents that captures fill/done events in a vector. 
 *        Also has a method for Batch aggregator to pass chunk-based timestamps.
 *
 * Refactored so OnOrderFillWithTimestamp now takes int64_t for price. 
 */
class BacktestFillLogger : public constellation::interfaces::orders::IOrderEvents {
public:
  BacktestFillLogger();
  ~BacktestFillLogger() override;

  // Legacy or older usage:
  void OnOrderFilled(std::uint64_t order_id,
                     std::int64_t fill_price,
                     std::uint32_t fill_qty) override;

  void OnOrderDone(std::uint64_t order_id,
                   constellation::interfaces::orders::OrderFinalState final_state) override;

  /**
   * @brief Batch aggregator calls this with chunk-based timestamp.
   *        Price is now int64 at nano scale.
   */
  void OnOrderFillWithTimestamp(std::uint64_t timestamp,
                                std::uint64_t order_id,
                                std::uint32_t instrument_id,
                                constellation::interfaces::orders::OrderSide side,
                                std::int64_t fill_price,
                                std::uint32_t fill_qty) override;

  /**
   * @brief (Legacy) For older code that sets side/timestamp separately.
   */
  void RecordTimestampAndSide(std::uint64_t order_id,
                              bool is_buy,
                              std::uint32_t instr_id,
                              std::uint64_t ts);

  /**
   * @brief Return the vector of fill events.
   */
  std::vector<FillRecord> GetFillRecords() const;

  /**
   * @brief Clear all captured fill logs (and any internal mapping).
   */
  void Clear();

private:
  mutable std::mutex mtx_;
  struct OrderContext {
    bool is_buy{true};
    std::uint32_t instrument_id{0};
    std::uint64_t timestamp{0};
  };
  std::unordered_map<std::uint64_t, OrderContext> order_map_;

  std::vector<FillRecord> fill_log_;
};

/**
 * @class BatchBacktestEngine
 * @brief High-level orchestrator that configures:
 *        - BatchAggregator (chunk-based ingestion)
 *        - An OrdersEngine for fills
 *        - A fill logger
 *        - A user Batch strategy (C++ or Python)
 *        - Phase 4: Multi-instrument feature calculation
 *
 * M4: Adds capturing of fill logs for final retrieval in Python or C++ analysis.
 * Phase 4: Adds support for multi-instrument features and calculations.
 * Refactored to keep fill prices as int64 throughout.
 */
class BatchBacktestEngine {
public:
  BatchBacktestEngine();
  ~BatchBacktestEngine();

  /**
   * @brief Set aggregator config. Must be called before running any process call.
   */
  void SetAggregatorConfig(const BatchAggregatorConfig& config);

  /**
   * @brief Provide a Batch chunk-based strategy for aggregator calls (IStrategy).
   */
  void SetStrategy(const std::shared_ptr<constellation::interfaces::strategy::IStrategy>& strategy);

  /**
   * @brief Replay multiple DBN files in Batch chunk mode.
   */
  void ProcessFiles(const std::vector<std::string>& dbn_files);

  /**
   * @brief Single file convenience wrapper.
   */
  void ProcessSingleFile(const std::string& dbn_file);

  /**
   * @brief Return the captured fill events for user analysis.
   */
  std::vector<FillRecord> GetFills() const;

  /**
   * @brief Access the underlying OrdersEngine if you want direct place/modify/cancel externally.
   */
  std::shared_ptr<constellation::interfaces::orders::IOrdersEngine> GetOrdersEngine() const;

  /**
   * @brief Phase 4: Set a feature manager to calculate features across multiple instruments.
   * @param feature_manager A MultiInstrumentFeatureManager to calculate and provide features
   */
  void SetFeatureManager(
      const std::shared_ptr<constellation::interfaces::features::MultiInstrumentFeatureManager>& feature_manager);

  /**
   * @brief Phase 4: Get the current feature manager if set.
   * @return The current feature manager or nullptr if not set
   */
  std::shared_ptr<constellation::interfaces::features::MultiInstrumentFeatureManager> GetFeatureManager() const;

  /**
   * @brief Reset aggregator stats and fill logs, allowing re-use of this engine instance.
   *        Optional convenience method if your code wants to run multiple scenarios.
   */
  void ResetStats();
  
  /**
   * @brief Phase 5: Get a read-only view of the market state for Python strategies.
   * @return A shared_ptr to an IMarketView interface for querying order book state
   */
  std::shared_ptr<constellation::interfaces::orderbook::IMarketView> GetMarketView() const;

private:
  std::shared_ptr<BatchAggregator> aggregator_;
  std::shared_ptr<constellation::interfaces::orders::IOrdersEngine> ordersEngine_;
  std::shared_ptr<BacktestFillLogger> fillLogger_;
  std::shared_ptr<constellation::interfaces::features::MultiInstrumentFeatureManager> featureManager_;
};

} // end namespace replay
} // end namespace applications
} // end namespace constellation
