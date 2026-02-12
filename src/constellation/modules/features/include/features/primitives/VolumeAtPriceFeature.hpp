#pragma once

#include <atomic>
#include <string>
#include "features/AbstractFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"
#include "interfaces/orderbook/IInstrumentBook.hpp"

namespace constellation::modules::features::primitives {

/**
 * @brief VolumeAtPriceFeature: queries aggregator->VolumeAtPrice(instrument, config_.price)
 *        The 'price' is stored as int64 nano. The user sets "price" param as a string, we parse stoll.
 */
class VolumeAtPriceFeature
  : public virtual constellation::modules::features::AbstractFeature,
    public virtual constellation::interfaces::features::IConfigurableFeature
{
public:
  struct Config {
    std::uint32_t instrument_id{0};
    constellation::interfaces::orderbook::BookSide side{
      constellation::interfaces::orderbook::BookSide::Bid
    }; // (not strictly used, but kept for older code)
    std::int64_t price{0}; // store raw nano price
  };

  VolumeAtPriceFeature();
  explicit VolumeAtPriceFeature(const Config& cfg);
  ~VolumeAtPriceFeature() override = default;

  double GetValue(const std::string& name) const override;
  bool HasFeature(const std::string& name) const override;
  void SetParam(const std::string& key, const std::string& value) override;

protected:
  void ComputeUpdate(const constellation::interfaces::orderbook::IMarketBookDataSource& source,
                     const constellation::interfaces::orderbook::IMarketView* market) override;

private:
  Config config_;
  std::atomic<std::uint64_t> volume_;
};

} // end namespace
