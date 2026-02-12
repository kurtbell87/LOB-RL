#pragma once

#include <atomic>
#include <string>
#include "features/AbstractFeature.hpp"
#include "interfaces/features/IConfigurableFeature.hpp"
#include "interfaces/orderbook/IInstrumentBook.hpp"

namespace constellation::modules::features::primitives {

class MicroDepthFeature
  : public virtual constellation::modules::features::AbstractFeature,
    public virtual constellation::interfaces::features::IConfigurableFeature
{
public:
  struct Config {
    std::uint32_t instrument_id{0};
    constellation::interfaces::orderbook::BookSide side{
      constellation::interfaces::orderbook::BookSide::Bid
    };
    std::size_t depth_index{0}; // 0=best
  };

  MicroDepthFeature();
  explicit MicroDepthFeature(const Config& cfg);
  ~MicroDepthFeature() override = default;

  double GetValue(const std::string& name) const override;
  bool HasFeature(const std::string& name) const override;
  void SetParam(const std::string& key, const std::string& value) override;

protected:
  void ComputeUpdate(const constellation::interfaces::orderbook::IMarketBookDataSource& source,
                     const constellation::interfaces::orderbook::IMarketView* market) override;

private:
  Config config_;
  std::atomic<double> price_;           // real currency
  std::atomic<std::uint64_t> size_;     // just store size as raw
};

} // end namespace
