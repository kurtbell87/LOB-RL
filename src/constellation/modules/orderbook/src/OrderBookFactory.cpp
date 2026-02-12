#include "orderbook/OrderBookFactory.hpp"

#include "orderbook/LimitOrderBook.hpp"
#include "orderbook/MarketBook.hpp"

namespace constellation::modules::orderbook {

std::shared_ptr<constellation::interfaces::orderbook::IInstrumentBook>
CreateLimitOrderBook(std::uint32_t instrument_id,
    std::shared_ptr<constellation::interfaces::logging::ILogger> logger)
{
  auto lob = std::make_shared<LimitOrderBook>(instrument_id, logger);
  return std::static_pointer_cast<constellation::interfaces::orderbook::IInstrumentBook>(lob);
}

std::shared_ptr<constellation::interfaces::orderbook::IMarketView>
CreateMarketBook(std::shared_ptr<constellation::interfaces::logging::ILogger> logger)
{
  // Create a new MarketBook instance (no longer a singleton)
  auto market_book = std::make_shared<MarketBook>(logger);
  return std::static_pointer_cast<constellation::interfaces::orderbook::IMarketView>(market_book);
}

std::shared_ptr<constellation::interfaces::orderbook::IMarketBook>
CreateIMarketBook(std::shared_ptr<constellation::interfaces::logging::ILogger> logger)
{
  // Create a new MarketBook instance
  auto market_book = std::make_shared<MarketBook>(logger);
  return market_book;
}

} // end namespace constellation::modules::orderbook
