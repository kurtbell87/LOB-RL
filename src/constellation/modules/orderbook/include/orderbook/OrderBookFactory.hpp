#pragma once

#include <memory>
#include <cstdint>
#include "interfaces/orderbook/IInstrumentBook.hpp"
#include "interfaces/orderbook/IMarketView.hpp"
#include "interfaces/orderbook/IMarketBook.hpp"
#include "interfaces/logging/ILogger.hpp"

namespace constellation::modules::orderbook {

/**
 * @brief Factory functions for constructing orderbook module objects
 *        but exposing them only as interfaces (IInstrumentBook, IMarketView, etc.).
 */

/**
 * @brief Create a new LimitOrderBook that implements IInstrumentBook.
 * @param instrument_id The numeric instrument identifier.
 * @param logger An optional ILogger for debugging or trace logs.
 */
std::shared_ptr<constellation::interfaces::orderbook::IInstrumentBook>
CreateLimitOrderBook(std::uint32_t instrument_id,
    std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr);

/**
 * @brief Create a new MarketBook that implements both IMarketView and
 *        IMarketBookDataSource (accessible through a dynamic_cast if needed).
 * @param logger An optional ILogger for logging.
 * @return An IMarketView interface pointer to the created MarketBook
 */
std::shared_ptr<constellation::interfaces::orderbook::IMarketView>
CreateMarketBook(std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr);

/**
 * @brief Create a new MarketBook that implements IMarketBook interface.
 * @param logger An optional ILogger for logging.
 * @return An IMarketBook interface pointer to the created MarketBook
 */
std::shared_ptr<constellation::interfaces::orderbook::IMarketBook>
CreateIMarketBook(std::shared_ptr<constellation::interfaces::logging::ILogger> logger = nullptr);

} // end namespace constellation::modules::orderbook
