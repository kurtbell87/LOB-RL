#include "orderbook/commands/OrderCommands.hpp"
#include "orderbook/LimitOrderBook.hpp"
#include "orderbook/MarketBook.hpp"

namespace constellation::modules::orderbook {

// ---------------------- AddOrderCommand
void AddOrderCommand::Execute(LimitOrderBook& lob) {
  lob.OnMboUpdate(mbo_);
}
void AddOrderCommand::Execute(MarketBook& book) {
  book.OnMboUpdate(mbo_);
}

// ---------------------- CancelOrderCommand
void CancelOrderCommand::Execute(LimitOrderBook& lob) {
  lob.OnMboUpdate(mbo_);
}
void CancelOrderCommand::Execute(MarketBook& book) {
  book.OnMboUpdate(mbo_);
}

// ---------------------- ModifyOrderCommand
void ModifyOrderCommand::Execute(LimitOrderBook& lob) {
  lob.OnMboUpdate(mbo_);
}
void ModifyOrderCommand::Execute(MarketBook& book) {
  book.OnMboUpdate(mbo_);
}

// ---------------------- TradeOrderCommand
void TradeOrderCommand::Execute(LimitOrderBook& lob) {
  lob.OnMboUpdate(mbo_);
}
void TradeOrderCommand::Execute(MarketBook& book) {
  book.OnMboUpdate(mbo_);
}

// ---------------------- ClearOrderBookCommand
void ClearOrderBookCommand::Execute(LimitOrderBook& lob) {
  lob.OnMboUpdate(mbo_);
}
void ClearOrderBookCommand::Execute(MarketBook& book) {
  book.OnMboUpdate(mbo_);
}

} // end namespace constellation::modules::orderbook
