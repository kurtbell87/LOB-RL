#pragma once

#include "orderbook/commands/IOrderCommand.hpp"
#include "databento/record.hpp"

namespace constellation::modules::orderbook {

/**
 * @brief Example command classes implementing IOrderCommand, each carrying an MboMsg.
 */

class AddOrderCommand : public IOrderCommand {
public:
  explicit AddOrderCommand(const databento::MboMsg& mbo) : mbo_(mbo) {}
  void Execute(LimitOrderBook& lob) override;
  void Execute(MarketBook& book) override;
private:
  databento::MboMsg mbo_;
};

class CancelOrderCommand : public IOrderCommand {
public:
  explicit CancelOrderCommand(const databento::MboMsg& mbo) : mbo_(mbo) {}
  void Execute(LimitOrderBook& lob) override;
  void Execute(MarketBook& book) override;
private:
  databento::MboMsg mbo_;
};

class ModifyOrderCommand : public IOrderCommand {
public:
  explicit ModifyOrderCommand(const databento::MboMsg& mbo) : mbo_(mbo) {}
  void Execute(LimitOrderBook& lob) override;
  void Execute(MarketBook& book) override;
private:
  databento::MboMsg mbo_;
};

class TradeOrderCommand : public IOrderCommand {
public:
  explicit TradeOrderCommand(const databento::MboMsg& mbo) : mbo_(mbo) {}
  void Execute(LimitOrderBook& lob) override;
  void Execute(MarketBook& book) override;
private:
  databento::MboMsg mbo_;
};

class ClearOrderBookCommand : public IOrderCommand {
public:
  explicit ClearOrderBookCommand(const databento::MboMsg& mbo) : mbo_(mbo) {}
  void Execute(LimitOrderBook& lob) override;
  void Execute(MarketBook& book) override;
private:
  databento::MboMsg mbo_;
};

} // end namespace constellation::modules::orderbook
