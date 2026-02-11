#pragma once

#include "dbn_message_map.h"
#include "lob/source.h"
#include <string>
#include <cstdint>
#include <memory>

class DbnFileSource : public IMessageSource {
public:
    DbnFileSource(const std::string& path, uint32_t instrument_id = 0);
    ~DbnFileSource() override;

    bool next(Message& msg) override;
    void reset() override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
