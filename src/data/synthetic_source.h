#pragma once
#include "lob/source.h"
#include <vector>
#include <cstdint>

class SyntheticSource : public IMessageSource {
public:
    explicit SyntheticSource(uint64_t seed = 12345);

    bool next(Message& msg) override;
    void reset() override;

private:
    void generate();

    uint64_t seed_;
    std::vector<Message> messages_;
    size_t index_ = 0;
};
