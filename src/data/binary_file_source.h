#pragma once
#include "lob/source.h"
#include <string>
#include <vector>
#include <cstdint>

class BinaryFileSource : public IMessageSource {
public:
    explicit BinaryFileSource(const std::string& path);

    bool next(Message& msg) override;
    void reset() override;

    uint32_t record_count() const { return record_count_; }
    uint32_t instrument_id() const { return instrument_id_; }

private:
    // On-disk record layout (36 bytes)
    struct __attribute__((packed)) FlatRecord {
        uint64_t ts_ns;
        uint64_t order_id;
        int64_t  price_raw;
        uint32_t qty;
        uint8_t  action;
        uint8_t  side;
        uint8_t  flags;
        uint8_t  pad;
        uint32_t reserved;
    };
    static_assert(sizeof(FlatRecord) == 36, "FlatRecord must be 36 bytes");

    Message convert(const FlatRecord& rec) const;

    std::vector<FlatRecord> records_;
    uint32_t record_count_ = 0;
    uint32_t instrument_id_ = 0;
    size_t index_ = 0;
};
