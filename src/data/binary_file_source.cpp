#include "binary_file_source.h"
#include <fstream>
#include <stdexcept>
#include <cstring>

BinaryFileSource::BinaryFileSource(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Read header (16 bytes): magic(4) + version(u32) + record_count(u32) + instrument_id(u32)
    char magic[4];
    uint32_t version;

    file.read(magic, 4);
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&record_count_), sizeof(record_count_));
    file.read(reinterpret_cast<char*>(&instrument_id_), sizeof(instrument_id_));

    if (!file.good()) {
        throw std::runtime_error("Failed to read header from: " + path);
    }

    if (std::memcmp(magic, "LOBR", 4) != 0) {
        throw std::runtime_error("Invalid magic bytes in: " + path);
    }

    if (version != 1) {
        throw std::runtime_error("Unsupported version in: " + path);
    }

    // Read all records in one bulk read
    records_.resize(record_count_);
    if (record_count_ > 0) {
        file.read(reinterpret_cast<char*>(records_.data()),
                  static_cast<std::streamsize>(record_count_) * sizeof(FlatRecord));
        if (!file.good()) {
            // Truncated file — keep only the fully-read records
            auto bytes_read = file.gcount();
            auto full_records = static_cast<uint32_t>(bytes_read / sizeof(FlatRecord));
            records_.resize(full_records);
            record_count_ = full_records;
        }
    }
}

bool BinaryFileSource::next(Message& msg) {
    if (index_ >= records_.size()) return false;
    msg = convert(records_[index_++]);
    return true;
}

void BinaryFileSource::reset() {
    index_ = 0;
}

Message BinaryFileSource::convert(const FlatRecord& rec) const {
    Message msg;
    msg.ts_ns = rec.ts_ns;
    msg.order_id = rec.order_id;
    msg.price = static_cast<double>(rec.price_raw) / 1e9;
    msg.qty = rec.qty;
    msg.flags = rec.flags;

    // Map action byte ('F' = Fill is treated as Trade)
    switch (rec.action) {
        case 'A': msg.action = Message::Action::Add; break;
        case 'C': msg.action = Message::Action::Cancel; break;
        case 'M': msg.action = Message::Action::Modify; break;
        case 'T': [[fallthrough]];
        case 'F': msg.action = Message::Action::Trade; break;
        default:  msg.action = Message::Action::Add; break;
    }

    // Map side byte ('N' = None is treated as Bid)
    switch (rec.side) {
        case 'B': [[fallthrough]];
        case 'N': msg.side = Message::Side::Bid; break;
        case 'A': msg.side = Message::Side::Ask; break;
        default:  msg.side = Message::Side::Bid; break;
    }

    return msg;
}
