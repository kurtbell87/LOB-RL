#include <databento/dbn_file_store.hpp>
#include <databento/record.hpp>
#include "dbn_file_source.h"
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <vector>

// Mapping from real databento::MboMsg to our Message type.
// Uses shared map_action()/map_side() from dbn_message_map.h,
// but handles type differences (UnixNanos, FlagSet) in the real databento types.
static bool convert_mbo(const databento::MboMsg& mbo, Message& msg,
                        uint32_t instrument_id) {
    if (instrument_id != 0 && mbo.hd.instrument_id != instrument_id) {
        return false;
    }

    if (!map_action(static_cast<char>(mbo.action), msg.action)) {
        return false;
    }
    msg.side = map_side(static_cast<char>(mbo.side));

    msg.price = static_cast<double>(mbo.price) / 1e9;
    msg.order_id = mbo.order_id;
    msg.qty = mbo.size;
    msg.ts_ns = mbo.hd.ts_event.time_since_epoch().count();
    msg.flags = mbo.flags.Raw();

    return true;
}

// ── Legacy .bin format support ─────────────────────────────────────────
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

static Message convert_flat(const FlatRecord& rec) {
    Message msg;
    msg.ts_ns = rec.ts_ns;
    msg.order_id = rec.order_id;
    msg.price = static_cast<double>(rec.price_raw) / 1e9;
    msg.qty = rec.qty;
    msg.flags = rec.flags;

    if (!map_action(static_cast<char>(rec.action), msg.action)) {
        msg.action = Message::Action::Add;  // Legacy default for unknown actions
    }
    msg.side = map_side(static_cast<char>(rec.side));

    return msg;
}

static constexpr char BIN_MAGIC[4] = {'L', 'O', 'B', 'R'};

static bool is_bin_format(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    char magic[4] = {};
    file.read(magic, 4);
    return file.good() && std::memcmp(magic, BIN_MAGIC, 4) == 0;
}

// ── Impl with enum-based dispatch ──────────────────────────────────────
struct DbnFileSource::Impl {
    enum class Format { Dbn, Bin };
    Format format;
    std::string path;
    uint32_t instrument_id;

    // DBN state
    std::unique_ptr<databento::DbnFileStore> store;

    // BIN state
    std::vector<FlatRecord> records;
    size_t index = 0;

    Impl(const std::string& p, uint32_t inst_id, Format fmt)
        : format(fmt), path(p), instrument_id(inst_id) {
        if (format == Format::Bin) {
            load_bin();
        } else {
            store = std::make_unique<databento::DbnFileStore>(p);
        }
    }

    void load_bin() {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + path);
        }

        char magic[4];
        uint32_t version, record_count, inst_id;
        file.read(magic, 4);
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        file.read(reinterpret_cast<char*>(&record_count), sizeof(record_count));
        file.read(reinterpret_cast<char*>(&inst_id), sizeof(inst_id));

        if (!file.good() || std::memcmp(magic, BIN_MAGIC, 4) != 0) {
            throw std::runtime_error("Invalid binary file: " + path);
        }

        records.resize(record_count);
        if (record_count > 0) {
            file.read(reinterpret_cast<char*>(records.data()),
                      static_cast<std::streamsize>(record_count) * sizeof(FlatRecord));
            if (!file.good()) {
                auto bytes_read = file.gcount();
                auto full_records = static_cast<uint32_t>(bytes_read / sizeof(FlatRecord));
                records.resize(full_records);
            }
        }
    }

    bool next(Message& msg) {
        if (format == Format::Bin) {
            if (index >= records.size()) return false;
            msg = convert_flat(records[index++]);
            return true;
        } else {
            while (true) {
                const databento::Record* rec = store->NextRecord();
                if (!rec) return false;
                const auto* mbo = rec->GetIf<databento::MboMsg>();
                if (!mbo) continue;
                if (convert_mbo(*mbo, msg, instrument_id)) return true;
            }
        }
    }

    void reset() {
        if (format == Format::Bin) {
            index = 0;
        } else {
            store = std::make_unique<databento::DbnFileStore>(path);
        }
    }
};

DbnFileSource::DbnFileSource(const std::string& path, uint32_t instrument_id) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    auto fmt = is_bin_format(path) ? Impl::Format::Bin : Impl::Format::Dbn;
    impl_ = std::make_unique<Impl>(path, instrument_id, fmt);
}

DbnFileSource::~DbnFileSource() = default;

bool DbnFileSource::next(Message& msg) {
    return impl_->next(msg);
}

void DbnFileSource::reset() {
    impl_->reset();
}
