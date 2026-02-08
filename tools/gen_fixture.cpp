// Generates tests/fixtures/test_mes.mbo.dbn.zst for DbnFileSource tests.
//
// Requirements from the test file:
//   - instrument_id = 12345 for matching records
//   - At least 10 matching records after filtering
//   - Contains at least one Fill mapped to Trade
//   - Contains at least one Side::None mapped to Bid
//   - Contains some records with different instrument_id (for filtering tests)
//   - Contains some Clear ('R') and None ('N') actions that should be filtered
//   - All prices on 0.25 tick grid
//   - Timestamps non-decreasing
//   - After processing, book should have valid BBO (bid < ask)
//   - Timestamps in RTH window (13:30-20:00 UTC) for precompute integration
//
// Build: cmake --build . --target gen_fixture
// Run:   ./gen_fixture

#include <databento/dbn_encoder.hpp>
#include <databento/file_stream.hpp>
#include <databento/detail/zstd_stream.hpp>
#include <databento/constants.hpp>
#include <databento/enums.hpp>
#include <databento/record.hpp>
#include <databento/dbn.hpp>

#include <filesystem>
#include <iostream>
#include <chrono>
#include <vector>

static databento::MboMsg make_record(
    uint32_t instrument_id, uint64_t ts_ns, uint64_t order_id,
    int64_t price, uint32_t size,
    databento::Action action, databento::Side side,
    uint8_t flags = 0)
{
    databento::MboMsg mbo{};
    mbo.hd.length = sizeof(databento::MboMsg) / 4;
    mbo.hd.rtype = databento::RType::Mbo;
    mbo.hd.publisher_id = 1;  // GlbxMdp3
    mbo.hd.instrument_id = instrument_id;
    mbo.hd.ts_event = databento::UnixNanos{std::chrono::nanoseconds{ts_ns}};
    mbo.order_id = order_id;
    mbo.price = price;
    mbo.size = size;
    mbo.flags = databento::FlagSet{flags};
    mbo.channel_id = 0;
    mbo.action = action;
    mbo.side = side;
    mbo.ts_recv = databento::UnixNanos{std::chrono::nanoseconds{ts_ns + 100}};
    mbo.ts_in_delta = {};
    mbo.sequence = 0;
    return mbo;
}

int main(int argc, char* argv[]) {
    // Output path: tests/fixtures/test_mes.mbo.dbn.zst
    // The tests/ directory may be read-only, so allow override via command-line arg
    std::string out_path;
    if (argc > 1) {
        out_path = argv[1];
    } else {
        std::filesystem::path src_dir = std::filesystem::path(CMAKE_SOURCE_DIR);
        std::filesystem::path out_dir = src_dir / "tests" / "fixtures";
        std::filesystem::create_directories(out_dir);
        out_path = (out_dir / "test_mes.mbo.dbn.zst").string();
    }

    // Instrument IDs
    constexpr uint32_t MES_ID = 12345;
    constexpr uint32_t OTHER_ID = 99999;

    // Base timestamp: 2024-01-15 13:31:00 UTC (just after RTH open at 13:30)
    // DAY_BASE_NS = 19737 * 24 * 3600 * 1e9 = 1705276800000000000
    constexpr uint64_t DAY_BASE_NS = 19737ULL * 24ULL * 3600ULL * 1000000000ULL;
    constexpr uint64_t RTH_OPEN  = DAY_BASE_NS + 13ULL * 3600ULL * 1000000000ULL + 30ULL * 60ULL * 1000000000ULL;
    uint64_t ts = RTH_OPEN + 60ULL * 1000000000ULL; // 13:31 UTC

    // Price constants (fixed-point 1e-9): MES around $5000
    constexpr int64_t P_4999_25 = 4999250000000LL;
    constexpr int64_t P_4999_50 = 4999500000000LL;
    constexpr int64_t P_4999_75 = 4999750000000LL;
    constexpr int64_t P_5000_00 = 5000000000000LL;
    constexpr int64_t P_5000_25 = 5000250000000LL;
    constexpr int64_t P_5000_50 = 5000500000000LL;
    constexpr int64_t P_5000_75 = 5000750000000LL;
    constexpr int64_t P_5001_00 = 5001000000000LL;

    uint64_t oid = 100;
    uint64_t step = 1000000000ULL; // 1 second between messages

    std::vector<databento::MboMsg> records;

    // == Build the order book with Adds for MES ==
    // Bids: 4999.75, 4999.50, 4999.25
    records.push_back(make_record(MES_ID, ts, oid++, P_4999_75, 10, databento::Action::Add, databento::Side::Bid, 0x80));
    ts += step;
    records.push_back(make_record(MES_ID, ts, oid++, P_4999_50, 20, databento::Action::Add, databento::Side::Bid));
    ts += step;
    records.push_back(make_record(MES_ID, ts, oid++, P_4999_25, 30, databento::Action::Add, databento::Side::Bid));
    ts += step;

    // Asks: 5000.00, 5000.25, 5000.50
    records.push_back(make_record(MES_ID, ts, oid++, P_5000_00, 10, databento::Action::Add, databento::Side::Ask));
    ts += step;
    records.push_back(make_record(MES_ID, ts, oid++, P_5000_25, 20, databento::Action::Add, databento::Side::Ask));
    ts += step;
    records.push_back(make_record(MES_ID, ts, oid++, P_5000_50, 30, databento::Action::Add, databento::Side::Ask));
    ts += step;

    // == Some messages from a DIFFERENT instrument (should be filtered) ==
    records.push_back(make_record(OTHER_ID, ts, oid++, P_5000_75, 5, databento::Action::Add, databento::Side::Bid));
    ts += step;
    records.push_back(make_record(OTHER_ID, ts, oid++, P_5001_00, 5, databento::Action::Add, databento::Side::Ask));
    ts += step;

    // == Clear and None actions for MES (should be skipped) ==
    records.push_back(make_record(MES_ID, ts, oid++, P_4999_75, 10, databento::Action::Clear, databento::Side::None));
    ts += step;

    // == More valid MES orders ==
    records.push_back(make_record(MES_ID, ts, oid++, P_4999_75, 15, databento::Action::Add, databento::Side::Bid));
    ts += step;
    records.push_back(make_record(MES_ID, ts, oid++, P_5000_00, 15, databento::Action::Add, databento::Side::Ask));
    ts += step;

    // == Modify an existing order ==
    records.push_back(make_record(MES_ID, ts, 100, P_4999_75, 5, databento::Action::Modify, databento::Side::Bid));
    ts += step;

    // == Cancel an order ==
    records.push_back(make_record(MES_ID, ts, 101, P_4999_50, 20, databento::Action::Cancel, databento::Side::Bid));
    ts += step;

    // == Trade (regular) ==
    records.push_back(make_record(MES_ID, ts, oid++, P_5000_00, 3, databento::Action::Trade, databento::Side::Ask));
    ts += step;

    // == Fill ('F') — should map to Trade ==
    records.push_back(make_record(MES_ID, ts, oid++, P_4999_75, 2, databento::Action::Fill, databento::Side::Ask));
    ts += step;

    // == Trade with Side::None — should map to Bid ==
    records.push_back(make_record(MES_ID, ts, oid++, P_5000_00, 1, databento::Action::Trade, databento::Side::None));
    ts += step;

    // == A few more adds to ensure >= 10 matching records ==
    records.push_back(make_record(MES_ID, ts, oid++, P_4999_25, 40, databento::Action::Add, databento::Side::Bid));
    ts += step;
    records.push_back(make_record(MES_ID, ts, oid++, P_5000_50, 40, databento::Action::Add, databento::Side::Ask));
    ts += step;

    // == Another record from OTHER instrument ==
    records.push_back(make_record(OTHER_ID, ts, oid++, P_5000_00, 10, databento::Action::Trade, databento::Side::Ask));
    ts += step;

    // Now create the DBN file
    {
        databento::OutFileStream out_file{out_path};
        databento::detail::ZstdCompressStream stream{&out_file};

        databento::Metadata metadata{};
        metadata.version = databento::kDbnVersion;
        metadata.dataset = "GLBX.MDP3";
        metadata.schema = databento::Schema::Mbo;
        metadata.start = databento::UnixNanos{std::chrono::nanoseconds{RTH_OPEN}};
        metadata.end = databento::UnixNanos{std::chrono::nanoseconds{ts}};
        metadata.limit = 0;
        metadata.stype_in = databento::SType::RawSymbol;
        metadata.stype_out = databento::SType::InstrumentId;
        metadata.ts_out = false;
        metadata.symbol_cstr_len = databento::kSymbolCstrLen;
        metadata.symbols = {"MES"};
        metadata.partial = {};
        metadata.not_found = {};
        metadata.mappings = {};

        databento::DbnEncoder encoder{metadata, &stream};

        for (const auto& rec : records) {
            encoder.EncodeRecord(rec);
        }
    }  // streams close, flushing to disk

    std::cout << "Created fixture: " << out_path << std::endl;
    std::cout << "Total records: " << records.size() << std::endl;

    // Count matching records for MES_ID (excluding Clear and None actions)
    int matching = 0;
    for (const auto& r : records) {
        if (r.hd.instrument_id == MES_ID) {
            char a = static_cast<char>(r.action);
            if (a != 'R' && a != 'N') ++matching;
        }
    }
    std::cout << "Matching records (instrument_id=" << MES_ID << ", valid actions): " << matching << std::endl;

    return 0;
}
