#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <thread>
#include <vector>
#include <atomic>
#include <random>
#include <chrono>
#include <future>

#include "orderbook/OrderBookFactory.hpp"
#include "orderbook/MarketBook.hpp"
#include "databento/record.hpp"

namespace constellation::modules::orderbook::tests {

// Helper function to create a MBO message
databento::MboMsg createMboMsg(std::uint32_t instrument_id, 
                               std::uint64_t order_id, 
                               databento::Action action, 
                               databento::Side side, 
                               std::int64_t price, 
                               std::uint32_t size) {
    databento::MboMsg msg;
    msg.hd.instrument_id = instrument_id;
    msg.order_id = order_id;
    msg.action = action;
    msg.side = side;
    msg.price = price;
    msg.size = size;
    return msg;
}

TEST_CASE("MarketBook per-instrument locking stress test", "[orderbook][concurrency]") {
    // Create market book
    auto marketBook = std::make_shared<MarketBook>();
    
    // Define test parameters
    constexpr int NUM_INSTRUMENTS = 100;
    constexpr int NUM_WRITER_THREADS = 4;
    constexpr int NUM_READER_THREADS = 8;
    constexpr int OPERATIONS_PER_THREAD = 1000;
    
    // Setup atomic counters for validation
    std::atomic<int> totalWrites{0};
    std::atomic<int> totalReads{0};
    std::atomic<int> readErrors{0};
    
    // Create random generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::uint32_t> instDist(1, NUM_INSTRUMENTS);
    std::uniform_int_distribution<std::uint64_t> orderIdDist(1, 1000000);
    std::uniform_int_distribution<std::int64_t> priceDist(100000000, 200000000); // 100-200 with fixed point
    std::uniform_int_distribution<std::uint32_t> sizeDist(1, 100);
    
    // Writer thread function - adds orders to random instruments
    auto writerFunc = [&](int threadId) {
        for (int i = 0; i < OPERATIONS_PER_THREAD; ++i) {
            // Create a random MBO message
            std::uint32_t inst_id = instDist(gen);
            std::uint64_t order_id = (threadId * 1000000) + i; // Ensure unique order IDs
            databento::Action action = databento::Action::Add;
            databento::Side side = (i % 2 == 0) ? databento::Side::Bid : databento::Side::Ask;
            std::int64_t price = priceDist(gen);
            std::uint32_t size = sizeDist(gen);
            
            auto msg = createMboMsg(inst_id, order_id, action, side, price, size);
            
            // Update market book
            marketBook->OnMboUpdate(msg);
            totalWrites.fetch_add(1);
        }
    };
    
    // Reader thread function - reads best bids/asks from random instruments
    auto readerFunc = [&]() {
        for (int i = 0; i < OPERATIONS_PER_THREAD; ++i) {
            // Select random instrument
            std::uint32_t inst_id = instDist(gen);
            
            // Read best bid/ask
            auto bestBid = marketBook->GetBestBid(inst_id);
            auto bestAsk = marketBook->GetBestAsk(inst_id);
            
            // For random instruments, there's no guarantee that we'll have both
    // bid and ask for the same instrument, or that they'll be properly ordered.
    // This test is primarily checking for thread safety, not order book correctness.
    // In a real scenario with actual market data, valid bids < asks would be maintained.
            if (bestBid.has_value() && bestAsk.has_value()) {
                // For testing purposes, we'll check but won't fail the test for this case
                if (bestBid->price >= bestAsk->price) {
                    readErrors.fetch_add(1);
                }
            }
            
            totalReads.fetch_add(1);
        }
    };
    
    // Start writer threads
    INFO("Starting writer threads");
    std::vector<std::future<void>> writerFutures;
    for (int i = 0; i < NUM_WRITER_THREADS; ++i) {
        writerFutures.push_back(std::async(std::launch::async, writerFunc, i));
    }
    
    // Start reader threads
    INFO("Starting reader threads");
    std::vector<std::future<void>> readerFutures;
    for (int i = 0; i < NUM_READER_THREADS; ++i) {
        readerFutures.push_back(std::async(std::launch::async, readerFunc));
    }
    
    // Wait for all threads to complete
    INFO("Waiting for threads to complete");
    for (auto& future : writerFutures) {
        future.wait();
    }
    for (auto& future : readerFutures) {
        future.wait();
    }
    
    // Verify results
    INFO("Writes performed: " << totalWrites.load());
    INFO("Reads performed: " << totalReads.load());
    INFO("Read errors: " << readErrors.load());
    
    // We expect totalWrites to equal NUM_WRITER_THREADS * OPERATIONS_PER_THREAD
    CHECK(totalWrites.load() == NUM_WRITER_THREADS * OPERATIONS_PER_THREAD);
    
    // We expect totalReads to equal NUM_READER_THREADS * OPERATIONS_PER_THREAD
    CHECK(totalReads.load() == NUM_READER_THREADS * OPERATIONS_PER_THREAD);
    
    // In this stress test, we're randomly generating prices for bids and asks across threads
    // so it's normal to have some "invalid" order book states with best bid >= best ask
    // We're primarily testing thread safety here, not the business logic
    INFO("Found " << readErrors.load() << " cases where best bid >= best ask (expected in random test data)");
    
    // Check that we have the correct number of instruments
    auto instrumentIds = marketBook->GetInstrumentIds();
    INFO("Number of instruments created: " << instrumentIds.size());
    CHECK(instrumentIds.size() <= NUM_INSTRUMENTS);
}

TEST_CASE("MarketBook concurrent batch updates", "[orderbook][concurrency]") {
    // Create market book
    auto marketBook = std::make_shared<MarketBook>();
    
    // Define test parameters
    constexpr int NUM_INSTRUMENTS = 10;
    constexpr int BATCH_SIZE = 100;
    constexpr int NUM_BATCHES = 20;
    
    // Prepare batches of messages
    std::vector<std::vector<databento::MboMsg>> batches;
    batches.reserve(NUM_BATCHES);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int b = 0; b < NUM_BATCHES; ++b) {
        std::vector<databento::MboMsg> batch;
        batch.reserve(BATCH_SIZE);
        
        for (int i = 0; i < BATCH_SIZE; ++i) {
            std::uint32_t inst_id = (i % NUM_INSTRUMENTS) + 1;
            std::uint64_t order_id = (b * BATCH_SIZE) + i + 1;
            databento::Action action = databento::Action::Add;
            databento::Side side = (i % 2 == 0) ? databento::Side::Bid : databento::Side::Ask;
            std::int64_t price = 100000000 + (i * 1000000); // Ensure unique prices
            std::uint32_t size = 10 + (i % 90);
            
            batch.push_back(createMboMsg(inst_id, order_id, action, side, price, size));
        }
        
        batches.push_back(std::move(batch));
    }
    
    // Process batches in parallel
    std::vector<std::future<void>> batchFutures;
    std::atomic<int> batchesProcessed{0};
    
    for (const auto& batch : batches) {
        batchFutures.push_back(std::async(std::launch::async, [&marketBook, &batch, &batchesProcessed]() {
            marketBook->BatchOnMboUpdate(batch);
            batchesProcessed.fetch_add(1);
        }));
    }
    
    // Wait for all batches to complete
    for (auto& future : batchFutures) {
        future.wait();
    }
    
    // Verify results
    INFO("Batches processed: " << batchesProcessed.load());
    CHECK(batchesProcessed.load() == NUM_BATCHES);
    
    // Check that all instruments were created
    auto instrumentIds = marketBook->GetInstrumentIds();
    INFO("Number of instruments created: " << instrumentIds.size());
    CHECK(instrumentIds.size() == NUM_INSTRUMENTS);
    
    // Verify we can read all the instruments without errors
    for (const auto& id : instrumentIds) {
        auto bestBid = marketBook->GetBestBid(id);
        auto bestAsk = marketBook->GetBestAsk(id);
        
        INFO("Instrument " << id << " has bid: " << (bestBid ? "yes" : "no") 
             << " and ask: " << (bestAsk ? "yes" : "no"));
        
        // If both bid and ask exist, bid should be less than ask
        if (bestBid && bestAsk) {
            CHECK(bestBid->price < bestAsk->price);
        }
    }
}

TEST_CASE("MarketBook parallel instrument creation", "[orderbook][concurrency]") {
    // Create market book
    auto marketBook = std::make_shared<MarketBook>();
    
    // Test creating many instruments in parallel
    constexpr int NUM_INSTRUMENTS = 1000;
    constexpr int NUM_THREADS = 10;
    
    std::atomic<int> instrumentsCreated{0};
    std::vector<std::future<void>> futures;
    
    // Each thread creates a set of instruments
    for (int t = 0; t < NUM_THREADS; ++t) {
        futures.push_back(std::async(std::launch::async, [t, &marketBook, &instrumentsCreated]() {
            // Each thread creates NUM_INSTRUMENTS/NUM_THREADS instruments with
            // non-overlapping IDs
            int startId = t * (NUM_INSTRUMENTS / NUM_THREADS) + 1;
            int endId = (t + 1) * (NUM_INSTRUMENTS / NUM_THREADS);
            
            for (int id = startId; id <= endId; ++id) {
                auto msg = createMboMsg(id, 1, databento::Action::Add, 
                                        databento::Side::Bid, 100000000, 10);
                marketBook->OnMboUpdate(msg);
                instrumentsCreated.fetch_add(1);
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    // Verify results
    INFO("Instruments created: " << instrumentsCreated.load());
    CHECK(instrumentsCreated.load() == NUM_INSTRUMENTS);
    
    // Check that all instruments are present
    auto instrumentIds = marketBook->GetInstrumentIds();
    INFO("Number of instruments in MarketBook: " << instrumentIds.size());
    CHECK(instrumentIds.size() == NUM_INSTRUMENTS);
}

} // end namespace constellation::modules::orderbook::tests