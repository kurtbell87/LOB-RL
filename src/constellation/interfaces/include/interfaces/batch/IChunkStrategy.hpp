#pragma once

#include <cstdint>
#include <string>
#include "interfaces/batch/IRecordBatch.hpp"

/**
 * @file IChunkStrategy.hpp
 * @brief Declares the core Batch strategy interface that processes entire
 *        record batches (chunks) at a time, ensuring no look-ahead bias
 *        and minimal overhead for Python bridging.
 *
 * ### Chunk Flow Diagram
 * \dot
 * digraph BatchChunkFlow {
 *   rankdir="LR";
 *   node [shape=rectangle, fontsize=10];
 *   Aggregator -> IRecordBatch [label="Populate chunk"];
 *   IRecordBatch -> BatchStrategy [label="OnDataChunk(*batch)"];
 *   BatchStrategy -> OrdersEngine [label="(optional) place orders"];
 * }
 * \enddot
 *
 * ### Sequence of Operations
 *  1. Batch aggregator reads a set of MBO messages into an IRecordBatch.
 *  2. Once the chunk is complete (time T), aggregator calls `OnDataChunk(...)`.
 *  3. Strategy processes the chunk, possibly referencing aggregator states
 *     (e.g. MarketBook) or placing orders.
 *  4. The aggregator then proceeds to the next chunk [T+delta, ...].
 *
 * ### Concurrency Model
 *  - Single-writer concurrency for IRecordBatch: aggregator is the only writer.
 *  - Strategy sees a fully built (finalized) batch, with no partial data.
 *  - No look-ahead: The strategy cannot see data from future timestamps.
 *
 * ### Python/C++ Integration
 *  - A Python-based implementation might hold a `PyObject` or pass the data
 *    to a PyTorch tensor or NumPy array, executing user-defined logic in Python.
 */

namespace constellation {
namespace interfaces {
namespace batch {

/**
 * @class IChunkStrategy
 * @brief Batch strategy interface for chunk-level ingestion of market data.
 *
 * By processing entire chunks at a time, the strategy avoids per-message overhead
 * and can leverage Batch-friendly data structures (e.g., Tensors).
 */
class IChunkStrategy {
public:
  virtual ~IChunkStrategy() = default;

  /**
   * @brief Called once a complete record batch has been applied to the aggregator.
   *        The aggregator ensures no future or out-of-order data is present.
   *
   * @param recordBatch A reference to the Batch record batch containing [T0..T1] data.
   *                    The batch is fully built and finalized. The strategy must
   *                    not modify it.
   *
   * @note Single-threaded call after aggregator finishes chunk ingestion.
   *       The strategy can place new orders (via a separate OrdersEngine reference)
   *       or gather features for ML/RL.
   */
  virtual void OnDataChunk(const IRecordBatch& recordBatch) = 0;

  /**
   * @brief Optional callback for time-based events or scheduled tasks.
   *        Aggregators or orchestrators might call this every N seconds.
   */
  virtual void OnTimeEvent(std::uint64_t currentTimestamp) = 0;

  /**
   * @brief Shutdown hook for gracefully stopping or releasing resources
   *        in Batch strategies.
   */
  virtual void Shutdown() = 0;
};

} // end namespace batch
} // end namespace interfaces
} // end namespace constellation
