#pragma once

#include <cstdint>
#include <string>
#include <memory>

#include "interfaces/logging/ILogger.hpp"

namespace constellation::modules::replay {

/**
 * @brief ReplayConfig holds configuration parameters for parallel replay.
 */
struct ReplayConfig {
  /**
   * @brief Number of worker threads. If 0, auto-detect based on hardware_concurrency.
   */
  std::uint32_t num_threads{0};

  /**
   * @brief Batch size for chunked reading from DBN files. The engine will read
   *        up to 'batch_size' messages at a time before distributing them.
   */
  std::uint32_t batch_size{20000};

  /**
   * @brief Input file buffer size in bytes for each file stream. A larger buffer
   *        can reduce I/O overhead. (Used by ifstream or dbn-file reading.)
   */
  std::uint32_t buffer_size{1 << 20}; // 1 MB

  /**
   * @brief Maximum number of instruments expected. This can help with
   *        pre-allocation in some data structures.
   */
  std::uint32_t max_instruments{64};

  /**
   * @brief Expected average messages per instrument (for pre-allocation).
   */
  std::uint32_t msgs_per_instrument{20000};

  /**
   * @brief Optional logger instance. If null, no logs are emitted.
   */
  std::shared_ptr<constellation::interfaces::logging::ILogger> logger{nullptr};

  /**
   * @brief A path to a JSON (or other) symbology file for advanced symbol mapping
   *        if needed. This is optional. For simple usage, can be empty.
   */
  std::string symbology_file;
};

} // end namespace constellation::modules::replay
