#pragma once

#include <string>
#include <vector>

#include "replay/ReplayConfig.hpp"
#include "replay/ReplayStats.hpp"

namespace constellation::modules::replay {

/**
 * @brief IReplayEngine defines the abstract interface for
 *        high-throughput DBN file replay engines.
 *
 *        Implementations should provide parallel read + parse of
 *        DBN files, distribution of MBO messages by instrument,
 *        and updating a global MarketBook or set of LOBs.
 */
class IReplayEngine {
public:
  virtual ~IReplayEngine() = default;

  /**
   * @brief Initialize with a ReplayConfig (threading, buffers, etc.).
   *        May allocate resources and set up internal data structures.
   */
  virtual void Initialize(const ReplayConfig& config) = 0;

  /**
   * @brief Process multiple DBN files in parallel or sequentially
   *        (depending on the implementation). This should result
   *        in the underlying MarketBook or LOB structures being
   *        updated accordingly.
   */
  virtual void ProcessFiles(const std::vector<std::string>& dbn_files) = 0;

  /**
   * @brief Process a single DBN file. Typically used by higher-level
   *        code if it wants to handle the file listing itself.
   */
  virtual void ProcessSingleFile(const std::string& file_path) = 0;

  /**
   * @brief Retrieve current performance statistics (throughput, etc.).
   */
  virtual ReplayStats GetStats() const = 0;

  /**
   * @brief Reset performance counters or other stats to zero.
   */
  virtual void ResetStats() = 0;
};

} // end namespace constellation::modules::replay
