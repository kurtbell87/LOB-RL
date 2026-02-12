#pragma once

#include <cstdint>

namespace constellation::modules::replay {

/**
 * @brief ReplayStats holds performance counters for replay operations.
 */
struct ReplayStats {
  std::uint64_t total_messages{0};
  std::uint64_t total_files{0};
  double elapsed_seconds{0.0};
  double messages_per_second{0.0};
};

} // end namespace constellation::modules::replay
