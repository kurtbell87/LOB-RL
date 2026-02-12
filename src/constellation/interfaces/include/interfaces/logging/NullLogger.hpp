#pragma once

#include "interfaces/logging/ILogger.hpp"

#include <mutex>
#include <vector>
#include <string>
#include <cstdarg>
#include <cstdio>

/**
 * @brief A no-op logger that silently discards all log messages.
 *
 * Other modules can use NullLogger if they do not care about logging or if
 * no logger was injected. This avoids coupling to any concrete implementation
 * (e.g. spdlog or a custom logger).
 */
namespace constellation::interfaces::logging {

class NullLogger final : public ILogger {
public:
  void Log(Level /*level*/, const char* /*fmt*/, ...) override {
    // no-op
  }

  void Trace(const char* /*fmt*/, ...) override {
    // no-op
  }

  void Debug(const char* /*fmt*/, ...) override {
    // no-op
  }

  void Info(const char* /*fmt*/, ...) override {
    // no-op
  }

  void Warn(const char* /*fmt*/, ...) override {
    // no-op
  }

  void Error(const char* /*fmt*/, ...) override {
    // no-op
  }

  void Critical(const char* /*fmt*/, ...) override {
    // no-op
  }
};

} // end namespace constellation::interfaces::logging
