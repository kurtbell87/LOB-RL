#pragma once

#include <cstdarg>

/**
 * @brief A generic logging interface used throughout Constellation,
 *        decoupled from any concrete logging backend.
 *
 * By moving ILogger here into module_interfaces, other modules can
 * inject or implement a logger without depending on a specific
 * logging implementation or factory.
 */
namespace constellation::interfaces::logging {

class ILogger {
public:
  enum class Level {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Critical
  };

  virtual ~ILogger() = default;

  /**
   * @brief Log a formatted message at a given level, using printf-style formatting.
   * @param level The log level
   * @param fmt   The format string
   * @param ...   Format arguments
   */
  virtual void Log(Level level, const char* fmt, ...) = 0;

  /**
   * @brief Convenience wrappers for each log level, if desired.
   */
  virtual void Trace(const char* fmt, ...) = 0;
  virtual void Debug(const char* fmt, ...) = 0;
  virtual void Info(const char* fmt, ...)  = 0;
  virtual void Warn(const char* fmt, ...)  = 0;
  virtual void Error(const char* fmt, ...) = 0;
  virtual void Critical(const char* fmt, ...) = 0;
};

} // end constellation::interfaces::logging
