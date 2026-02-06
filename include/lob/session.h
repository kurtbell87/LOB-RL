#pragma once
#include <cstdint>
#include <stdexcept>

struct SessionConfig {
    uint64_t rth_open_ns = 0;
    uint64_t rth_close_ns = 0;
    int warmup_messages = -1;

    bool is_valid() const {
        return rth_close_ns > rth_open_ns;
    }

    static SessionConfig default_rth() {
        SessionConfig cfg;
        // 13:30 UTC in nanoseconds since midnight
        cfg.rth_open_ns = 13ULL * 3600'000'000'000ULL + 30ULL * 60'000'000'000ULL;
        // 20:00 UTC in nanoseconds since midnight
        cfg.rth_close_ns = 20ULL * 3600'000'000'000ULL;
        cfg.warmup_messages = -1;
        return cfg;
    }
};

class SessionFilter {
public:
    enum class Phase { PreMarket, RTH, PostMarket };

    SessionFilter() : cfg_(SessionConfig::default_rth()) {}
    explicit SessionFilter(const SessionConfig& cfg) : cfg_(cfg) {
        if (!cfg.is_valid()) {
            throw std::invalid_argument("Invalid SessionConfig: rth_close_ns must be greater than rth_open_ns");
        }
    }

    uint64_t rth_open_ns() const { return cfg_.rth_open_ns; }
    uint64_t rth_close_ns() const { return cfg_.rth_close_ns; }
    uint64_t rth_duration_ns() const { return cfg_.rth_close_ns - cfg_.rth_open_ns; }

    Phase classify(uint64_t ts_ns) const {
        uint64_t tod = time_of_day_ns(ts_ns);
        if (tod < cfg_.rth_open_ns) return Phase::PreMarket;
        if (tod >= cfg_.rth_close_ns) return Phase::PostMarket;
        return Phase::RTH;
    }

    static uint64_t time_of_day_ns(uint64_t ts_ns) {
        static constexpr uint64_t NS_PER_DAY = 24ULL * 3600'000'000'000ULL;
        return ts_ns % NS_PER_DAY;
    }

    float session_progress(uint64_t ts_ns) const {
        uint64_t tod = time_of_day_ns(ts_ns);
        if (tod <= cfg_.rth_open_ns) return 0.0f;
        if (tod >= cfg_.rth_close_ns) return 1.0f;
        uint64_t elapsed = tod - cfg_.rth_open_ns;
        uint64_t duration = cfg_.rth_close_ns - cfg_.rth_open_ns;
        return static_cast<float>(static_cast<double>(elapsed) / static_cast<double>(duration));
    }

private:
    SessionConfig cfg_;
};
