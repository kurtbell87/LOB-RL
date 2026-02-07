"""Aggregate tick-level LOB data into bar-level features."""

import numpy as np


# Number of intra-bar features
NUM_BAR_FEATURES = 13


def aggregate_bars(obs, mid, spread, bar_size):
    """Aggregate tick-level arrays into bar-level features.

    Args:
        obs: (N, 43) float32 — C++ precomputed tick features
        mid: (N,) float64 — mid prices per tick
        spread: (N,) float64 — bid-ask spreads per tick
        bar_size: int — number of ticks per bar

    Returns:
        bar_features: (B, 13) float32 — bar-level features
        bar_mid_close: (B,) float64 — closing mid price per bar
        bar_spread_close: (B,) float64 — closing spread per bar
    """
    obs = np.asarray(obs, dtype=np.float32)
    mid = np.asarray(mid, dtype=np.float64)
    spread = np.asarray(spread, dtype=np.float64)

    N = obs.shape[0]
    n_full = N // bar_size
    remainder = N - n_full * bar_size
    threshold = bar_size // 4

    # Determine number of bars
    if remainder >= threshold:
        n_bars = n_full + 1
    else:
        n_bars = n_full

    if n_bars == 0:
        return (np.empty((0, NUM_BAR_FEATURES), dtype=np.float32),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64))

    bar_features = np.zeros((n_bars, NUM_BAR_FEATURES), dtype=np.float32)
    bar_mid_close = np.zeros(n_bars, dtype=np.float64)
    bar_spread_close = np.zeros(n_bars, dtype=np.float64)

    for b in range(n_bars):
        start = b * bar_size
        end = min(start + bar_size, N)
        n_ticks = end - start

        obs_chunk = obs[start:end]
        mid_chunk = mid[start:end]
        spread_chunk = spread[start:end]

        mid_open = mid_chunk[0]
        mid_close = mid_chunk[-1]
        mid_high = np.max(mid_chunk)
        mid_low = np.min(mid_chunk)

        # 0: bar_return
        bar_features[b, 0] = (mid_close - mid_open) / mid_open if mid_open != 0 else 0.0

        # 1: bar_range
        bar_features[b, 1] = (mid_high - mid_low) / mid_open if mid_open != 0 else 0.0

        # 2: bar_volatility
        if n_ticks >= 2:
            bar_features[b, 2] = float(np.std(mid_chunk) / mid_open) if mid_open != 0 else 0.0
        else:
            bar_features[b, 2] = 0.0

        # 3: spread_mean
        bar_features[b, 3] = float(np.mean(spread_chunk))

        # 4: spread_close
        bar_features[b, 4] = float(spread_chunk[-1])

        # 5: imbalance_mean
        bar_features[b, 5] = float(np.mean(obs_chunk[:, 41]))

        # 6: imbalance_close
        bar_features[b, 6] = float(obs_chunk[-1, 41])

        # 7: bid_volume_mean
        bid_sums = np.sum(obs_chunk[:, 10:20], axis=1)
        bar_features[b, 7] = float(np.mean(bid_sums))

        # 8: ask_volume_mean
        ask_sums = np.sum(obs_chunk[:, 30:40], axis=1)
        bar_features[b, 8] = float(np.mean(ask_sums))

        # 9: volume_imbalance
        denom = bid_sums + ask_sums
        safe_denom = np.where(denom > 0, denom, 1.0)
        vi_per_tick = np.where(denom > 0, (bid_sums - ask_sums) / safe_denom, 0.0)
        bar_features[b, 9] = float(np.mean(vi_per_tick))

        # 10: microprice_offset
        bid0 = obs_chunk[-1, 0]
        bidsize0 = obs_chunk[-1, 10]
        ask0 = obs_chunk[-1, 20]
        asksize0 = obs_chunk[-1, 30]
        denom_mp = bidsize0 + asksize0
        if denom_mp > 0:
            microprice = (ask0 * bidsize0 + bid0 * asksize0) / denom_mp
            bar_features[b, 10] = float(microprice / mid_close - 1.0)
        else:
            bar_features[b, 10] = 0.0

        # 11: time_remaining
        bar_features[b, 11] = float(obs_chunk[-1, 42])

        # 12: n_ticks_norm
        bar_features[b, 12] = float(n_ticks / bar_size)

        # Output arrays
        bar_mid_close[b] = mid_close
        bar_spread_close[b] = spread_chunk[-1]

    return bar_features, bar_mid_close, bar_spread_close
