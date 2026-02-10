"""Aggregate tick-level LOB data into bar-level features."""

import numpy as np

from lob_rl._obs_layout import (
    BID_PRICES as _BID_PRICES,
    BID_SIZES as _BID_SIZES,
    ASK_PRICES as _ASK_PRICES,
    ASK_SIZES as _ASK_SIZES,
    IMBALANCE as _IMBALANCE,
    TIME_LEFT as _TIME_LEFT,
)
from lob_rl._bar_layout import (
    BAR_RETURN, BAR_RANGE, BAR_VOLATILITY,
    SPREAD_MEAN, SPREAD_CLOSE,
    IMBALANCE_MEAN, IMBALANCE_CLOSE,
    BID_VOLUME_MEAN, ASK_VOLUME_MEAN, VOLUME_IMBALANCE,
    MICROPRICE_OFFSET, TIME_REMAINING, N_TICKS_NORM,
    NUM_BAR_FEATURES,
)


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

        bar_features[b, BAR_RETURN] = (mid_close - mid_open) / mid_open if mid_open != 0 else 0.0
        bar_features[b, BAR_RANGE] = (mid_high - mid_low) / mid_open if mid_open != 0 else 0.0

        if n_ticks >= 2:
            bar_features[b, BAR_VOLATILITY] = float(np.std(mid_chunk) / mid_open) if mid_open != 0 else 0.0

        bar_features[b, SPREAD_MEAN] = float(np.mean(spread_chunk))
        bar_features[b, SPREAD_CLOSE] = float(spread_chunk[-1])
        bar_features[b, IMBALANCE_MEAN] = float(np.mean(obs_chunk[:, _IMBALANCE]))
        bar_features[b, IMBALANCE_CLOSE] = float(obs_chunk[-1, _IMBALANCE])

        bid_sums = np.sum(obs_chunk[:, _BID_SIZES], axis=1)
        bar_features[b, BID_VOLUME_MEAN] = float(np.mean(bid_sums))

        ask_sums = np.sum(obs_chunk[:, _ASK_SIZES], axis=1)
        bar_features[b, ASK_VOLUME_MEAN] = float(np.mean(ask_sums))

        denom = bid_sums + ask_sums
        safe_denom = np.where(denom > 0, denom, 1.0)
        vi_per_tick = np.where(denom > 0, (bid_sums - ask_sums) / safe_denom, 0.0)
        bar_features[b, VOLUME_IMBALANCE] = float(np.mean(vi_per_tick))

        bid0 = obs_chunk[-1, _BID_PRICES.start]
        bidsize0 = obs_chunk[-1, _BID_SIZES.start]
        ask0 = obs_chunk[-1, _ASK_PRICES.start]
        asksize0 = obs_chunk[-1, _ASK_SIZES.start]
        denom_mp = bidsize0 + asksize0
        if denom_mp > 0:
            microprice = (ask0 * bidsize0 + bid0 * asksize0) / denom_mp
            bar_features[b, MICROPRICE_OFFSET] = float(microprice / mid_close - 1.0)

        bar_features[b, TIME_REMAINING] = float(obs_chunk[-1, _TIME_LEFT])
        bar_features[b, N_TICKS_NORM] = float(n_ticks / bar_size)

        # Output arrays
        bar_mid_close[b] = mid_close
        bar_spread_close[b] = spread_chunk[-1]

    return bar_features, bar_mid_close, bar_spread_close
