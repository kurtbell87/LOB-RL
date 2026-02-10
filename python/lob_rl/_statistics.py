"""Shared statistical utilities for LOB environments."""

import numpy as np


def rolling_std(arr, window, warmup=True):
    """Compute rolling standard deviation using cumulative sums.

    Returns an array of the same length as *arr* where output[t] is the
    std of arr[max(0, t-window):t].  Indices with fewer than 2 values
    are left at 0.0.

    If *warmup* is True (default), indices 2..window-1 use a growing
    window (size 2..t).  If False, only indices >= window are filled.

    Uses O(N) cumulative-sum algorithm (no Python loops over N).
    """
    N = len(arr)
    result = np.zeros(N, dtype=np.float32)
    if N < 2:
        return result

    arr64 = arr.astype(np.float64)
    cumsum = np.concatenate(([0.0], np.cumsum(arr64)))
    cumsum_sq = np.concatenate(([0.0], np.cumsum(arr64 ** 2)))

    # Warmup phase: t in [2, min(N, window)) — window grows from 2..t
    if warmup:
        for t in range(2, min(N, window)):
            w = t
            roll_sum = cumsum[t]
            roll_sum2 = cumsum_sq[t]
            roll_mean = roll_sum / w
            roll_var = roll_sum2 / w - roll_mean ** 2
            result[t] = np.sqrt(max(roll_var, 0.0))

    # Steady-state: full windows of size *window*
    if N > window:
        roll_sum = cumsum[window:N] - cumsum[:N - window]
        roll_sum2 = cumsum_sq[window:N] - cumsum_sq[:N - window]
        roll_mean = roll_sum / window
        roll_var = roll_sum2 / window - roll_mean ** 2
        np.maximum(roll_var, 0.0, out=roll_var)
        result[window:N] = np.sqrt(roll_var).astype(np.float32)

    return result
