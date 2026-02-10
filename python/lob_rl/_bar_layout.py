"""Bar-level observation layout constants.

These indices map into the (B, 13) bar_features array produced by
aggregate_bars() and consumed by BarLevelEnv.
"""

# Intra-bar features (13 total)
BAR_RETURN = 0
BAR_RANGE = 1
BAR_VOLATILITY = 2
SPREAD_MEAN = 3
SPREAD_CLOSE = 4
IMBALANCE_MEAN = 5
IMBALANCE_CLOSE = 6
BID_VOLUME_MEAN = 7
ASK_VOLUME_MEAN = 8
VOLUME_IMBALANCE = 9
MICROPRICE_OFFSET = 10
TIME_REMAINING = 11
N_TICKS_NORM = 12

NUM_BAR_FEATURES = 13
