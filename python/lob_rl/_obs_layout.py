"""C++ observation layout constants — shared across precomputed_env and bar_aggregation.

These indices must match the C++ FeatureBuilder observation layout (43 features
per timestep, without the position field).
"""

# Price levels (10-deep each side)
BID_PRICES = slice(0, 10)
BID_SIZES = slice(10, 20)
ASK_PRICES = slice(20, 30)
ASK_SIZES = slice(30, 40)

# Scalar features
REL_SPREAD = 40
IMBALANCE = 41
TIME_LEFT = 42

# Sizes
BASE_OBS_SIZE = 43  # features per tick from C++ precompute
