"""Generate test fixture binary files for PrecomputedMultiDayEnv tests.

Creates multiple small .bin files simulating different trading days with
RTH-range timestamps (13:30-20:00 UTC = 48600-72000 seconds since midnight).
Each day has distinct price ranges so tests can verify which day is active.

Strategy: During warmup, build a simple book with one bid and one ask.
During RTH, cancel the best bid/ask and add new ones at slightly different
prices, which triggers BBO changes that precompute() will capture.
"""
import struct
import os

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

# Header: magic(4) + version(4) + record_count(4) + instrument_id(4) = 16 bytes
HEADER_FMT = "<4sIII"
# Record: ts_ns(8) + order_id(8) + price_raw(8) + qty(4) + action(1) + side(1) + flags(1) + pad(1) + reserved(4) = 36 bytes
RECORD_FMT = "<QQqIBBBBI"

# Jan 15, 2025 midnight UTC in nanoseconds
BASE_DATE_NS = 1736899200_000_000_000
# RTH open: 13:30 UTC in nanoseconds from midnight
RTH_OPEN_OFFSET_NS = 48_600_000_000_000
# RTH close: 20:00 UTC in nanoseconds from midnight
RTH_CLOSE_OFFSET_NS = 72_000_000_000_000

ADD = ord('A')
CANCEL = ord('C')
BID = ord('B')
ASK = ord('A')


def write_precomputed_day_fixture(filename, day_index, n_bbo_changes=6):
    """Create a single-day fixture that produces n_bbo_changes during RTH.

    Each day has a distinct base_bid so tests can identify which day's data.

    Structure:
    - 2 pre-market warmup messages: establish initial BBO (1 bid + 1 ask)
    - n_bbo_changes * 2 RTH messages: for each BBO change, cancel current
      best bid/ask and add a replacement at a slightly different price.
      This produces exactly n_bbo_changes BBO snapshots.
    """
    records = []
    base_bid = (5000 + day_index * 500) * 1_000_000_000  # unique per day
    base_ask = base_bid + 1_000_000_000  # spread of 1.0
    oid = 1

    # --- Pre-market warmup: establish BBO ---
    warmup_ts_base = BASE_DATE_NS + RTH_OPEN_OFFSET_NS - 10_000_000_000  # 10s before RTH

    # Add initial bid
    bid_oid = oid; oid += 1
    records.append((warmup_ts_base, bid_oid, base_bid, 10, ADD, BID, 0))

    # Add initial ask
    ask_oid = oid; oid += 1
    records.append((warmup_ts_base + 1_000_000_000, ask_oid, base_ask, 10, ADD, ASK, 0))

    # --- RTH: cancel and re-add at different prices to create BBO changes ---
    rth_duration_ns = RTH_CLOSE_OFFSET_NS - RTH_OPEN_OFFSET_NS

    for i in range(n_bbo_changes):
        # Time within RTH session
        progress = i / max(n_bbo_changes - 1, 1)
        ts = BASE_DATE_NS + RTH_OPEN_OFFSET_NS + int(progress * rth_duration_ns * 0.95)
        ts += 1  # avoid exact duplicates

        # Small price shift — oscillate bid up/down to create BBO changes
        price_shift = ((i % 5) - 2) * 250_000_000  # -0.50, -0.25, 0, +0.25, +0.50

        # Cancel current best bid
        records.append((ts, bid_oid, base_bid, 10, CANCEL, BID, 0))

        # Add new bid at shifted price
        new_bid_price = base_bid + price_shift
        bid_oid = oid; oid += 1
        records.append((ts + 1, bid_oid, new_bid_price, 10, ADD, BID, 0))

        # Cancel current best ask
        records.append((ts + 2, ask_oid, base_ask, 10, CANCEL, ASK, 0))

        # Add new ask at shifted price (opposite direction)
        new_ask_price = base_ask - price_shift
        ask_oid = oid; oid += 1
        records.append((ts + 3, ask_oid, new_ask_price, 10, ADD, ASK, 0))

    total_records = len(records)
    path = os.path.join(FIXTURE_DIR, filename)
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, b"LOBR", 1, total_records, 42005347))
        for ts, order_id, price, qty, action, side, flags in records:
            f.write(struct.pack(RECORD_FMT, ts, order_id, price, qty, action, side, flags, 0, 0))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes, {total_records} records)")
    return path


if __name__ == "__main__":
    for day_idx in range(5):
        write_precomputed_day_fixture(f"day{day_idx}.bin", day_idx)
    print("All precomputed multi-day test fixtures generated.")
