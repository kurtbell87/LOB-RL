"""Generate test fixture binary files for MultiDayEnv tests (Step 4).

Creates multiple small .bin files simulating different trading days,
each with distinct price ranges so tests can verify which day is active.
"""
import struct
import os

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

# Header: magic(4) + version(4) + record_count(4) + instrument_id(4) = 16 bytes
HEADER_FMT = "<4sIII"
# Record: ts_ns(8) + order_id(8) + price_raw(8) + qty(4) + action(1) + side(1) + flags(1) + pad(1) + reserved(4) = 36 bytes
RECORD_FMT = "<QQqIBBBBI"


def write_day_fixture(filename, day_index, record_count=200):
    """Create a single-day fixture with unique price range based on day_index.

    Each day has a distinct base_bid so tests can identify which day's data
    is currently being used. Day 0 has bid ~1000, day 1 ~2000, etc.
    """
    records = []
    base_bid = (1000 + day_index * 1000) * 1_000_000_000  # 1000, 2000, 3000... * 1e9
    base_ask = base_bid + 2_000_000_000  # spread of 2.0

    for i in range(record_count):
        ts = 1_000_000_000 + i
        order_id = i + 1
        if i % 2 == 0:
            price_offset = (i % 10) * 250_000_000
            price = base_bid - price_offset
            side = ord('B')
        else:
            price_offset = (i % 10) * 250_000_000
            price = base_ask + price_offset
            side = ord('A')
        qty = 10 + (i % 5) * 5
        action = ord('A')
        records.append((ts, order_id, price, qty, action, side, 0))

    path = os.path.join(FIXTURE_DIR, filename)
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, b"LOBR", 1, len(records), 42005347))
        for ts, oid, price, qty, action, side, flags in records:
            f.write(struct.pack(RECORD_FMT, ts, oid, price, qty, action, side, flags, 0, 0))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes, {len(records)} records)")
    return path


if __name__ == "__main__":
    for day_idx in range(5):
        write_day_fixture(f"day{day_idx}.bin", day_idx)
    print("All multi-day test fixtures generated.")
