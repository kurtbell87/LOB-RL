"""Generate a binary fixture file with pre-market + RTH timestamps for precompute tests.

Creates a file with messages that span pre-market warmup and RTH, suitable for
testing the precompute() convenience overload that takes a file path.

Run: uv run python tests/fixtures/generate_precompute_fixture.py
"""
import struct
import os

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

# Header: magic(4) + version(4) + record_count(4) + instrument_id(4) = 16 bytes
HEADER_FMT = "<4sIII"
# Record: ts_ns(8) + order_id(8) + price_raw(8) + qty(4) + action(1) + side(1) + flags(1) + pad(1) + reserved(4) = 36 bytes
RECORD_FMT = "<QQqIBBBBI"

NS_PER_SEC = 1_000_000_000
NS_PER_MIN = 60 * NS_PER_SEC
NS_PER_HOUR = 60 * NS_PER_MIN

# Reference day: 2024-01-15 00:00:00 UTC
DAY_BASE_NS = 19737 * 24 * NS_PER_HOUR

# RTH boundaries
RTH_OPEN_NS = 13 * NS_PER_HOUR + 30 * NS_PER_MIN  # 13:30 UTC
RTH_CLOSE_NS = 20 * NS_PER_HOUR  # 20:00 UTC


def price_raw(price_float):
    """Convert price to raw int64 (price * 1e9)."""
    return int(price_float * 1e9)


def write_precompute_fixture():
    """Create a fixture with pre-market warmup + RTH messages that change BBO."""
    records = []
    oid = 1

    # Pre-market: establish book at 09:00 UTC
    pre_start = DAY_BASE_NS + 9 * NS_PER_HOUR

    # Bid at 999.75
    records.append((pre_start, oid, price_raw(999.75), 100, ord('A'), ord('B'), 0))
    oid += 1
    # Ask at 1000.25
    records.append((pre_start + NS_PER_MIN, oid, price_raw(1000.25), 100, ord('A'), ord('A'), 0))
    oid += 1
    # Deeper bid at 999.50
    records.append((pre_start + 2 * NS_PER_MIN, oid, price_raw(999.50), 50, ord('A'), ord('B'), 0))
    oid += 1
    # Deeper ask at 1000.50
    records.append((pre_start + 3 * NS_PER_MIN, oid, price_raw(1000.50), 50, ord('A'), ord('A'), 0))
    oid += 1

    # RTH: 3 messages that change BBO
    rth_start = DAY_BASE_NS + RTH_OPEN_NS

    # Improve best bid to 1000.00 (BBO change)
    records.append((rth_start + NS_PER_MIN, oid, price_raw(1000.00), 100, ord('A'), ord('B'), 0))
    oid += 1
    # Improve best ask to 1000.125 (BBO change)
    records.append((rth_start + 2 * NS_PER_MIN, oid, price_raw(1000.125), 100, ord('A'), ord('A'), 0))
    oid += 1
    # Improve best bid to 1000.0625 (BBO change)
    records.append((rth_start + 3 * NS_PER_MIN, oid, price_raw(1000.0625), 100, ord('A'), ord('B'), 0))
    oid += 1

    path = os.path.join(FIXTURE_DIR, "precompute_rth.bin")
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, b"LOBR", 1, len(records), 42005347))
        for ts, order_id, price, qty, action, side, flags in records:
            f.write(struct.pack(RECORD_FMT, ts, order_id, price, qty, action, side, flags, 0, 0))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes, {len(records)} records)")


if __name__ == "__main__":
    write_precompute_fixture()
