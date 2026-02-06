"""Generate test fixture binary files for Python bindings tests (B1).

Creates a .bin file with enough records to run a full episode (100+ messages)
with realistic bid/ask data that builds a proper order book.
"""
import struct
import os

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

# Header: magic(4) + version(4) + record_count(4) + instrument_id(4) = 16 bytes
HEADER_FMT = "<4sIII"
# Record: ts_ns(8) + order_id(8) + price_raw(8) + qty(4) + action(1) + side(1) + flags(1) + pad(1) + reserved(4) = 36 bytes
RECORD_FMT = "<QQqIBBBBI"


def write_episode_fixture():
    """Create a 200-record fixture with interleaved bids and asks.

    Enough records to run several episodes with steps_per_episode=50.
    Prices oscillate slightly to create realistic book dynamics.
    """
    records = []
    base_bid = 999_000_000_000  # 999.00 * 1e9
    base_ask = 1_001_000_000_000  # 1001.00 * 1e9

    for i in range(200):
        ts = 1_000_000_000 + i
        order_id = i + 1
        # Alternate bid/ask adds with slight price variation
        if i % 2 == 0:
            # Bid side
            price_offset = (i % 10) * 250_000_000  # 0.25 increments
            price = base_bid - price_offset
            side = ord('B')
        else:
            # Ask side
            price_offset = (i % 10) * 250_000_000
            price = base_ask + price_offset
            side = ord('A')
        qty = 10 + (i % 5) * 5
        action = ord('A')  # All adds
        records.append((ts, order_id, price, qty, action, side, 0))

    path = os.path.join(FIXTURE_DIR, "episode_200records.bin")
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, b"LOBR", 1, len(records), 42005347))
        for ts, oid, price, qty, action, side, flags in records:
            f.write(struct.pack(RECORD_FMT, ts, oid, price, qty, action, side, flags, 0, 0))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


def write_session_fixture():
    """Create a fixture with timestamps spanning pre-market, RTH, and post-market.

    RTH is 13:30-20:00 UTC. We create records with timestamps that cross
    these boundaries so session-aware LOBEnv can be tested.

    Timestamps use a realistic Unix nanosecond epoch for Jan 15, 2025.
    """
    records = []
    # Jan 15, 2025 midnight UTC in nanoseconds
    day_start_ns = 1_736_899_200_000_000_000

    # Pre-market: 13:00 UTC (before 13:30 open)
    pre_market_ns = day_start_ns + 13 * 3_600_000_000_000

    # RTH: 13:30-20:00 UTC
    rth_open_ns = day_start_ns + 13 * 3_600_000_000_000 + 30 * 60_000_000_000

    # Post-market: 20:00+ UTC
    post_market_ns = day_start_ns + 20 * 3_600_000_000_000

    base_bid = 999_000_000_000
    base_ask = 1_001_000_000_000

    order_id = 1

    # 20 pre-market messages (for warmup)
    for i in range(20):
        ts = pre_market_ns + i * 1_000_000  # 1ms apart
        price = base_bid - (i % 5) * 250_000_000 if i % 2 == 0 else base_ask + (i % 5) * 250_000_000
        side = ord('B') if i % 2 == 0 else ord('A')
        records.append((ts, order_id, price, 10, ord('A'), side, 0))
        order_id += 1

    # 150 RTH messages (enough for multiple episodes)
    for i in range(150):
        ts = rth_open_ns + i * 1_000_000_000  # 1s apart
        price = base_bid - (i % 5) * 250_000_000 if i % 2 == 0 else base_ask + (i % 5) * 250_000_000
        side = ord('B') if i % 2 == 0 else ord('A')
        qty = 10 + (i % 3) * 5
        records.append((ts, order_id, price, qty, ord('A'), side, 0))
        order_id += 1

    # 10 post-market messages
    for i in range(10):
        ts = post_market_ns + i * 1_000_000
        price = base_bid if i % 2 == 0 else base_ask
        side = ord('B') if i % 2 == 0 else ord('A')
        records.append((ts, order_id, price, 10, ord('A'), side, 0))
        order_id += 1

    path = os.path.join(FIXTURE_DIR, "session_180records.bin")
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, b"LOBR", 1, len(records), 42005347))
        for ts, oid, price, qty, action, side, flags in records:
            f.write(struct.pack(RECORD_FMT, ts, oid, price, qty, action, side, flags, 0, 0))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    write_episode_fixture()
    write_session_fixture()
    print("All bindings test fixtures generated.")
