"""Generate a session fixture that produces valid BBO changes for precompute().

Creates precompute_session.bin — a file with pre-market warmup + RTH messages
that trigger BBO changes, suitable for PrecomputedEnv.from_file() tests.
"""
import struct
import os

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

HEADER_FMT = "<4sIII"
RECORD_FMT = "<QQqIBBBBI"

BASE_DATE_NS = 1736899200_000_000_000  # Jan 15, 2025 midnight UTC
RTH_OPEN_OFFSET_NS = 48_600_000_000_000
RTH_CLOSE_OFFSET_NS = 72_000_000_000_000

ADD = ord('A')
CANCEL = ord('C')
BID_SIDE = ord('B')
ASK_SIDE = ord('A')


def generate():
    records = []
    base_bid = 5000 * 1_000_000_000
    base_ask = 5001 * 1_000_000_000
    oid = 1
    n_bbo_changes = 50

    # Pre-market warmup: establish initial BBO
    warmup_ts = BASE_DATE_NS + RTH_OPEN_OFFSET_NS - 5_000_000_000
    bid_oid = oid; oid += 1
    records.append((warmup_ts, bid_oid, base_bid, 10, ADD, BID_SIDE, 0))
    ask_oid = oid; oid += 1
    records.append((warmup_ts + 1_000_000, ask_oid, base_ask, 10, ADD, ASK_SIDE, 0))

    # RTH: cancel + re-add to create BBO changes
    rth_duration = RTH_CLOSE_OFFSET_NS - RTH_OPEN_OFFSET_NS
    for i in range(n_bbo_changes):
        progress = i / max(n_bbo_changes - 1, 1)
        ts = BASE_DATE_NS + RTH_OPEN_OFFSET_NS + int(progress * rth_duration * 0.95)
        ts += 1

        shift = ((i % 5) - 2) * 250_000_000

        records.append((ts, bid_oid, base_bid, 10, CANCEL, BID_SIDE, 0))
        new_bid = base_bid + shift
        bid_oid = oid; oid += 1
        records.append((ts + 1, bid_oid, new_bid, 10, ADD, BID_SIDE, 0))

        records.append((ts + 2, ask_oid, base_ask, 10, CANCEL, ASK_SIDE, 0))
        new_ask = base_ask - shift
        ask_oid = oid; oid += 1
        records.append((ts + 3, ask_oid, new_ask, 10, ADD, ASK_SIDE, 0))

    path = os.path.join(FIXTURE_DIR, "precompute_session.bin")
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, b"LOBR", 1, len(records), 42005347))
        for ts, order_id, price, qty, action, side, flags in records:
            f.write(struct.pack(RECORD_FMT, ts, order_id, price, qty, action, side, flags, 0, 0))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes, {len(records)} records)")


if __name__ == "__main__":
    generate()
