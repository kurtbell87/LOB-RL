"""Generate flagged_records.bin fixture for testing BinaryFileSource flags copying.

Creates 4 records with known flag values:
  Record 0: flags=0x82 (F_LAST | PUBLISHER_SPECIFIC) — event-terminal
  Record 1: flags=0x00 (none) — mid-event
  Record 2: flags=0x82 (F_LAST | PUBLISHER_SPECIFIC) — event-terminal
  Record 3: flags=0x28 (F_SNAPSHOT | BAD_TS_RECV) — snapshot record
"""
import struct
import os

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

# Header: magic(4) + version(u32) + record_count(u32) + instrument_id(u32) = 16 bytes
HEADER_FMT = "<4sIII"
# Record: ts_ns(8) + order_id(8) + price_raw(8) + qty(4) + action(1) + side(1) + flags(1) + pad(1) + reserved(4) = 36 bytes
RECORD_FMT = "<QQqIBBBBI"

records = [
    # ts_ns, order_id, price_raw, qty, action, side, flags
    (1000000000, 1, 999_750_000_000, 10, ord('A'), ord('B'), 0x82),
    (1000000001, 2, 1_000_250_000_000, 10, ord('A'), ord('A'), 0x00),
    (1000000002, 3, 1_000_000_000_000, 50, ord('A'), ord('B'), 0x82),
    (1000000003, 4, 999_500_000_000, 20, ord('A'), ord('B'), 0x28),
]

path = os.path.join(FIXTURE_DIR, "flagged_records.bin")
with open(path, "wb") as f:
    f.write(struct.pack(HEADER_FMT, b"LOBR", 1, len(records), 42005347))
    for ts, oid, price, qty, action, side, flags in records:
        f.write(struct.pack(RECORD_FMT, ts, oid, price, qty, action, side, flags, 0, 0))
print(f"Wrote {path} ({os.path.getsize(path)} bytes)")
