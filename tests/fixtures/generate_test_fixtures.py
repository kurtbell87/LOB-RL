"""Generate test fixture binary files for BinaryFileSource tests.

This script creates small .bin files with known values matching the FlatRecord
binary format described in docs/step2a-databento-source.md.

Run this script once to generate fixtures. The fixture files are checked into
the repo so they don't need to be regenerated.
"""
import struct
import os

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

# Header: magic(4) + version(4) + record_count(4) + instrument_id(4) = 16 bytes
HEADER_FMT = "<4sIII"
# Record: ts_ns(8) + order_id(8) + price_raw(8) + qty(4) + action(1) + side(1) + flags(1) + pad(1) + reserved(4) = 40 bytes
RECORD_FMT = "<QQqIBBBBI"


def write_valid_fixture():
    """Create a valid 10-record fixture file with known values."""
    records = [
        # ts_ns, order_id, price_raw (price * 1e9), qty, action, side, flags
        # 5 bid adds at prices 999.75, 999.50, 999.25, 999.00, 998.75
        (1000000000, 1, 999_750_000_000, 10, ord('A'), ord('B'), 0),
        (1000000001, 2, 999_500_000_000, 20, ord('A'), ord('B'), 0),
        (1000000002, 3, 999_250_000_000, 15, ord('A'), ord('B'), 0),
        (1000000003, 4, 999_000_000_000, 25, ord('A'), ord('B'), 0),
        (1000000004, 5, 998_750_000_000, 30, ord('A'), ord('B'), 0),
        # 5 ask adds at prices 1000.00, 1000.25, 1000.50, 1000.75, 1001.00
        (1000000005, 6, 1_000_000_000_000, 10, ord('A'), ord('A'), 0),
        (1000000006, 7, 1_000_250_000_000, 20, ord('A'), ord('A'), 0),
        (1000000007, 8, 1_000_500_000_000, 15, ord('A'), ord('A'), 0),
        (1000000008, 9, 1_000_750_000_000, 25, ord('A'), ord('A'), 0),
        (1000000009, 10, 1_001_000_000_000, 30, ord('A'), ord('A'), 0),
    ]

    path = os.path.join(FIXTURE_DIR, "valid_10records.bin")
    with open(path, "wb") as f:
        # Header
        f.write(struct.pack(HEADER_FMT, b"LOBR", 1, len(records), 42005347))
        # Records
        for ts, oid, price, qty, action, side, flags in records:
            f.write(struct.pack(RECORD_FMT, ts, oid, price, qty, action, side, flags, 0, 0))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


def write_mixed_actions_fixture():
    """Create a fixture with various actions: Add, Cancel, Modify, Trade, Fill."""
    records = [
        # Add bid
        (2000000000, 100, 999_750_000_000, 10, ord('A'), ord('B'), 0),
        # Add ask
        (2000000001, 101, 1_000_250_000_000, 10, ord('A'), ord('A'), 0),
        # Modify bid qty
        (2000000002, 100, 999_750_000_000, 25, ord('M'), ord('B'), 0),
        # Trade on ask
        (2000000003, 101, 1_000_250_000_000, 3, ord('T'), ord('A'), 0),
        # Fill on ask (should map to Trade)
        (2000000004, 101, 1_000_250_000_000, 2, ord('F'), ord('A'), 0),
        # Cancel bid
        (2000000005, 100, 999_750_000_000, 25, ord('C'), ord('B'), 0),
        # Record with Side='N' (maps to Bid)
        (2000000006, 200, 999_000_000_000, 5, ord('A'), ord('N'), 0),
    ]

    path = os.path.join(FIXTURE_DIR, "mixed_actions.bin")
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, b"LOBR", 1, len(records), 42005347))
        for ts, oid, price, qty, action, side, flags in records:
            f.write(struct.pack(RECORD_FMT, ts, oid, price, qty, action, side, flags, 0, 0))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


def write_empty_fixture():
    """Create a valid file with zero records."""
    path = os.path.join(FIXTURE_DIR, "empty.bin")
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, b"LOBR", 1, 0, 42005347))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


def write_bad_magic_fixture():
    """Create a file with invalid magic bytes."""
    path = os.path.join(FIXTURE_DIR, "bad_magic.bin")
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, b"BAAD", 1, 0, 42005347))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


def write_bad_version_fixture():
    """Create a file with unsupported version number."""
    path = os.path.join(FIXTURE_DIR, "bad_version.bin")
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, b"LOBR", 99, 0, 42005347))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


def write_truncated_fixture():
    """Create a file with header claiming 5 records but only containing 2."""
    records = [
        (3000000000, 1, 999_750_000_000, 10, ord('A'), ord('B'), 0),
        (3000000001, 2, 999_500_000_000, 20, ord('A'), ord('B'), 0),
    ]
    path = os.path.join(FIXTURE_DIR, "truncated.bin")
    with open(path, "wb") as f:
        # Header claims 5 records
        f.write(struct.pack(HEADER_FMT, b"LOBR", 1, 5, 42005347))
        # But only 2 records written
        for ts, oid, price, qty, action, side, flags in records:
            f.write(struct.pack(RECORD_FMT, ts, oid, price, qty, action, side, flags, 0, 0))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


def write_precision_test_fixture():
    """Create a fixture with precise prices to test int64_t -> double conversion.

    Tests typical /MES price: $5000.123456789 (price_raw = 5000123456789)
    This value is well under 2^53, so should round-trip without precision loss.
    """
    records = [
        # price_raw = 5000123456789 (5000.123456789 * 1e9)
        (4000000000, 1, 5_000_123_456_789, 10, ord('A'), ord('B'), 0),
    ]

    path = os.path.join(FIXTURE_DIR, "precision_test.bin")
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, b"LOBR", 1, len(records), 42005347))
        for ts, oid, price, qty, action, side, flags in records:
            f.write(struct.pack(RECORD_FMT, ts, oid, price, qty, action, side, flags, 0, 0))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


def write_precision_test_high_fixture():
    """Create a fixture with high-value price to test precision at upper range.

    Tests $100,000.123456789 (price_raw = 100000123456789)
    Still under 2^53 (~9e15), so should preserve precision.
    """
    records = [
        # price_raw = 100000123456789 (100000.123456789 * 1e9)
        (5000000000, 1, 100_000_123_456_789, 10, ord('A'), ord('B'), 0),
    ]

    path = os.path.join(FIXTURE_DIR, "precision_test_high.bin")
    with open(path, "wb") as f:
        f.write(struct.pack(HEADER_FMT, b"LOBR", 1, len(records), 42005347))
        for ts, oid, price, qty, action, side, flags in records:
            f.write(struct.pack(RECORD_FMT, ts, oid, price, qty, action, side, flags, 0, 0))
    print(f"Wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    write_valid_fixture()
    write_mixed_actions_fixture()
    write_empty_fixture()
    write_bad_magic_fixture()
    write_bad_version_fixture()
    write_truncated_fixture()
    write_precision_test_fixture()
    write_precision_test_high_fixture()
    print("All fixtures generated.")
