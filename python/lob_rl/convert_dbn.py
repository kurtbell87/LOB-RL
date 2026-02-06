"""Convert Databento DBN records to flat binary format for BinaryFileSource."""
import argparse
import glob
import json
import os
import re
import struct


# Binary format constants
HEADER_FMT = "<4sIII"  # magic(4) + version(4) + record_count(4) + instrument_id(4)
HEADER_SIZE = struct.calcsize(HEADER_FMT)

RECORD_FMT = "<QQqIBBBBI"  # ts_ns + order_id + price_raw + qty + action + side + flags + pad + reserved
RECORD_SIZE = struct.calcsize(RECORD_FMT)

# Actions to skip during conversion
SKIP_ACTIONS = {"R"}

# Action remapping: Fill -> Trade
ACTION_REMAP = {"F": "T"}


def _to_str(value):
    """Convert a Databento enum or plain str to str."""
    return value if isinstance(value, str) else str(value)


class MockRecord:
    """Mock DBN record for testing purposes."""

    def __init__(self, ts_event, order_id, price, size, action, side, flags,
                 instrument_id):
        self.ts_event = ts_event
        self.order_id = order_id
        self.price = price
        self.size = size
        self.action = action
        self.side = side
        self.flags = flags
        self.instrument_id = instrument_id


def convert_file(records, output_path, instrument_id):
    """Convert a list of record-like objects to a flat binary file.

    Args:
        records: List of record objects with fields: ts_event, order_id, price,
                 size, action, side, flags, instrument_id.
        output_path: Path to write the output .bin file.
        instrument_id: Only include records matching this instrument_id.
    """
    # Filter records by instrument_id and skip action='R'
    filtered = []
    for rec in records:
        if rec.instrument_id != instrument_id:
            continue
        action_str = _to_str(rec.action)
        if action_str in SKIP_ACTIONS:
            continue
        filtered.append(rec)

    with open(output_path, "wb") as f:
        # Write header
        f.write(struct.pack(HEADER_FMT, b"LOBR", 1, len(filtered), instrument_id))

        # Write records
        for rec in filtered:
            action_str = _to_str(rec.action)
            side_str = _to_str(rec.side)
            action_str = ACTION_REMAP.get(action_str, action_str)
            f.write(struct.pack(
                RECORD_FMT,
                rec.ts_event,
                rec.order_id,
                rec.price,
                rec.size,
                ord(action_str),
                ord(side_str),
                rec.flags,
                0,  # pad
                0,  # reserved
            ))


def write_mock_dbn_file(input_dir, date, instrument_id, num_records):
    """Create a mock .dbn.zst file in the input directory for testing.

    Creates a simple file that convert_directory can find and process.
    The file contains MockRecord-compatible data serialized as JSON.
    """
    filename = f"{date}.mock.json"
    filepath = os.path.join(input_dir, filename)

    records = []
    for i in range(num_records):
        records.append({
            "ts_event": 1000000000 + i,
            "order_id": i + 1,
            "price": 999_750_000_000 + i * 250_000_000,
            "size": 10 + i,
            "action": "A",
            "side": "B" if i % 2 == 0 else "A",
            "flags": 0,
            "instrument_id": instrument_id,
        })

    with open(filepath, "w") as f:
        json.dump({"date": date, "records": records}, f)


def _read_bin_stats(output_path):
    """Read back header and compute first/last timestamps from a .bin file."""
    with open(output_path, "rb") as f:
        header_data = f.read(HEADER_SIZE)
        _, _, record_count, _ = struct.unpack(HEADER_FMT, header_data)

        first_ts = 0
        last_ts = 0
        if record_count > 0:
            rec_data = f.read(RECORD_SIZE)
            first_ts = struct.unpack("<Q", rec_data[:8])[0]
            f.seek(HEADER_SIZE + (record_count - 1) * RECORD_SIZE)
            rec_data = f.read(RECORD_SIZE)
            last_ts = struct.unpack("<Q", rec_data[:8])[0]

    return record_count, first_ts, last_ts


def _extract_date_from_dbn_filename(filename):
    """Extract YYYYMMDD date from glbx-mdp3-YYYYMMDD.mbo.dbn.zst filename."""
    match = re.search(r'glbx-mdp3-(\d{8})\.mbo\.dbn\.zst', filename)
    if match:
        return match.group(1)
    return None


def _convert_and_record(records, date, output_dir, instrument_id):
    """Convert records to binary and return a manifest entry.

    Args:
        records: List of record-like objects.
        date: Date string (YYYYMMDD) for the output filename.
        output_dir: Directory to write the .bin file.
        instrument_id: Instrument ID to filter records.

    Returns:
        dict with date, record_count, first_ts, last_ts.
    """
    output_path = os.path.join(output_dir, f"{date}.bin")
    convert_file(records, output_path, instrument_id=instrument_id)

    record_count, first_ts, last_ts = _read_bin_stats(output_path)
    return {
        "date": date,
        "record_count": record_count,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def convert_directory(input_dir, output_dir, symbol, instrument_id):
    """Convert data files in input_dir to flat binary files in output_dir.

    Processes both .mock.json files and .dbn.zst files.
    Also writes a manifest.json with metadata about each output file.

    Args:
        input_dir: Directory containing .mock.json and/or .dbn.zst files.
        output_dir: Directory to write .bin files and manifest.json.
        symbol: Symbol name (e.g., "MESH5").
        instrument_id: Instrument ID to filter records.
    """
    os.makedirs(output_dir, exist_ok=True)

    manifest_files = []

    # Process mock data files first
    mock_files = sorted(glob.glob(os.path.join(input_dir, "*.mock.json")))

    for mock_file in mock_files:
        with open(mock_file) as f:
            data = json.load(f)

        records = [MockRecord(**r) for r in data["records"]]
        entry = _convert_and_record(records, data["date"], output_dir,
                                    instrument_id)
        manifest_files.append(entry)

    # Process .dbn.zst files
    dbn_files = sorted(glob.glob(os.path.join(input_dir, "*.mbo.dbn.zst")))

    if dbn_files:
        try:
            import databento
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "The 'databento' package is required to read .dbn.zst files. "
                "Install it with: uv pip install databento"
            )

        for dbn_file in dbn_files:
            date = _extract_date_from_dbn_filename(os.path.basename(dbn_file))
            if date is None:
                continue

            store = databento.DBNStore.from_file(dbn_file)
            entry = _convert_and_record(list(store), date, output_dir,
                                        instrument_id)
            manifest_files.append(entry)

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({"files": manifest_files}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Databento DBN files to flat binary format"
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing .dbn.zst or .mock.json files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write .bin files and manifest.json")
    parser.add_argument("--instrument-id", required=True, type=int,
                        help="Instrument ID to filter records")
    parser.add_argument("--symbol", default="MESH5",
                        help="Symbol name (default: MESH5)")

    args = parser.parse_args()
    convert_directory(args.input_dir, args.output_dir,
                      symbol=args.symbol, instrument_id=args.instrument_id)
