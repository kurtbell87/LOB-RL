"""Tests for the Databento DBN to flat binary converter.

Tests the convert_dbn module which reads .dbn.zst files and writes
flat binary .bin files for the C++ BinaryFileSource.
"""
import json
import os
import struct
import tempfile

import pytest

# Header and record struct formats matching the spec
HEADER_FMT = "<4sIII"  # magic(4) + version(4) + record_count(4) + instrument_id(4)
HEADER_SIZE = struct.calcsize(HEADER_FMT)

RECORD_FMT = "<QQqIBBBBI"  # ts_ns + order_id + price_raw + qty + action + side + flags + pad + reserved
RECORD_SIZE = struct.calcsize(RECORD_FMT)


# ===========================================================================
# Module import
# ===========================================================================


def test_convert_dbn_module_importable():
    """convert_dbn module should be importable."""
    from lob_rl import convert_dbn  # noqa: F401


# ===========================================================================
# Flat Binary Format: Header
# ===========================================================================


class TestFlatBinaryHeader:
    """Tests for the output binary file header format."""

    def _write_mock_dbn_and_convert(self, tmp_path, records, instrument_id=42005347, symbol="MESH5"):
        """Helper: create mock DBN data and run the converter.

        Returns the path to the output .bin file.
        """
        from lob_rl.convert_dbn import convert_file

        # convert_file should accept a list of record-like objects and output path
        output_path = os.path.join(tmp_path, "test_output.bin")
        convert_file(records, output_path, instrument_id=instrument_id)
        return output_path

    def test_header_magic_bytes(self, tmp_path):
        """Output file header should start with magic bytes 'LOBR'."""
        from lob_rl.convert_dbn import convert_file

        output_path = os.path.join(tmp_path, "test.bin")
        convert_file([], output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            magic = f.read(4)
        assert magic == b"LOBR", f"Expected magic 'LOBR', got {magic!r}"

    def test_header_version_is_one(self, tmp_path):
        """Output file header version should be 1."""
        from lob_rl.convert_dbn import convert_file

        output_path = os.path.join(tmp_path, "test.bin")
        convert_file([], output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            data = f.read(HEADER_SIZE)
        _, version, _, _ = struct.unpack(HEADER_FMT, data)
        assert version == 1

    def test_header_record_count_matches(self, tmp_path):
        """Header record_count should match actual number of records written."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [
            MockRecord(ts_event=1000, order_id=1, price=999750000000,
                       size=10, action="A", side="B", flags=0,
                       instrument_id=42005347),
            MockRecord(ts_event=1001, order_id=2, price=1000250000000,
                       size=20, action="A", side="A", flags=0,
                       instrument_id=42005347),
        ]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            data = f.read(HEADER_SIZE)
        _, _, count, _ = struct.unpack(HEADER_FMT, data)
        assert count == 2

    def test_header_instrument_id_matches(self, tmp_path):
        """Header instrument_id should match the specified instrument."""
        from lob_rl.convert_dbn import convert_file

        output_path = os.path.join(tmp_path, "test.bin")
        convert_file([], output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            data = f.read(HEADER_SIZE)
        _, _, _, iid = struct.unpack(HEADER_FMT, data)
        assert iid == 42005347

    def test_header_size_is_16_bytes(self, tmp_path):
        """Header should be exactly 16 bytes."""
        assert HEADER_SIZE == 16, f"Header should be 16 bytes, got {HEADER_SIZE}"


# ===========================================================================
# Flat Binary Format: Records
# ===========================================================================


class TestFlatBinaryRecords:
    """Tests for the output binary file record format."""

    def test_record_size_is_36_bytes(self):
        """Each record should be exactly 36 bytes."""
        assert RECORD_SIZE == 36, f"Record should be 36 bytes, got {RECORD_SIZE}"

    def test_file_size_matches_header_plus_records(self, tmp_path):
        """File size should equal header (16) + record_count * record_size."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [
            MockRecord(ts_event=1000, order_id=1, price=999750000000,
                       size=10, action="A", side="B", flags=0,
                       instrument_id=42005347),
        ]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        file_size = os.path.getsize(output_path)
        expected = HEADER_SIZE + len(records) * RECORD_SIZE
        assert file_size == expected, f"Expected {expected} bytes, got {file_size}"

    def test_record_fields_written_correctly(self, tmp_path):
        """Record fields should be written in correct binary format."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [
            MockRecord(ts_event=1000000000, order_id=42, price=999750000000,
                       size=15, action="A", side="B", flags=0,
                       instrument_id=42005347),
        ]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            f.seek(HEADER_SIZE)  # skip header
            data = f.read(RECORD_SIZE)

        ts_ns, order_id, price_raw, qty, action, side, flags, pad, reserved = \
            struct.unpack(RECORD_FMT, data)

        assert ts_ns == 1000000000
        assert order_id == 42
        assert price_raw == 999750000000
        assert qty == 15
        assert action == ord("A")
        assert side == ord("B")
        assert flags == 0
        assert pad == 0
        assert reserved == 0


# ===========================================================================
# Action Mapping
# ===========================================================================


class TestActionMapping:
    """Tests for DBN action -> FlatRecord action byte mapping."""

    def test_add_action_mapped(self, tmp_path):
        """DBN action 'A' should map to byte 'A' (0x41)."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [MockRecord(ts_event=1, order_id=1, price=1000000000000,
                              size=1, action="A", side="B", flags=0,
                              instrument_id=42005347)]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            f.seek(HEADER_SIZE)
            data = f.read(RECORD_SIZE)
        _, _, _, _, action, _, _, _, _ = struct.unpack(RECORD_FMT, data)
        assert action == ord("A")

    def test_cancel_action_mapped(self, tmp_path):
        """DBN action 'C' should map to byte 'C' (0x43)."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [MockRecord(ts_event=1, order_id=1, price=1000000000000,
                              size=1, action="C", side="B", flags=0,
                              instrument_id=42005347)]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            f.seek(HEADER_SIZE)
            data = f.read(RECORD_SIZE)
        _, _, _, _, action, _, _, _, _ = struct.unpack(RECORD_FMT, data)
        assert action == ord("C")

    def test_modify_action_mapped(self, tmp_path):
        """DBN action 'M' should map to byte 'M' (0x4D)."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [MockRecord(ts_event=1, order_id=1, price=1000000000000,
                              size=1, action="M", side="B", flags=0,
                              instrument_id=42005347)]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            f.seek(HEADER_SIZE)
            data = f.read(RECORD_SIZE)
        _, _, _, _, action, _, _, _, _ = struct.unpack(RECORD_FMT, data)
        assert action == ord("M")

    def test_trade_action_mapped(self, tmp_path):
        """DBN action 'T' should map to byte 'T' (0x54)."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [MockRecord(ts_event=1, order_id=1, price=1000000000000,
                              size=1, action="T", side="A", flags=0,
                              instrument_id=42005347)]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            f.seek(HEADER_SIZE)
            data = f.read(RECORD_SIZE)
        _, _, _, _, action, _, _, _, _ = struct.unpack(RECORD_FMT, data)
        assert action == ord("T")

    def test_fill_action_mapped(self, tmp_path):
        """DBN action 'F' should be remapped to 'T' (Trade) byte."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [MockRecord(ts_event=1, order_id=1, price=1000000000000,
                              size=1, action="F", side="A", flags=0,
                              instrument_id=42005347)]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            f.seek(HEADER_SIZE)
            data = f.read(RECORD_SIZE)
        _, _, _, _, action, _, _, _, _ = struct.unpack(RECORD_FMT, data)
        assert action == ord("T")

    def test_clear_action_skipped(self, tmp_path):
        """DBN action 'R' (clear/reset) should be skipped entirely."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [
            MockRecord(ts_event=1, order_id=1, price=1000000000000,
                       size=1, action="A", side="B", flags=0,
                       instrument_id=42005347),
            MockRecord(ts_event=2, order_id=2, price=1000000000000,
                       size=1, action="R", side="N", flags=0,
                       instrument_id=42005347),
            MockRecord(ts_event=3, order_id=3, price=1000000000000,
                       size=1, action="A", side="A", flags=0,
                       instrument_id=42005347),
        ]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            data = f.read(HEADER_SIZE)
        _, _, count, _ = struct.unpack(HEADER_FMT, data)
        # Only 2 records should be written (the 'R' is skipped)
        assert count == 2, f"Expected 2 records (R skipped), got {count}"


# ===========================================================================
# Side Mapping
# ===========================================================================


class TestSideMapping:
    """Tests for DBN side -> FlatRecord side byte mapping."""

    def test_bid_side_mapped(self, tmp_path):
        """DBN side 'B' should map to byte 'B' (0x42)."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [MockRecord(ts_event=1, order_id=1, price=1000000000000,
                              size=1, action="A", side="B", flags=0,
                              instrument_id=42005347)]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            f.seek(HEADER_SIZE)
            data = f.read(RECORD_SIZE)
        _, _, _, _, _, side, _, _, _ = struct.unpack(RECORD_FMT, data)
        assert side == ord("B")

    def test_ask_side_mapped(self, tmp_path):
        """DBN side 'A' should map to byte 'A' (0x41)."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [MockRecord(ts_event=1, order_id=1, price=1000000000000,
                              size=1, action="A", side="A", flags=0,
                              instrument_id=42005347)]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            f.seek(HEADER_SIZE)
            data = f.read(RECORD_SIZE)
        _, _, _, _, _, side, _, _, _ = struct.unpack(RECORD_FMT, data)
        assert side == ord("A")

    def test_none_side_mapped(self, tmp_path):
        """DBN side 'N' should map to byte 'N' (0x4E)."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [MockRecord(ts_event=1, order_id=1, price=1000000000000,
                              size=1, action="A", side="N", flags=0,
                              instrument_id=42005347)]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            f.seek(HEADER_SIZE)
            data = f.read(RECORD_SIZE)
        _, _, _, _, _, side, _, _, _ = struct.unpack(RECORD_FMT, data)
        assert side == ord("N")


# ===========================================================================
# Instrument ID Filtering
# ===========================================================================


class TestInstrumentIdFiltering:
    """Tests for filtering records by instrument_id."""

    def test_filters_to_specified_instrument_id(self, tmp_path):
        """Only records matching the specified instrument_id should be included."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [
            MockRecord(ts_event=1, order_id=1, price=1000000000000,
                       size=1, action="A", side="B", flags=0,
                       instrument_id=42005347),
            MockRecord(ts_event=2, order_id=2, price=1000000000000,
                       size=1, action="A", side="A", flags=0,
                       instrument_id=99999999),  # different instrument
            MockRecord(ts_event=3, order_id=3, price=1000000000000,
                       size=1, action="A", side="B", flags=0,
                       instrument_id=42005347),
        ]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            data = f.read(HEADER_SIZE)
        _, _, count, _ = struct.unpack(HEADER_FMT, data)
        assert count == 2, f"Expected 2 records (filtered by instrument_id), got {count}"

    def test_no_matching_instruments_produces_empty_file(self, tmp_path):
        """If no records match the instrument_id, output should have zero records."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [
            MockRecord(ts_event=1, order_id=1, price=1000000000000,
                       size=1, action="A", side="B", flags=0,
                       instrument_id=99999999),
        ]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            data = f.read(HEADER_SIZE)
        _, _, count, _ = struct.unpack(HEADER_FMT, data)
        assert count == 0


# ===========================================================================
# Empty Input
# ===========================================================================


class TestEmptyInput:
    """Tests for handling empty input."""

    def test_empty_records_produces_header_only(self, tmp_path):
        """Empty record list should produce a valid file with just a header."""
        from lob_rl.convert_dbn import convert_file

        output_path = os.path.join(tmp_path, "test.bin")
        convert_file([], output_path, instrument_id=42005347)

        file_size = os.path.getsize(output_path)
        assert file_size == HEADER_SIZE, f"Expected {HEADER_SIZE} bytes, got {file_size}"

        with open(output_path, "rb") as f:
            data = f.read(HEADER_SIZE)
        magic, version, count, iid = struct.unpack(HEADER_FMT, data)
        assert magic == b"LOBR"
        assert version == 1
        assert count == 0


# ===========================================================================
# Manifest JSON
# ===========================================================================


class TestManifest:
    """Tests for the manifest.json output."""

    def test_manifest_is_written(self, tmp_path):
        """Converter should write a manifest.json file."""
        from lob_rl.convert_dbn import convert_directory

        # Create a minimal mock input directory structure
        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        convert_directory(input_dir, output_dir, symbol="MESH5",
                          instrument_id=42005347)

        manifest_path = os.path.join(output_dir, "manifest.json")
        assert os.path.exists(manifest_path), "manifest.json should be created"

    def test_manifest_is_valid_json(self, tmp_path):
        """manifest.json should be valid JSON."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        convert_directory(input_dir, output_dir, symbol="MESH5",
                          instrument_id=42005347)

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path) as f:
            data = json.load(f)
        assert isinstance(data, (dict, list))

    def test_manifest_contains_file_entries(self, tmp_path):
        """manifest.json should list output files with metadata."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        convert_directory(input_dir, output_dir, symbol="MESH5",
                          instrument_id=42005347)

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path) as f:
            data = json.load(f)

        # Manifest should have a 'files' key or be a list of file entries
        if isinstance(data, dict):
            assert "files" in data, "Manifest should have a 'files' key"

    def test_manifest_entry_has_required_fields(self, tmp_path):
        """Each manifest entry should have date, record_count, first_ts, last_ts."""
        from lob_rl.convert_dbn import convert_directory, write_mock_dbn_file

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create a mock .dbn.zst file
        write_mock_dbn_file(input_dir, date="20241226",
                            instrument_id=42005347, num_records=5)

        convert_directory(input_dir, output_dir, symbol="MESH5",
                          instrument_id=42005347)

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path) as f:
            data = json.load(f)

        files = data["files"] if isinstance(data, dict) else data
        assert len(files) > 0, "Should have at least one file entry"

        entry = files[0]
        for field in ["date", "record_count", "first_ts", "last_ts"]:
            assert field in entry, f"Manifest entry missing '{field}' field"


# ===========================================================================
# Combined Filtering: instrument_id + action='R' skip
# ===========================================================================


class TestCombinedFiltering:
    """Tests for combined instrument_id filtering and action='R' skipping."""

    def test_both_filters_applied_together(self, tmp_path):
        """Both instrument_id filter and action='R' skip should apply."""
        from lob_rl.convert_dbn import convert_file, MockRecord

        records = [
            # Correct instrument, valid action -> included
            MockRecord(ts_event=1, order_id=1, price=1000000000000,
                       size=1, action="A", side="B", flags=0,
                       instrument_id=42005347),
            # Correct instrument, R action -> skipped
            MockRecord(ts_event=2, order_id=2, price=1000000000000,
                       size=1, action="R", side="N", flags=0,
                       instrument_id=42005347),
            # Wrong instrument, valid action -> skipped
            MockRecord(ts_event=3, order_id=3, price=1000000000000,
                       size=1, action="A", side="B", flags=0,
                       instrument_id=99999999),
            # Correct instrument, valid action -> included
            MockRecord(ts_event=4, order_id=4, price=1000000000000,
                       size=1, action="T", side="A", flags=0,
                       instrument_id=42005347),
        ]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        with open(output_path, "rb") as f:
            data = f.read(HEADER_SIZE)
        _, _, count, _ = struct.unpack(HEADER_FMT, data)
        assert count == 2, f"Expected 2 records after filtering, got {count}"
