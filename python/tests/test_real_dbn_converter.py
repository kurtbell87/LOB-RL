"""Tests for B2: Make convert_dbn.py read real .dbn.zst files.

Tests the following spec requirements:
1. Action 'F' (fill) mapped to 'T' (trade) before writing
2. convert_directory() finds and processes *.mbo.dbn.zst files
3. Date extracted from filename via regex glbx-mdp3-(YYYYMMDD).mbo.dbn.zst
4. CLI entry point via __main__ / argparse
5. Graceful handling when databento is not installed
6. Backward compatibility with .mock.json files
7. Manifest correctness for mixed file types
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
from unittest import mock

import pytest

# Binary format constants (must match convert_dbn.py)
HEADER_FMT = "<4sIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
RECORD_FMT = "<QQqIBBBBI"
RECORD_SIZE = struct.calcsize(RECORD_FMT)


def _read_header(path):
    """Read and unpack the binary file header."""
    with open(path, "rb") as f:
        data = f.read(HEADER_SIZE)
    return struct.unpack(HEADER_FMT, data)


def _read_record(path, index=0):
    """Read and unpack a single record at the given index."""
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE + index * RECORD_SIZE)
        data = f.read(RECORD_SIZE)
    return struct.unpack(RECORD_FMT, data)


def _read_all_records(path):
    """Read all records from a binary file."""
    _, _, count, _ = _read_header(path)
    records = []
    for i in range(count):
        records.append(_read_record(path, i))
    return records


# ===========================================================================
# Requirement 1: Action 'F' mapped to 'T'
# ===========================================================================


class TestFillActionMapping:
    """Action 'F' (fill) should be mapped to 'T' (trade) in the output binary."""

    def test_fill_action_written_as_trade_byte(self, tmp_path):
        """A record with action='F' should be written with action byte ord('T')."""
        from lob_rl.convert_dbn import MockRecord, convert_file

        records = [
            MockRecord(
                ts_event=1000000000,
                order_id=1,
                price=1000000000000,
                size=5,
                action="F",
                side="A",
                flags=0,
                instrument_id=42005347,
            )
        ]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        _, _, _, _, action, _, _, _, _ = _read_record(output_path)
        assert action == ord("T"), (
            f"Action 'F' should be mapped to 'T' (0x{ord('T'):02X}), "
            f"got 0x{action:02X}"
        )

    def test_fill_action_not_written_as_F(self, tmp_path):
        """A record with action='F' must NOT be written as byte 'F'."""
        from lob_rl.convert_dbn import MockRecord, convert_file

        records = [
            MockRecord(
                ts_event=1000000000,
                order_id=1,
                price=1000000000000,
                size=5,
                action="F",
                side="A",
                flags=0,
                instrument_id=42005347,
            )
        ]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        _, _, _, _, action, _, _, _, _ = _read_record(output_path)
        assert action != ord("F"), "Action 'F' should be remapped, not written as 'F'"

    def test_fill_action_is_not_skipped(self, tmp_path):
        """Action 'F' should NOT be skipped — it should be included as a trade."""
        from lob_rl.convert_dbn import MockRecord, convert_file

        records = [
            MockRecord(
                ts_event=1,
                order_id=1,
                price=1000000000000,
                size=1,
                action="F",
                side="A",
                flags=0,
                instrument_id=42005347,
            )
        ]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        _, _, count, _ = _read_header(output_path)
        assert count == 1, f"Fill action should be included, got record_count={count}"

    def test_trade_action_still_written_as_trade(self, tmp_path):
        """Action 'T' should still be written as 'T' (no regression)."""
        from lob_rl.convert_dbn import MockRecord, convert_file

        records = [
            MockRecord(
                ts_event=1,
                order_id=1,
                price=1000000000000,
                size=1,
                action="T",
                side="A",
                flags=0,
                instrument_id=42005347,
            )
        ]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        _, _, _, _, action, _, _, _, _ = _read_record(output_path)
        assert action == ord("T")

    def test_mixed_f_and_t_actions_both_become_trade(self, tmp_path):
        """Both 'F' and 'T' actions should appear as 'T' in output."""
        from lob_rl.convert_dbn import MockRecord, convert_file

        records = [
            MockRecord(
                ts_event=1,
                order_id=1,
                price=1000000000000,
                size=1,
                action="T",
                side="A",
                flags=0,
                instrument_id=42005347,
            ),
            MockRecord(
                ts_event=2,
                order_id=2,
                price=1000000000000,
                size=1,
                action="F",
                side="A",
                flags=0,
                instrument_id=42005347,
            ),
        ]
        output_path = os.path.join(tmp_path, "test.bin")
        convert_file(records, output_path, instrument_id=42005347)

        all_recs = _read_all_records(output_path)
        assert len(all_recs) == 2
        assert all_recs[0][4] == ord("T"), "First record (action='T') should be 'T'"
        assert all_recs[1][4] == ord("T"), "Second record (action='F') should be mapped to 'T'"

    def test_other_actions_unchanged_after_fill_mapping(self, tmp_path):
        """Actions A, C, M should remain unchanged (not affected by F->T mapping)."""
        from lob_rl.convert_dbn import MockRecord, convert_file

        actions = ["A", "C", "M"]
        for act in actions:
            records = [
                MockRecord(
                    ts_event=1,
                    order_id=1,
                    price=1000000000000,
                    size=1,
                    action=act,
                    side="B",
                    flags=0,
                    instrument_id=42005347,
                )
            ]
            output_path = os.path.join(tmp_path, f"test_{act}.bin")
            convert_file(records, output_path, instrument_id=42005347)

            _, _, _, _, action_byte, _, _, _, _ = _read_record(output_path)
            assert action_byte == ord(act), (
                f"Action '{act}' should remain '{act}', got chr({action_byte})"
            )


# ===========================================================================
# Requirement 2: .dbn.zst file processing in convert_directory
# ===========================================================================


class FakeMBOMsg:
    """Fake MBOMsg that mimics databento record attributes."""

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


class FakeDBNStore:
    """Fake databento.DBNStore for testing without the databento package."""

    def __init__(self, records):
        self._records = records

    @classmethod
    def from_file(cls, path):
        """Create a FakeDBNStore. Records are loaded from a sidecar JSON file."""
        json_path = path + ".testdata.json"
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            records = [FakeMBOMsg(**r) for r in data]
            return cls(records)
        return cls([])

    def __iter__(self):
        return iter(self._records)


def _create_fake_dbn_zst(input_dir, date, records):
    """Create a fake .dbn.zst file with a sidecar JSON containing test records.

    The actual .dbn.zst file is empty — FakeDBNStore.from_file reads from the
    sidecar .testdata.json instead.
    """
    filename = f"glbx-mdp3-{date}.mbo.dbn.zst"
    filepath = os.path.join(input_dir, filename)
    # Create empty .dbn.zst file (just needs to exist for glob)
    with open(filepath, "wb") as f:
        f.write(b"")

    # Write sidecar with record data
    json_path = filepath + ".testdata.json"
    record_dicts = [
        {
            "ts_event": r.ts_event,
            "order_id": r.order_id,
            "price": r.price,
            "size": r.size,
            "action": r.action,
            "side": r.side,
            "flags": r.flags,
            "instrument_id": r.instrument_id,
        }
        for r in records
    ]
    with open(json_path, "w") as f:
        json.dump(record_dicts, f)

    return filepath


class TestDbnZstFileProcessing:
    """convert_directory() should find and process *.mbo.dbn.zst files."""

    def test_convert_directory_finds_dbn_zst_files(self, tmp_path):
        """convert_directory() should glob for *.mbo.dbn.zst files."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        records = [
            FakeMBOMsg(
                ts_event=1000000000,
                order_id=1,
                price=999750000000,
                size=10,
                action="A",
                side="B",
                flags=0,
                instrument_id=42005347,
            ),
        ]
        _create_fake_dbn_zst(input_dir, "20250102", records)

        # Patch databento.DBNStore to use our fake
        fake_databento = mock.MagicMock()
        fake_databento.DBNStore = FakeDBNStore
        with mock.patch.dict("sys.modules", {"databento": fake_databento}):
            convert_directory(input_dir, output_dir, symbol="MESH5",
                              instrument_id=42005347)

        # Should produce a .bin file for the date
        bin_path = os.path.join(output_dir, "20250102.bin")
        assert os.path.exists(bin_path), (
            f"Expected {bin_path} to exist after processing .dbn.zst file"
        )

    def test_dbn_zst_records_written_to_binary(self, tmp_path):
        """Records from .dbn.zst files should be written to the output .bin file."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        records = [
            FakeMBOMsg(ts_event=2000000000, order_id=100, price=999750000000,
                       size=10, action="A", side="B", flags=0,
                       instrument_id=42005347),
            FakeMBOMsg(ts_event=2000000001, order_id=101, price=1000250000000,
                       size=5, action="A", side="A", flags=0,
                       instrument_id=42005347),
        ]
        _create_fake_dbn_zst(input_dir, "20250103", records)

        fake_databento = mock.MagicMock()
        fake_databento.DBNStore = FakeDBNStore
        with mock.patch.dict("sys.modules", {"databento": fake_databento}):
            convert_directory(input_dir, output_dir, symbol="MESH5",
                              instrument_id=42005347)

        bin_path = os.path.join(output_dir, "20250103.bin")
        _, _, count, iid = _read_header(bin_path)
        assert count == 2, f"Expected 2 records, got {count}"
        assert iid == 42005347

    def test_dbn_zst_instrument_filtering(self, tmp_path):
        """Records from .dbn.zst with wrong instrument_id should be filtered out."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        records = [
            FakeMBOMsg(ts_event=1, order_id=1, price=999750000000, size=10,
                       action="A", side="B", flags=0,
                       instrument_id=42005347),
            FakeMBOMsg(ts_event=2, order_id=2, price=999750000000, size=10,
                       action="A", side="B", flags=0,
                       instrument_id=99999),  # wrong instrument
        ]
        _create_fake_dbn_zst(input_dir, "20250104", records)

        fake_databento = mock.MagicMock()
        fake_databento.DBNStore = FakeDBNStore
        with mock.patch.dict("sys.modules", {"databento": fake_databento}):
            convert_directory(input_dir, output_dir, symbol="MESH5",
                              instrument_id=42005347)

        bin_path = os.path.join(output_dir, "20250104.bin")
        _, _, count, _ = _read_header(bin_path)
        assert count == 1, f"Expected 1 record (filtered), got {count}"

    def test_multiple_dbn_zst_files_processed(self, tmp_path):
        """Multiple .dbn.zst files should each produce a corresponding .bin file."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        for date in ["20250102", "20250103", "20250106"]:
            records = [
                FakeMBOMsg(ts_event=1, order_id=1, price=999750000000, size=10,
                           action="A", side="B", flags=0,
                           instrument_id=42005347),
            ]
            _create_fake_dbn_zst(input_dir, date, records)

        fake_databento = mock.MagicMock()
        fake_databento.DBNStore = FakeDBNStore
        with mock.patch.dict("sys.modules", {"databento": fake_databento}):
            convert_directory(input_dir, output_dir, symbol="MESH5",
                              instrument_id=42005347)

        for date in ["20250102", "20250103", "20250106"]:
            bin_path = os.path.join(output_dir, f"{date}.bin")
            assert os.path.exists(bin_path), f"Expected {bin_path} to exist"


# ===========================================================================
# Requirement 3: Date extraction from filename
# ===========================================================================


class TestDateExtraction:
    """Date should be extracted from glbx-mdp3-YYYYMMDD.mbo.dbn.zst filename."""

    def test_date_extracted_from_standard_filename(self, tmp_path):
        """Output .bin filename should use date from glbx-mdp3-YYYYMMDD.mbo.dbn.zst."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        records = [
            FakeMBOMsg(ts_event=1, order_id=1, price=999750000000, size=10,
                       action="A", side="B", flags=0,
                       instrument_id=42005347),
        ]
        _create_fake_dbn_zst(input_dir, "20250115", records)

        fake_databento = mock.MagicMock()
        fake_databento.DBNStore = FakeDBNStore
        with mock.patch.dict("sys.modules", {"databento": fake_databento}):
            convert_directory(input_dir, output_dir, symbol="MESH5",
                              instrument_id=42005347)

        # The output file should be named 20250115.bin
        expected_bin = os.path.join(output_dir, "20250115.bin")
        assert os.path.exists(expected_bin), (
            f"Expected output file named 20250115.bin from "
            f"glbx-mdp3-20250115.mbo.dbn.zst"
        )

    def test_date_in_manifest_from_dbn_filename(self, tmp_path):
        """Manifest entry date should match the date extracted from .dbn.zst filename."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        records = [
            FakeMBOMsg(ts_event=1, order_id=1, price=999750000000, size=10,
                       action="A", side="B", flags=0,
                       instrument_id=42005347),
        ]
        _create_fake_dbn_zst(input_dir, "20250120", records)

        fake_databento = mock.MagicMock()
        fake_databento.DBNStore = FakeDBNStore
        with mock.patch.dict("sys.modules", {"databento": fake_databento}):
            convert_directory(input_dir, output_dir, symbol="MESH5",
                              instrument_id=42005347)

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        files = manifest["files"]
        dates = [entry["date"] for entry in files]
        assert "20250120" in dates, (
            f"Expected date '20250120' in manifest, got dates: {dates}"
        )


# ===========================================================================
# Requirement 4: CLI entry point
# ===========================================================================


class TestCLIEntryPoint:
    """CLI should be runnable as `python -m lob_rl.convert_dbn`."""

    def test_main_module_exists(self):
        """The convert_dbn module should be runnable with -m flag.

        This tests that either __main__.py exists in lob_rl/ or convert_dbn.py
        has an if __name__ == '__main__' block with argparse.
        """
        # Check that the module has CLI-related code
        from lob_rl import convert_dbn
        source = open(convert_dbn.__file__).read()

        has_main_block = "__name__" in source and "__main__" in source
        has_main_module = os.path.exists(
            os.path.join(os.path.dirname(convert_dbn.__file__), "__main__.py")
        )

        assert has_main_block or has_main_module, (
            "convert_dbn must have a '__main__' block or lob_rl/__main__.py must exist "
            "to support `python -m lob_rl.convert_dbn`"
        )

    def test_cli_requires_input_dir(self, tmp_path):
        """CLI should require --input-dir argument."""
        result = subprocess.run(
            [sys.executable, "-m", "lob_rl.convert_dbn",
             "--output-dir", str(tmp_path / "out"),
             "--instrument-id", "42005347"],
            capture_output=True, text=True,
            cwd=str(tmp_path),
            env={**os.environ, "PYTHONPATH": os.path.join(
                os.path.dirname(__file__), "..", "..")},
        )
        assert result.returncode != 0, (
            "CLI should fail when --input-dir is missing"
        )

    def test_cli_requires_output_dir(self, tmp_path):
        """CLI should require --output-dir argument."""
        result = subprocess.run(
            [sys.executable, "-m", "lob_rl.convert_dbn",
             "--input-dir", str(tmp_path / "in"),
             "--instrument-id", "42005347"],
            capture_output=True, text=True,
            cwd=str(tmp_path),
            env={**os.environ, "PYTHONPATH": os.path.join(
                os.path.dirname(__file__), "..", "..")},
        )
        assert result.returncode != 0, (
            "CLI should fail when --output-dir is missing"
        )

    def test_cli_requires_instrument_id(self, tmp_path):
        """CLI should require --instrument-id argument."""
        result = subprocess.run(
            [sys.executable, "-m", "lob_rl.convert_dbn",
             "--input-dir", str(tmp_path / "in"),
             "--output-dir", str(tmp_path / "out")],
            capture_output=True, text=True,
            cwd=str(tmp_path),
            env={**os.environ, "PYTHONPATH": os.path.join(
                os.path.dirname(__file__), "..", "..")},
        )
        assert result.returncode != 0, (
            "CLI should fail when --instrument-id is missing"
        )

    def test_cli_default_symbol_is_mesh5(self):
        """CLI --symbol should default to 'MESH5'."""
        # We verify by checking that argparse is configured with this default.
        # Import the module and check if it defines argument parsing.
        from lob_rl import convert_dbn
        import argparse
        import inspect

        source = inspect.getsource(convert_dbn)
        # The source should reference MESH5 as a default
        assert "MESH5" in source, (
            "CLI should have 'MESH5' as the default symbol value"
        )

    def test_cli_runs_successfully_with_empty_input(self, tmp_path):
        """CLI should run successfully with valid args and empty input dir."""
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()

        result = subprocess.run(
            [sys.executable, "-m", "lob_rl.convert_dbn",
             "--input-dir", str(input_dir),
             "--output-dir", str(output_dir),
             "--instrument-id", "42005347"],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": os.path.join(
                os.path.dirname(__file__), "..", "..")},
        )
        assert result.returncode == 0, (
            f"CLI should succeed with empty input dir. "
            f"stderr: {result.stderr}"
        )
        # Manifest should still be written
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists(), "manifest.json should be written even with empty input"


# ===========================================================================
# Requirement 5: Graceful databento import failure
# ===========================================================================


class TestDatabentoimportError:
    """When databento is not installed, should handle gracefully."""

    def test_convert_directory_with_dbn_zst_without_databento_gives_helpful_error(self, tmp_path):
        """If .dbn.zst files exist but databento is not installed, raise ImportError
        with a helpful message."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        # Create a .dbn.zst file
        dbn_path = os.path.join(input_dir, "glbx-mdp3-20250102.mbo.dbn.zst")
        with open(dbn_path, "wb") as f:
            f.write(b"")

        # Ensure databento is NOT importable
        with mock.patch.dict("sys.modules", {"databento": None}):
            with pytest.raises((ImportError, ModuleNotFoundError)) as exc_info:
                convert_directory(input_dir, output_dir, symbol="MESH5",
                                  instrument_id=42005347)

            # The error message should mention databento or pip install
            error_msg = str(exc_info.value).lower()
            assert "databento" in error_msg, (
                f"ImportError should mention 'databento', got: {exc_info.value}"
            )

    def test_convert_directory_mock_only_works_without_databento(self, tmp_path):
        """If only .mock.json files exist (no .dbn.zst), should work without databento."""
        from lob_rl.convert_dbn import convert_directory, write_mock_dbn_file

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        write_mock_dbn_file(input_dir, date="20241226",
                            instrument_id=42005347, num_records=3)

        # Even if databento is not importable, mock files should work
        with mock.patch.dict("sys.modules", {"databento": None}):
            convert_directory(input_dir, output_dir, symbol="MESH5",
                              instrument_id=42005347)

        manifest_path = os.path.join(output_dir, "manifest.json")
        assert os.path.exists(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert len(manifest["files"]) == 1


# ===========================================================================
# Requirement 6: Backward compatibility with .mock.json files
# ===========================================================================


class TestBackwardCompatibility:
    """Existing .mock.json path must continue to work."""

    def test_mock_json_files_still_processed(self, tmp_path):
        """convert_directory should still process .mock.json files."""
        from lob_rl.convert_dbn import convert_directory, write_mock_dbn_file

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        write_mock_dbn_file(input_dir, date="20241226",
                            instrument_id=42005347, num_records=5)

        convert_directory(input_dir, output_dir, symbol="MESH5",
                          instrument_id=42005347)

        bin_path = os.path.join(output_dir, "20241226.bin")
        assert os.path.exists(bin_path)
        _, _, count, _ = _read_header(bin_path)
        assert count == 5

    def test_mock_json_processed_before_dbn_zst(self, tmp_path):
        """When both .mock.json and .dbn.zst exist, mock files should be
        processed first (for test compatibility)."""
        from lob_rl.convert_dbn import convert_directory, write_mock_dbn_file

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        # Create mock file for date A
        write_mock_dbn_file(input_dir, date="20241225",
                            instrument_id=42005347, num_records=3)

        # Create dbn.zst for date B
        records = [
            FakeMBOMsg(ts_event=1, order_id=1, price=999750000000, size=10,
                       action="A", side="B", flags=0,
                       instrument_id=42005347),
        ]
        _create_fake_dbn_zst(input_dir, "20250102", records)

        fake_databento = mock.MagicMock()
        fake_databento.DBNStore = FakeDBNStore
        with mock.patch.dict("sys.modules", {"databento": fake_databento}):
            convert_directory(input_dir, output_dir, symbol="MESH5",
                              instrument_id=42005347)

        # Both should produce output files
        assert os.path.exists(os.path.join(output_dir, "20241225.bin"))
        assert os.path.exists(os.path.join(output_dir, "20250102.bin"))

        # Check manifest ordering: mock file date should come first
        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        dates = [entry["date"] for entry in manifest["files"]]
        mock_idx = dates.index("20241225")
        dbn_idx = dates.index("20250102")
        assert mock_idx < dbn_idx, (
            f"Mock files should appear before .dbn.zst files in manifest. "
            f"Got order: {dates}"
        )


# ===========================================================================
# Requirement 7: Manifest correctness for .dbn.zst files
# ===========================================================================


class TestManifestForDbnZst:
    """Manifest should contain correct metadata for .dbn.zst converted files."""

    def test_manifest_includes_dbn_zst_entries(self, tmp_path):
        """Manifest should include entries for files converted from .dbn.zst."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        records = [
            FakeMBOMsg(ts_event=5000000000, order_id=1, price=999750000000,
                       size=10, action="A", side="B", flags=0,
                       instrument_id=42005347),
            FakeMBOMsg(ts_event=6000000000, order_id=2, price=1000250000000,
                       size=5, action="A", side="A", flags=0,
                       instrument_id=42005347),
        ]
        _create_fake_dbn_zst(input_dir, "20250110", records)

        fake_databento = mock.MagicMock()
        fake_databento.DBNStore = FakeDBNStore
        with mock.patch.dict("sys.modules", {"databento": fake_databento}):
            convert_directory(input_dir, output_dir, symbol="MESH5",
                              instrument_id=42005347)

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert len(manifest["files"]) == 1
        entry = manifest["files"][0]
        assert entry["date"] == "20250110"
        assert entry["record_count"] == 2
        assert entry["first_ts"] == 5000000000
        assert entry["last_ts"] == 6000000000

    def test_manifest_record_count_correct_after_filtering(self, tmp_path):
        """Manifest record_count should reflect post-filtering count."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        records = [
            FakeMBOMsg(ts_event=1, order_id=1, price=999750000000, size=10,
                       action="A", side="B", flags=0,
                       instrument_id=42005347),
            FakeMBOMsg(ts_event=2, order_id=2, price=999750000000, size=10,
                       action="R", side="N", flags=0,
                       instrument_id=42005347),  # skipped
            FakeMBOMsg(ts_event=3, order_id=3, price=999750000000, size=10,
                       action="A", side="B", flags=0,
                       instrument_id=99999),  # wrong instrument
        ]
        _create_fake_dbn_zst(input_dir, "20250111", records)

        fake_databento = mock.MagicMock()
        fake_databento.DBNStore = FakeDBNStore
        with mock.patch.dict("sys.modules", {"databento": fake_databento}):
            convert_directory(input_dir, output_dir, symbol="MESH5",
                              instrument_id=42005347)

        manifest_path = os.path.join(output_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        entry = manifest["files"][0]
        assert entry["record_count"] == 1


# ===========================================================================
# Edge case: Zero-record .dbn.zst files
# ===========================================================================


class TestZeroRecordDbnZst:
    """Handle .dbn.zst files that produce zero records for the instrument."""

    def test_zero_records_after_filtering(self, tmp_path):
        """A .dbn.zst with no matching records should not crash."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        # All records have wrong instrument_id
        records = [
            FakeMBOMsg(ts_event=1, order_id=1, price=999750000000, size=10,
                       action="A", side="B", flags=0,
                       instrument_id=99999),
        ]
        _create_fake_dbn_zst(input_dir, "20250112", records)

        fake_databento = mock.MagicMock()
        fake_databento.DBNStore = FakeDBNStore
        with mock.patch.dict("sys.modules", {"databento": fake_databento}):
            # Should not raise
            convert_directory(input_dir, output_dir, symbol="MESH5",
                              instrument_id=42005347)

    def test_empty_dbn_zst_file(self, tmp_path):
        """A .dbn.zst with zero records total should not crash."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        _create_fake_dbn_zst(input_dir, "20250113", [])

        fake_databento = mock.MagicMock()
        fake_databento.DBNStore = FakeDBNStore
        with mock.patch.dict("sys.modules", {"databento": fake_databento}):
            convert_directory(input_dir, output_dir, symbol="MESH5",
                              instrument_id=42005347)

        # Manifest should still be valid
        manifest_path = os.path.join(output_dir, "manifest.json")
        assert os.path.exists(manifest_path)


# ===========================================================================
# Edge case: Fill action mapping in dbn.zst pipeline
# ===========================================================================


class TestFillActionInDbnZstPipeline:
    """Fill action 'F' should be mapped to 'T' even when coming through
    the .dbn.zst pipeline in convert_directory."""

    def test_fill_mapped_to_trade_via_dbn_zst(self, tmp_path):
        """Records with action='F' from .dbn.zst should be written as 'T'."""
        from lob_rl.convert_dbn import convert_directory

        input_dir = os.path.join(tmp_path, "input")
        output_dir = os.path.join(tmp_path, "output")
        os.makedirs(input_dir)

        records = [
            FakeMBOMsg(ts_event=1, order_id=1, price=999750000000, size=10,
                       action="A", side="B", flags=0,
                       instrument_id=42005347),
            FakeMBOMsg(ts_event=2, order_id=2, price=999750000000, size=5,
                       action="F", side="A", flags=0,
                       instrument_id=42005347),
        ]
        _create_fake_dbn_zst(input_dir, "20250114", records)

        fake_databento = mock.MagicMock()
        fake_databento.DBNStore = FakeDBNStore
        with mock.patch.dict("sys.modules", {"databento": fake_databento}):
            convert_directory(input_dir, output_dir, symbol="MESH5",
                              instrument_id=42005347)

        bin_path = os.path.join(output_dir, "20250114.bin")
        all_recs = _read_all_records(bin_path)
        assert len(all_recs) == 2
        # Second record had action='F', should be written as 'T'
        assert all_recs[1][4] == ord("T"), (
            f"Fill action from .dbn.zst should be mapped to 'T', "
            f"got 0x{all_recs[1][4]:02X}"
        )
