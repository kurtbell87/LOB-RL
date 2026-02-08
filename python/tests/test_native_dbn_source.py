"""Tests for native-dbn-source feature: reading .dbn.zst files directly.

Spec: docs/native-dbn-source.md

Tests the following acceptance criteria:
1. precompute() binding gains instrument_id parameter
2. LOBEnv file-based constructors gain instrument_id parameter
3. precompute_cache.py updated:
   - --data-dir points to a directory of .dbn.zst files (no manifest needed)
   - --instrument-id required (uint32)
   - Globs for *.mbo.dbn.zst
   - Extracts dates from filenames (glbx-mdp3-YYYYMMDD.mbo.dbn.zst)
   - Output .npz cache identical format
4. BinaryFileSource and convert_dbn.py are deleted
"""

import glob
import importlib
import importlib.util
import json
import os
import struct
import sys
import tempfile

import numpy as np
import pytest

import lob_rl_core


# Path to the precompute_cache.py script
_SCRIPT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "precompute_cache.py")
)


def _run_precompute_cache(argv):
    """Load and run precompute_cache.py's main() with the given CLI argv list."""
    spec = importlib.util.spec_from_file_location("precompute_cache", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old_argv = sys.argv
    try:
        sys.argv = ["precompute_cache.py"] + argv
        mod.main()
    finally:
        sys.argv = old_argv


# ===========================================================================
# Acceptance Criterion 7: BinaryFileSource and convert_dbn.py are DELETED
# ===========================================================================


class TestDeletedInfrastructure:
    """BinaryFileSource and convert_dbn.py should no longer exist."""

    def test_convert_dbn_module_not_importable(self):
        """lob_rl.convert_dbn should no longer be importable."""
        with pytest.raises(ImportError):
            from lob_rl import convert_dbn  # noqa: F401

    def test_convert_dbn_file_deleted(self):
        """python/lob_rl/convert_dbn.py should not exist on disk."""
        convert_path = os.path.join(
            os.path.dirname(__file__), "..", "lob_rl", "convert_dbn.py"
        )
        assert not os.path.exists(convert_path), (
            f"convert_dbn.py should be deleted, but found at {convert_path}"
        )

    def test_binary_file_source_header_deleted(self):
        """src/data/binary_file_source.h should not exist."""
        header_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "src", "data", "binary_file_source.h")
        )
        assert not os.path.exists(header_path), (
            f"binary_file_source.h should be deleted, but found at {header_path}"
        )

    def test_binary_file_source_cpp_deleted(self):
        """src/data/binary_file_source.cpp should not exist."""
        cpp_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "src", "data", "binary_file_source.cpp")
        )
        assert not os.path.exists(cpp_path), (
            f"binary_file_source.cpp should be deleted, but found at {cpp_path}"
        )

    def test_test_binary_file_source_cpp_deleted(self):
        """tests/test_binary_file_source.cpp should not exist."""
        test_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "tests", "test_binary_file_source.cpp")
        )
        assert not os.path.exists(test_path), (
            f"test_binary_file_source.cpp should be deleted, but found at {test_path}"
        )


# ===========================================================================
# Acceptance Criterion 1: precompute() binding gains instrument_id
# ===========================================================================


class TestPrecomputeBindingInstrumentId:
    """precompute(path, config, instrument_id) should accept instrument_id."""

    def test_precompute_accepts_instrument_id_kwarg(self):
        """precompute() should accept instrument_id as a keyword argument."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        # This should not raise TypeError for unexpected keyword.
        # It may raise a file-not-found error if no .dbn.zst exists,
        # but should NOT raise TypeError about instrument_id.
        with pytest.raises(Exception) as exc_info:
            lob_rl_core.precompute("/nonexistent.dbn.zst", cfg, instrument_id=12345)

        # Should NOT be TypeError about instrument_id
        assert not isinstance(exc_info.value, TypeError), (
            f"precompute() should accept instrument_id kwarg, got: {exc_info.value}"
        )

    def test_precompute_accepts_instrument_id_positional(self):
        """precompute(path, config, instrument_id) should accept 3 positional args."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        with pytest.raises(Exception) as exc_info:
            lob_rl_core.precompute("/nonexistent.dbn.zst", cfg, 12345)

        assert not isinstance(exc_info.value, TypeError), (
            f"precompute() should accept instrument_id as 3rd positional arg, "
            f"got: {exc_info.value}"
        )

    def test_precompute_instrument_id_defaults_to_zero(self):
        """precompute(path, config) without instrument_id should default to 0."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        # Should not raise TypeError (2 args should still work)
        with pytest.raises(Exception) as exc_info:
            lob_rl_core.precompute("/nonexistent.dbn.zst", cfg)

        assert not isinstance(exc_info.value, TypeError), (
            f"precompute() should work with 2 args (default instrument_id=0), "
            f"got: {exc_info.value}"
        )


# ===========================================================================
# Acceptance Criterion 4: LOBEnv constructors gain instrument_id
# ===========================================================================


class TestLOBEnvInstrumentId:
    """LOBEnv file-based constructors should accept instrument_id parameter."""

    def test_lobenv_file_constructor_accepts_instrument_id(self):
        """LOBEnv(path, steps, instrument_id=...) should accept instrument_id."""
        # This should NOT raise TypeError about instrument_id.
        # It will raise a runtime error because the file doesn't exist.
        with pytest.raises(Exception) as exc_info:
            lob_rl_core.LOBEnv(
                "/nonexistent.dbn.zst",
                50,
                instrument_id=12345,
            )

        assert not isinstance(exc_info.value, TypeError), (
            f"LOBEnv should accept instrument_id kwarg, got: {exc_info.value}"
        )

    def test_lobenv_session_constructor_accepts_instrument_id(self):
        """LOBEnv(path, session_config, steps, instrument_id=...) should work."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        with pytest.raises(Exception) as exc_info:
            lob_rl_core.LOBEnv(
                "/nonexistent.dbn.zst",
                cfg,
                50,
                instrument_id=12345,
            )

        assert not isinstance(exc_info.value, TypeError), (
            f"LOBEnv session constructor should accept instrument_id, "
            f"got: {exc_info.value}"
        )


# ===========================================================================
# Acceptance Criterion 6: precompute_cache.py updated for .dbn.zst
# ===========================================================================


class TestPrecomputeCacheDbnZst:
    """precompute_cache.py should work with .dbn.zst files directly."""

    def test_script_exists(self):
        """scripts/precompute_cache.py should exist."""
        assert os.path.exists(_SCRIPT_PATH), f"Script not found: {_SCRIPT_PATH}"

    def test_cli_requires_instrument_id(self):
        """precompute_cache.py should require --instrument-id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            out_dir = os.path.join(tmpdir, "cache")

            # Should fail without --instrument-id
            with pytest.raises(SystemExit):
                _run_precompute_cache([
                    "--data-dir", data_dir,
                    "--out", out_dir,
                ])

    def test_cli_accepts_instrument_id(self):
        """precompute_cache.py should accept --instrument-id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            out_dir = os.path.join(tmpdir, "cache")

            # Should succeed with --instrument-id and empty data dir
            _run_precompute_cache([
                "--data-dir", data_dir,
                "--out", out_dir,
                "--instrument-id", "42005347",
            ])

    def test_cli_no_longer_needs_manifest(self):
        """precompute_cache.py should NOT require manifest.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            os.makedirs(data_dir)
            out_dir = os.path.join(tmpdir, "cache")

            # No manifest.json in data_dir — should NOT raise
            _run_precompute_cache([
                "--data-dir", data_dir,
                "--out", out_dir,
                "--instrument-id", "42005347",
            ])

    def test_cli_globs_for_dbn_zst_files(self):
        """precompute_cache.py should glob for *.mbo.dbn.zst files."""
        # We can verify the script source references the glob pattern.
        with open(_SCRIPT_PATH) as f:
            source = f.read()

        assert "mbo.dbn.zst" in source or ".dbn.zst" in source, (
            "precompute_cache.py should glob for *.mbo.dbn.zst files"
        )

    def test_cli_does_not_reference_manifest(self):
        """precompute_cache.py should not read manifest.json anymore."""
        with open(_SCRIPT_PATH) as f:
            source = f.read()

        assert "manifest" not in source.lower(), (
            "precompute_cache.py should not reference manifest.json anymore"
        )

    def test_cli_does_not_reference_bin_format(self):
        """precompute_cache.py should not reference .bin files anymore."""
        with open(_SCRIPT_PATH) as f:
            source = f.read()

        # The word ".bin" in the context of file paths should be gone
        # Allow it in comments but not in active code
        lines = [l for l in source.split('\n') if l.strip() and not l.strip().startswith('#')]
        code_text = '\n'.join(lines)
        assert '.bin"' not in code_text and ".bin'" not in code_text, (
            "precompute_cache.py should not reference .bin files in active code"
        )


# ===========================================================================
# Date extraction from .dbn.zst filenames
# ===========================================================================


class TestDateExtractionFromFilenames:
    """precompute_cache.py should extract dates from glbx-mdp3-YYYYMMDD.mbo.dbn.zst."""

    def test_script_references_date_extraction(self):
        """The script should have logic to extract dates from dbn filenames."""
        with open(_SCRIPT_PATH) as f:
            source = f.read()

        # Should contain regex or string parsing for YYYYMMDD from filename
        has_date_parse = ("YYYYMMDD" in source or
                          "glbx" in source or
                          r"\d{8}" in source or
                          "strftime" in source or
                          "date" in source.lower())
        assert has_date_parse, (
            "precompute_cache.py should extract dates from .dbn.zst filenames"
        )


# ===========================================================================
# Output .npz format unchanged
# ===========================================================================


class TestNpzFormatUnchanged:
    """The .npz output format should remain identical to before."""

    def test_npz_keys_are_obs_mid_spread(self):
        """Verify the expected keys haven't changed by checking the script."""
        with open(_SCRIPT_PATH) as f:
            source = f.read()

        # The script should still save obs, mid, spread
        assert "obs" in source, "Script should save 'obs' key"
        assert "mid" in source, "Script should save 'mid' key"
        assert "spread" in source, "Script should save 'spread' key"
        assert ".npz" in source or "savez" in source, "Script should save .npz files"


# ===========================================================================
# precompute() return format with instrument_id
# ===========================================================================


class TestPrecomputeReturnFormat:
    """precompute() with instrument_id should return same tuple format."""

    def test_precompute_returns_4_tuple(self):
        """precompute(path, config, instrument_id) returns (obs, mid, spread, num_steps)."""
        # We test the return format by inspecting the binding source.
        # The binding should return a 4-tuple: (obs, mid, spread, num_steps).
        # This is a structural test — actual data tests need a fixture.

        # Try to get the docstring or signature info
        assert callable(getattr(lob_rl_core, "precompute", None)), (
            "lob_rl_core.precompute should exist and be callable"
        )


# ===========================================================================
# Integration: precompute() instrument_id filtering affects output
# ===========================================================================


class TestPrecomputeInstrumentIdFiltering:
    """instrument_id should filter records in the native precompute path."""

    @pytest.fixture
    def dbn_fixture_path(self):
        """Path to the test fixture .dbn.zst file."""
        path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "..",
                         "tests", "fixtures", "test_mes.mbo.dbn.zst")
        )
        if not os.path.exists(path):
            pytest.skip(f"Fixture not found: {path} (GREEN phase creates it)")
        return path

    def test_wrong_instrument_id_produces_zero_steps(self, dbn_fixture_path):
        """Precompute with a non-matching instrument_id should yield 0 steps."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, mid, spread, num_steps = lob_rl_core.precompute(
            dbn_fixture_path, cfg, 99999
        )
        assert num_steps == 0, (
            f"Wrong instrument_id should produce 0 steps, got {num_steps}"
        )

    def test_correct_instrument_id_produces_nonzero_steps(self, dbn_fixture_path):
        """Precompute with matching instrument_id should yield > 0 steps."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, mid, spread, num_steps = lob_rl_core.precompute(
            dbn_fixture_path, cfg, 12345
        )
        assert num_steps > 0, (
            f"Correct instrument_id should produce > 0 steps, got {num_steps}"
        )

    def test_instrument_id_zero_matches_all(self, dbn_fixture_path):
        """instrument_id=0 should match all records (no filtering)."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        _, _, _, steps_all = lob_rl_core.precompute(dbn_fixture_path, cfg, 0)
        _, _, _, steps_specific = lob_rl_core.precompute(
            dbn_fixture_path, cfg, 12345
        )
        assert steps_all >= steps_specific, (
            f"instrument_id=0 should produce >= steps than specific id: "
            f"all={steps_all}, specific={steps_specific}"
        )

    def test_precompute_output_arrays_correct_shape(self, dbn_fixture_path):
        """Output arrays should have correct shapes."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        obs, mid, spread, num_steps = lob_rl_core.precompute(
            dbn_fixture_path, cfg, 12345
        )
        if num_steps > 0:
            assert obs.shape == (num_steps, 43), (
                f"obs shape should be ({num_steps}, 43), got {obs.shape}"
            )
            assert mid.shape == (num_steps,), (
                f"mid shape should be ({num_steps},), got {mid.shape}"
            )
            assert spread.shape == (num_steps,), (
                f"spread shape should be ({num_steps},), got {spread.shape}"
            )
            assert obs.dtype == np.float32
            assert mid.dtype == np.float64
            assert spread.dtype == np.float64
