"""Tests for StreamingNormalizer Python bindings.

Spec: docs/streaming-normalizer.md

These tests verify that:
- lob_rl_core.StreamingNormalizer is accessible and constructable
- normalize() accepts a list of floats and returns a list of floats
- bars_seen() and n_features() return correct values
- reset() clears internal state
- Feature-count mismatch raises an exception
- Output matches the C++ normalize_features() reference
"""

import math

import numpy as np
import pytest

import lob_rl_core


# ===========================================================================
# Test 1: StreamingNormalizer class exists and is constructable
# ===========================================================================


class TestStreamingNormalizerExists:
    """StreamingNormalizer should be accessible from lob_rl_core module."""

    def test_class_exists(self):
        """lob_rl_core should have a StreamingNormalizer attribute."""
        assert hasattr(lob_rl_core, "StreamingNormalizer"), (
            "lob_rl_core module should expose StreamingNormalizer"
        )

    def test_construct_default_window(self):
        """Should construct with just n_features (window defaults to 2000)."""
        norm = lob_rl_core.StreamingNormalizer(22)
        assert norm.n_features() == 22
        assert norm.bars_seen() == 0

    def test_construct_custom_window(self):
        """Should construct with n_features and custom window."""
        norm = lob_rl_core.StreamingNormalizer(5, 100)
        assert norm.n_features() == 5
        assert norm.bars_seen() == 0


# ===========================================================================
# Test 2: normalize() returns correct types and sizes
# ===========================================================================


class TestStreamingNormalizerNormalize:
    """normalize() should accept a list and return a list of the same size."""

    def test_returns_list_of_correct_size(self):
        """normalize() should return a list with n_features elements."""
        norm = lob_rl_core.StreamingNormalizer(3)
        result = norm.normalize([1.0, 2.0, 3.0])
        assert isinstance(result, list), "Should return a list"
        assert len(result) == 3, "Should have n_features elements"

    def test_first_bar_all_zeros(self):
        """First bar should produce all-zero z-scores (std=0 → z=0)."""
        norm = lob_rl_core.StreamingNormalizer(3)
        result = norm.normalize([10.0, 20.0, 30.0])
        for i, z in enumerate(result):
            assert z == 0.0, f"First bar feature {i} should be 0.0, got {z}"

    def test_bars_seen_increments(self):
        """bars_seen() should increment after each normalize() call."""
        norm = lob_rl_core.StreamingNormalizer(2)
        assert norm.bars_seen() == 0
        norm.normalize([1.0, 2.0])
        assert norm.bars_seen() == 1
        norm.normalize([3.0, 4.0])
        assert norm.bars_seen() == 2


# ===========================================================================
# Test 3: reset() clears state
# ===========================================================================


class TestStreamingNormalizerReset:
    """reset() should clear all internal state."""

    def test_reset_clears_bars_seen(self):
        """After reset(), bars_seen() should return 0."""
        norm = lob_rl_core.StreamingNormalizer(2)
        norm.normalize([1.0, 2.0])
        norm.normalize([3.0, 4.0])
        assert norm.bars_seen() == 2
        norm.reset()
        assert norm.bars_seen() == 0

    def test_after_reset_first_bar_is_zero(self):
        """After reset, the next normalize() should behave like the first bar."""
        norm = lob_rl_core.StreamingNormalizer(1)
        norm.normalize([100.0])
        norm.normalize([200.0])
        norm.reset()
        result = norm.normalize([50.0])
        assert result[0] == 0.0, "After reset, first bar z-score should be 0"


# ===========================================================================
# Test 4: Feature count mismatch raises
# ===========================================================================


class TestStreamingNormalizerMismatch:
    """normalize() should raise when input size != n_features."""

    def test_too_few_features_raises(self):
        """Should raise when fewer features than expected."""
        norm = lob_rl_core.StreamingNormalizer(3)
        with pytest.raises(Exception):
            norm.normalize([1.0, 2.0])

    def test_too_many_features_raises(self):
        """Should raise when more features than expected."""
        norm = lob_rl_core.StreamingNormalizer(3)
        with pytest.raises(Exception):
            norm.normalize([1.0, 2.0, 3.0, 4.0])


# ===========================================================================
# Test 5: Z-score correctness (matches C++ reference)
# ===========================================================================


class TestStreamingNormalizerZScores:
    """Streaming z-scores should match the batch normalize_features() reference."""

    def test_two_bar_sequence(self):
        """Two bars [1, 3] → bar 1 z-score should be 1.0."""
        norm = lob_rl_core.StreamingNormalizer(1, 0)  # expanding
        z0 = norm.normalize([1.0])
        assert z0[0] == 0.0

        z1 = norm.normalize([3.0])
        assert abs(z1[0] - 1.0) < 1e-12

    def test_nan_replaced_with_zero(self):
        """NaN values should be treated as 0.0."""
        norm = lob_rl_core.StreamingNormalizer(1, 0)
        z0 = norm.normalize([float("nan")])
        assert z0[0] == 0.0, "NaN → 0 → first bar → z=0"

        # Bar 1: window=[0.0, 4.0], mean=2, std=2, z=(4-2)/2=1.0
        z1 = norm.normalize([4.0])
        assert abs(z1[0] - 1.0) < 1e-12

    def test_clipping(self):
        """Extreme z-scores should be clipped to [-5, 5]."""
        norm = lob_rl_core.StreamingNormalizer(1, 0)
        for _ in range(50):
            norm.normalize([0.0])
        z = norm.normalize([10000.0])
        assert z[0] == 5.0, f"Expected clip to 5.0, got {z[0]}"
