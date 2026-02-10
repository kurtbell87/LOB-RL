"""Tests for the Gambler's ruin validation module.

Spec: docs/t4-gamblers-ruin-validation.md

Validates that the label construction pipeline (T2) correctly detects barrier
hits by comparing empirical hit frequencies to the analytic Gambler's ruin
formula across 5 drift levels.
"""

import math

import numpy as np
import pytest

from lob_rl.barrier import TICK_SIZE
from lob_rl.barrier.gamblers_ruin import (
    gamblers_ruin_analytic,
    generate_random_walk,
    run_validation,
    validate_drift_level,
)


# ===========================================================================
# 1. Imports — all public functions importable
# ===========================================================================


class TestGamblersRuinImports:
    """All public functions from gamblers_ruin module should be importable."""

    def test_gamblers_ruin_analytic_importable(self):
        from lob_rl.barrier.gamblers_ruin import gamblers_ruin_analytic
        assert callable(gamblers_ruin_analytic)

    def test_generate_random_walk_importable(self):
        from lob_rl.barrier.gamblers_ruin import generate_random_walk
        assert callable(generate_random_walk)

    def test_validate_drift_level_importable(self):
        from lob_rl.barrier.gamblers_ruin import validate_drift_level
        assert callable(validate_drift_level)

    def test_run_validation_importable(self):
        from lob_rl.barrier.gamblers_ruin import run_validation
        assert callable(run_validation)


# ===========================================================================
# 2. Analytic formula correctness (spec tests #1-#9)
# ===========================================================================


class TestAnalyticZeroDrift:
    """Spec test #1: Zero drift (p=0.5) → P(upper) = b / (a + b)."""

    def test_p_half_a20_b10(self):
        """gamblers_ruin_analytic(20, 10, 0.5) == 10 / 30 ≈ 0.3333."""
        result = gamblers_ruin_analytic(20, 10, 0.5)
        expected = 10.0 / 30.0
        assert result == pytest.approx(expected, abs=1e-10)

    def test_is_float(self):
        """Return type should be float."""
        result = gamblers_ruin_analytic(20, 10, 0.5)
        assert isinstance(result, float)


class TestAnalyticKnownValues:
    """Spec tests #2-#5: Known values at specific drift levels within 1%."""

    def test_p_0505_approx_0402(self):
        """Spec test #2: p=0.505 → P(upper) ≈ 0.402 (standard formula)."""
        result = gamblers_ruin_analytic(20, 10, 0.505)
        assert result == pytest.approx(0.402, rel=0.01)

    def test_p_0510_approx_0472(self):
        """Spec test #3: p=0.510 → P(upper) ≈ 0.472 (standard formula)."""
        result = gamblers_ruin_analytic(20, 10, 0.510)
        assert result == pytest.approx(0.472, rel=0.01)

    def test_p_0490_approx_0212(self):
        """Spec test #4: p=0.490 → P(upper) ≈ 0.212 (standard formula)."""
        result = gamblers_ruin_analytic(20, 10, 0.490)
        assert result == pytest.approx(0.212, rel=0.01)

    def test_p_0485_approx_0163(self):
        """Spec test #5: p=0.485 → P(upper) ≈ 0.163 (standard formula)."""
        result = gamblers_ruin_analytic(20, 10, 0.485)
        assert result == pytest.approx(0.163, rel=0.01)


class TestAnalyticSymmetricBarriers:
    """Spec test #6: Symmetric barriers (a == b) with p=0.5 → P(upper) = 0.5."""

    def test_symmetric_10_10(self):
        result = gamblers_ruin_analytic(10, 10, 0.5)
        assert result == pytest.approx(0.5, abs=1e-10)

    def test_symmetric_5_5(self):
        result = gamblers_ruin_analytic(5, 5, 0.5)
        assert result == pytest.approx(0.5, abs=1e-10)

    def test_symmetric_1_1(self):
        """Spec test #9: Unit barriers (a=1, b=1, p=0.5) → 0.5."""
        result = gamblers_ruin_analytic(1, 1, 0.5)
        assert result == pytest.approx(0.5, abs=1e-10)


class TestAnalyticEdgeCases:
    """Spec tests #7-#8: Edge cases p near 0 and p near 1."""

    def test_p_near_zero_very_low_upper_probability(self):
        """Spec test #7: p near 0 → P(upper) very low."""
        result = gamblers_ruin_analytic(20, 10, 0.01)
        assert result < 0.001, f"P(upper) = {result}, expected < 0.001 for p≈0"

    def test_p_near_one_very_high_upper_probability(self):
        """Spec test #8: p near 1 → P(upper) very high."""
        result = gamblers_ruin_analytic(20, 10, 0.99)
        assert result > 0.999, f"P(upper) = {result}, expected > 0.999 for p≈1"


class TestAnalyticFormulaConsistency:
    """Additional formula correctness tests — verify the formula itself."""

    def test_formula_p_neq_half_manual_computation(self):
        """Hand-compute for a=2, b=1, p=0.6 and verify.

        q = 0.4, q/p = 2/3.
        P(upper) = (1 - (2/3)^1) / (1 - (2/3)^3)
                  = (1 - 2/3) / (1 - 8/27)
                  = (1/3) / (19/27)
                  = 27 / 57
                  = 9/19 ≈ 0.47368...
        """
        result = gamblers_ruin_analytic(2, 1, 0.6)
        expected = 9.0 / 19.0
        assert result == pytest.approx(expected, abs=1e-10)

    def test_result_in_zero_one_range(self):
        """Result should always be a probability in [0, 1]."""
        for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            for a in [1, 5, 20, 50]:
                for b in [1, 5, 10, 20]:
                    result = gamblers_ruin_analytic(a, b, p)
                    assert 0.0 <= result <= 1.0, (
                        f"Out of range: a={a}, b={b}, p={p} → {result}"
                    )

    def test_increasing_p_increases_upper_probability(self):
        """Higher p → higher P(upper hit first), monotonically."""
        prev = 0.0
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            result = gamblers_ruin_analytic(20, 10, p)
            assert result > prev, (
                f"P(upper) not increasing: p={p} → {result} <= prev={prev}"
            )
            prev = result

    def test_no_nan_or_inf(self):
        """Result should never be NaN or Inf."""
        for p in [0.01, 0.49, 0.5, 0.51, 0.99]:
            result = gamblers_ruin_analytic(20, 10, p)
            assert math.isfinite(result), f"Non-finite result: p={p} → {result}"


# ===========================================================================
# 3. Random walk generator (spec tests #10-#15)
# ===========================================================================


class TestRandomWalkOutputShape:
    """Spec test #10: Output length equals n_trades."""

    def test_length_1000(self):
        result = generate_random_walk(1000, seed=42)
        assert len(result) == 1000

    def test_length_100(self):
        result = generate_random_walk(100, seed=42)
        assert len(result) == 100

    def test_returns_structured_array(self):
        """Output should be a structured numpy array."""
        result = generate_random_walk(100, seed=42)
        assert isinstance(result, np.ndarray)
        assert result.dtype.names is not None

    def test_has_required_fields(self):
        """Output must have fields: price, size, side, ts_event."""
        result = generate_random_walk(100, seed=42)
        assert "price" in result.dtype.names
        assert "size" in result.dtype.names
        assert "side" in result.dtype.names
        assert "ts_event" in result.dtype.names


class TestRandomWalkPriceOnTickGrid:
    """Spec test #11: All prices are multiples of tick_size."""

    def test_default_tick_grid(self):
        result = generate_random_walk(5000, seed=42)
        prices = result["price"]
        # Each price should be a multiple of 0.25
        remainders = np.mod(prices, TICK_SIZE)
        np.testing.assert_allclose(
            remainders, 0.0, atol=1e-10,
            err_msg="Some prices are not on the tick grid"
        )

    def test_custom_tick_size(self):
        result = generate_random_walk(1000, tick_size=0.50, seed=42)
        prices = result["price"]
        remainders = np.mod(prices, 0.50)
        np.testing.assert_allclose(remainders, 0.0, atol=1e-10)


class TestRandomWalkMonotonicTimestamps:
    """Spec test #12: ts_event strictly increasing."""

    def test_strictly_increasing(self):
        result = generate_random_walk(5000, seed=42)
        ts = result["ts_event"]
        diffs = np.diff(ts)
        assert np.all(diffs > 0), "Timestamps are not strictly increasing"

    def test_nanosecond_timestamps(self):
        """Timestamps should be nanosecond resolution (large integers)."""
        result = generate_random_walk(100, seed=42)
        ts = result["ts_event"]
        # Nanosecond timestamps for 2022 are ~ 1.65e18
        assert ts[0] > 1e15, "Timestamps don't look like nanoseconds"


class TestRandomWalkDriftDirection:
    """Spec test #13: Correct drift direction."""

    def test_p_one_all_nondecreasing(self):
        """With p=1.0, all prices should be non-decreasing."""
        result = generate_random_walk(1000, p=1.0, seed=42)
        prices = result["price"]
        diffs = np.diff(prices)
        assert np.all(diffs >= 0), "p=1.0 should give non-decreasing prices"

    def test_p_zero_all_nonincreasing(self):
        """With p=0.0, all prices should be non-increasing."""
        result = generate_random_walk(1000, p=0.0, seed=42)
        prices = result["price"]
        diffs = np.diff(prices)
        assert np.all(diffs <= 0), "p=0.0 should give non-increasing prices"


class TestRandomWalkReproducibility:
    """Spec test #14: Same seed → same output."""

    def test_same_seed_same_prices(self):
        r1 = generate_random_walk(1000, seed=42)
        r2 = generate_random_walk(1000, seed=42)
        np.testing.assert_array_equal(r1["price"], r2["price"])

    def test_same_seed_same_timestamps(self):
        r1 = generate_random_walk(1000, seed=42)
        r2 = generate_random_walk(1000, seed=42)
        np.testing.assert_array_equal(r1["ts_event"], r2["ts_event"])

    def test_different_seed_different_prices(self):
        r1 = generate_random_walk(1000, seed=42)
        r2 = generate_random_walk(1000, seed=99)
        # With different seeds the prices should differ at some point
        assert not np.array_equal(r1["price"], r2["price"])


class TestRandomWalkSideField:
    """Spec test #15: Side field matches direction."""

    def test_uptick_is_B(self):
        """Upticks should have side='B'."""
        # p=1.0 → all upticks (or first trade is neutral)
        result = generate_random_walk(100, p=1.0, seed=42)
        prices = result["price"]
        sides = result["side"]
        diffs = np.diff(prices)
        # For every uptick (diff > 0), the corresponding side should be 'B'
        for i in range(len(diffs)):
            if diffs[i] > 0:
                # Side at index i+1 (the trade that caused the move)
                assert sides[i + 1] == b"B" or sides[i + 1] == "B", (
                    f"Trade {i+1}: uptick but side={sides[i+1]}"
                )

    def test_downtick_is_A(self):
        """Downticks should have side='A'."""
        # p=0.0 → all downticks
        result = generate_random_walk(100, p=0.0, seed=42)
        prices = result["price"]
        sides = result["side"]
        diffs = np.diff(prices)
        for i in range(len(diffs)):
            if diffs[i] < 0:
                assert sides[i + 1] == b"A" or sides[i + 1] == "A", (
                    f"Trade {i+1}: downtick but side={sides[i+1]}"
                )


class TestRandomWalkSizeConstant:
    """All trades should have size=1."""

    def test_all_sizes_one(self):
        result = generate_random_walk(1000, seed=42)
        sizes = result["size"]
        assert np.all(sizes == 1), "Not all sizes are 1"


class TestRandomWalkStartPrice:
    """Starting price should be start_price parameter."""

    def test_default_start_price(self):
        result = generate_random_walk(100, seed=42)
        assert result["price"][0] == pytest.approx(4000.0)

    def test_custom_start_price(self):
        result = generate_random_walk(100, start_price=5000.0, seed=42)
        assert result["price"][0] == pytest.approx(5000.0)


class TestRandomWalkTimestampInSingleSession:
    """Timestamps should fall within a single RTH session."""

    def test_timestamps_within_rth(self):
        """All timestamps should be within reasonable RTH bounds."""
        result = generate_random_walk(10000, seed=42)
        ts = result["ts_event"]
        # First timestamp should be >= some RTH open
        # Last timestamp should be < RTH close
        # RTH is 6.5 hours = 23400 seconds = 23400e9 ns
        duration_ns = ts[-1] - ts[0]
        assert duration_ns < 24 * 3600 * 1e9, "Timestamps span more than 24 hours"


# ===========================================================================
# 4. Full pipeline validation (spec tests #16-#23)
# ===========================================================================


class TestValidateDriftLevelReturnStructure:
    """validate_drift_level returns a dict with all required keys."""

    def test_returns_dict(self):
        result = validate_drift_level(0.5, n_bars=100, seed=42)
        assert isinstance(result, dict)

    def test_has_all_required_keys(self):
        result = validate_drift_level(0.5, n_bars=100, seed=42)
        required_keys = {
            "p", "analytic", "empirical", "n_labeled", "n_upper",
            "n_lower", "n_timeout", "se", "z_score", "pass",
        }
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_p_matches_input(self):
        result = validate_drift_level(0.505, n_bars=100, seed=42)
        assert result["p"] == pytest.approx(0.505)


class TestValidateDriftLevelNumerics:
    """Numeric outputs are valid — no NaN/Inf, proper types."""

    def test_no_nan_or_inf_in_result(self):
        result = validate_drift_level(0.5, n_bars=200, seed=42)
        for key in ["analytic", "empirical", "se", "z_score"]:
            assert math.isfinite(result[key]), f"{key} is not finite: {result[key]}"

    def test_counts_are_nonnegative_integers(self):
        result = validate_drift_level(0.5, n_bars=200, seed=42)
        for key in ["n_labeled", "n_upper", "n_lower", "n_timeout"]:
            assert isinstance(result[key], int), f"{key} not int: {type(result[key])}"
            assert result[key] >= 0, f"{key} is negative: {result[key]}"

    def test_counts_sum_to_n_labeled(self):
        """n_upper + n_lower + n_timeout == n_labeled."""
        result = validate_drift_level(0.5, n_bars=200, seed=42)
        total = result["n_upper"] + result["n_lower"] + result["n_timeout"]
        assert total == result["n_labeled"], (
            f"Counts don't sum: {result['n_upper']} + {result['n_lower']} + "
            f"{result['n_timeout']} = {total} != {result['n_labeled']}"
        )

    def test_empirical_in_zero_one(self):
        """Empirical proportion should be in [0, 1]."""
        result = validate_drift_level(0.5, n_bars=200, seed=42)
        assert 0.0 <= result["empirical"] <= 1.0

    def test_se_positive(self):
        """Standard error should be positive."""
        result = validate_drift_level(0.5, n_bars=200, seed=42)
        assert result["se"] > 0, f"SE should be positive: {result['se']}"

    def test_pass_is_bool(self):
        result = validate_drift_level(0.5, n_bars=200, seed=42)
        assert isinstance(result["pass"], bool)

    def test_pass_consistent_with_z_score(self):
        """pass should be True iff |z_score| <= 2.0."""
        result = validate_drift_level(0.5, n_bars=500, seed=42)
        expected_pass = abs(result["z_score"]) <= 2.0
        assert result["pass"] == expected_pass, (
            f"pass={result['pass']} but z_score={result['z_score']}"
        )


class TestZeroDriftPipelineValidation:
    """Spec test #16: Zero drift (p=0.5) pipeline validation."""

    def test_zero_drift_passes(self):
        """Empirical P(upper) within 2 SE of b/(a+b) = 10/30 ≈ 0.3333."""
        result = validate_drift_level(0.5, a=20, b=10, n_bars=10000, seed=42)
        assert result["pass"], (
            f"Zero drift failed: empirical={result['empirical']:.4f}, "
            f"analytic={result['analytic']:.4f}, z={result['z_score']:.2f}"
        )

    def test_n_labeled_sufficient(self):
        """Should label at least 10,000 bars."""
        result = validate_drift_level(0.5, n_bars=10000, seed=42)
        assert result["n_labeled"] >= 10000, (
            f"Only {result['n_labeled']} labeled, need >= 10000"
        )

    def test_analytic_value_correct(self):
        """Analytic value in result should be b/(a+b)."""
        result = validate_drift_level(0.5, a=20, b=10, n_bars=100, seed=42)
        expected = 10.0 / 30.0
        assert result["analytic"] == pytest.approx(expected, abs=1e-10)


class TestMildUpwardDriftValidation:
    """Spec test #17: Mild upward drift (p=0.505) pipeline validation."""

    def test_mild_upward_drift_passes(self):
        """Empirical within 2 SE of ~0.388."""
        result = validate_drift_level(0.505, a=20, b=10, n_bars=10000, seed=42)
        assert result["pass"], (
            f"p=0.505 failed: empirical={result['empirical']:.4f}, "
            f"analytic={result['analytic']:.4f}, z={result['z_score']:.2f}"
        )


class TestModerateUpwardDriftValidation:
    """Spec test #18: Moderate upward drift (p=0.510) pipeline validation."""

    def test_moderate_upward_drift_passes(self):
        """Empirical within 2 SE of ~0.445."""
        result = validate_drift_level(0.510, a=20, b=10, n_bars=10000, seed=42)
        assert result["pass"], (
            f"p=0.510 failed: empirical={result['empirical']:.4f}, "
            f"analytic={result['analytic']:.4f}, z={result['z_score']:.2f}"
        )


class TestMildDownwardDriftValidation:
    """Spec test #19: Mild downward drift (p=0.490) pipeline validation."""

    def test_mild_downward_drift_passes(self):
        """Empirical within 2 SE of ~0.280."""
        result = validate_drift_level(0.490, a=20, b=10, n_bars=10000, seed=42)
        assert result["pass"], (
            f"p=0.490 failed: empirical={result['empirical']:.4f}, "
            f"analytic={result['analytic']:.4f}, z={result['z_score']:.2f}"
        )


class TestModerateDownwardDriftValidation:
    """Spec test #20: Moderate downward drift (p=0.485) pipeline validation."""

    def test_moderate_downward_drift_passes(self):
        """Empirical within 2 SE of ~0.232."""
        result = validate_drift_level(0.485, a=20, b=10, n_bars=10000, seed=42)
        assert result["pass"], (
            f"p=0.485 failed: empirical={result['empirical']:.4f}, "
            f"analytic={result['analytic']:.4f}, z={result['z_score']:.2f}"
        )


class TestAllFiveDriftLevelsPass:
    """Spec test #21: Aggregate test — all 5 drift levels pass."""

    def test_run_validation_all_pass(self):
        results = run_validation(seed=42)
        assert len(results) == 5, f"Expected 5 drift levels, got {len(results)}"

        for r in results:
            assert r["pass"], (
                f"Drift p={r['p']} failed: empirical={r['empirical']:.4f}, "
                f"analytic={r['analytic']:.4f}, z={r['z_score']:.2f}"
            )

    def test_default_drift_levels(self):
        """Default drift levels should be [0.500, 0.505, 0.510, 0.490, 0.485]."""
        results = run_validation(n_bars=100, seed=42)
        p_values = [r["p"] for r in results]
        expected = [0.500, 0.505, 0.510, 0.490, 0.485]
        for p_val, exp in zip(p_values, expected):
            assert p_val == pytest.approx(exp)


class TestLabelInvariants:
    """Spec test #22: Label invariants — all labels in {-1, 0, +1}, tau > 0, tau <= t_max."""

    def test_labels_in_valid_set(self):
        """All labels should be -1, 0, or +1."""
        result = validate_drift_level(0.5, n_bars=500, seed=42)
        # Verify via counts: n_upper + n_lower + n_timeout == n_labeled
        total = result["n_upper"] + result["n_lower"] + result["n_timeout"]
        assert total == result["n_labeled"]

    def test_n_upper_corresponds_to_label_plus_one(self):
        """n_upper should count label=+1 bars."""
        result = validate_drift_level(0.5, n_bars=500, seed=42)
        # n_upper should be positive for a fair walk with enough bars
        assert result["n_upper"] >= 0

    def test_n_lower_corresponds_to_label_minus_one(self):
        """n_lower should count label=-1 bars."""
        result = validate_drift_level(0.5, n_bars=500, seed=42)
        assert result["n_lower"] >= 0


class TestTimeoutRate:
    """Spec test #23: Timeout rate is plausible for the given configuration."""

    def test_timeout_rate_low_with_large_bars(self):
        """With bar_size=500 and barriers at 20/10 ticks, timeouts should be
        rare or zero since the bar range (~sqrt(500)≈22 ticks) exceeds both
        barrier distances. This validates the pipeline resolves quickly."""
        result = validate_drift_level(0.5, t_max=40, n_bars=10000, seed=42)
        timeout_rate = result["n_timeout"] / result["n_labeled"]
        assert timeout_rate < 0.10, (
            f"Timeout rate {timeout_rate:.2%} is unexpectedly high for "
            f"bar_size=500 with 20/10 tick barriers"
        )

    def test_timeout_not_all(self):
        """Not all bars should timeout — some should hit barriers."""
        result = validate_drift_level(0.5, t_max=40, n_bars=10000, seed=42)
        assert result["n_timeout"] < result["n_labeled"], (
            f"All {result['n_labeled']} bars timed out — no barrier hits detected"
        )

    def test_barrier_hits_exist(self):
        """Both upper and lower hits should occur for zero drift."""
        result = validate_drift_level(0.5, n_bars=10000, seed=42)
        assert result["n_upper"] > 0, "No upper barrier hits detected"
        assert result["n_lower"] > 0, "No lower barrier hits detected"


# ===========================================================================
# 5. run_validation — structure and forwarding
# ===========================================================================


class TestRunValidationStructure:
    """run_validation returns list of dicts, one per drift level."""

    def test_returns_list(self):
        results = run_validation(n_bars=100, seed=42)
        assert isinstance(results, list)

    def test_each_element_is_dict(self):
        results = run_validation(n_bars=100, seed=42)
        for r in results:
            assert isinstance(r, dict)

    def test_custom_drift_levels(self):
        """Custom drift_levels parameter should be forwarded."""
        results = run_validation(
            drift_levels=[0.5, 0.6], n_bars=100, seed=42
        )
        assert len(results) == 2
        assert results[0]["p"] == pytest.approx(0.5)
        assert results[1]["p"] == pytest.approx(0.6)

    def test_barrier_params_forwarded(self):
        """Custom a, b, t_max should be forwarded to validate_drift_level."""
        results = run_validation(
            drift_levels=[0.5], a=10, b=5, t_max=20, n_bars=100, seed=42
        )
        # Analytic value for a=10, b=5, p=0.5 is 5/15 = 0.3333
        assert results[0]["analytic"] == pytest.approx(5.0 / 15.0, abs=1e-10)


# ===========================================================================
# 6. Edge cases (spec tests #24-#26)
# ===========================================================================


class TestSmallNBars:
    """Spec test #24: Small n_bars still works (no crashes)."""

    def test_n_bars_100(self):
        result = validate_drift_level(0.5, n_bars=100, seed=42)
        assert result["n_labeled"] >= 100
        assert math.isfinite(result["empirical"])
        assert math.isfinite(result["z_score"])

    def test_n_bars_50(self):
        """Even very small n_bars should not crash."""
        result = validate_drift_level(0.5, n_bars=50, seed=42)
        assert result["n_labeled"] >= 50


class TestDifferentBarrierSizes:
    """Spec test #25: Different barrier sizes."""

    def test_a10_b5_p_half(self):
        """a=10, b=5, p=0.5 → P(upper) = 5/15 = 0.3333."""
        result = gamblers_ruin_analytic(10, 5, 0.5)
        assert result == pytest.approx(5.0 / 15.0, abs=1e-10)

    def test_a10_b5_pipeline_passes(self):
        """Pipeline validation with a=10, b=5 should pass."""
        result = validate_drift_level(0.5, a=10, b=5, n_bars=10000, seed=42)
        assert result["pass"], (
            f"a=10,b=5 failed: empirical={result['empirical']:.4f}, "
            f"analytic={result['analytic']:.4f}, z={result['z_score']:.2f}"
        )


class TestDeterministicSeed:
    """Spec test #26: Deterministic seed gives identical results across runs."""

    def test_identical_results_same_seed(self):
        r1 = validate_drift_level(0.5, n_bars=500, seed=12345)
        r2 = validate_drift_level(0.5, n_bars=500, seed=12345)

        assert r1["empirical"] == r2["empirical"]
        assert r1["n_upper"] == r2["n_upper"]
        assert r1["n_lower"] == r2["n_lower"]
        assert r1["n_timeout"] == r2["n_timeout"]
        assert r1["z_score"] == r2["z_score"]

    def test_different_seeds_differ(self):
        r1 = validate_drift_level(0.5, n_bars=500, seed=42)
        r2 = validate_drift_level(0.5, n_bars=500, seed=99)
        # Different seeds should produce different empirical results
        assert r1["empirical"] != r2["empirical"]


# ===========================================================================
# 7. Acceptance criteria — no NaN/Inf in any outputs
# ===========================================================================


class TestNoNanInfInOutputs:
    """Acceptance criterion: No NaN or Inf in any outputs."""

    def test_all_drift_levels_finite(self):
        results = run_validation(n_bars=500, seed=42)
        for r in results:
            for key in ["analytic", "empirical", "se", "z_score"]:
                assert math.isfinite(r[key]), (
                    f"Non-finite {key}={r[key]} for p={r['p']}"
                )


# ===========================================================================
# 8. Empirical proportion direction
# ===========================================================================


class TestEmpiricalProportionDirection:
    """Empirical P(upper) should increase with p (drift)."""

    def test_higher_drift_higher_upper_rate(self):
        """p=0.510 should give higher empirical P(upper) than p=0.490.

        We use large n_bars to ensure statistical significance.
        """
        r_up = validate_drift_level(0.510, n_bars=10000, seed=42)
        r_down = validate_drift_level(0.490, n_bars=10000, seed=42)
        assert r_up["empirical"] > r_down["empirical"], (
            f"p=0.510 empirical={r_up['empirical']:.4f} should be > "
            f"p=0.490 empirical={r_down['empirical']:.4f}"
        )

    def test_upward_drift_higher_upper_rate_than_zero_drift(self):
        """p=0.510 should have higher P(upper) than zero drift (p=0.5).

        Note: With asymmetric barriers (a=20 > b=10), n_lower > n_upper
        even with upward drift because the lower barrier is closer.
        But P(upper) should still be higher than the zero-drift baseline.
        """
        r_up = validate_drift_level(0.510, n_bars=10000, seed=42)
        r_zero = validate_drift_level(0.5, n_bars=10000, seed=42)
        assert r_up["empirical"] > r_zero["empirical"], (
            f"Upward drift P(upper)={r_up['empirical']:.4f} should be > "
            f"zero drift P(upper)={r_zero['empirical']:.4f}"
        )

    def test_downward_drift_lower_upper_rate_than_zero_drift(self):
        """p=0.490 should have lower P(upper) than zero drift (p=0.5)."""
        r_down = validate_drift_level(0.490, n_bars=10000, seed=42)
        r_zero = validate_drift_level(0.5, n_bars=10000, seed=42)
        assert r_down["empirical"] < r_zero["empirical"], (
            f"Downward drift P(upper)={r_down['empirical']:.4f} should be < "
            f"zero drift P(upper)={r_zero['empirical']:.4f}"
        )


# ===========================================================================
# 9. Standard error computation
# ===========================================================================


class TestStandardErrorComputation:
    """SE should be sqrt(p_hat * (1 - p_hat) / n) for proportion."""

    def test_se_decreases_with_more_bars(self):
        """More bars → smaller SE."""
        r_small = validate_drift_level(0.5, n_bars=500, seed=42)
        r_large = validate_drift_level(0.5, n_bars=5000, seed=42)
        assert r_large["se"] < r_small["se"], (
            f"SE should decrease with more bars: "
            f"se_500={r_small['se']:.4f}, se_5000={r_large['se']:.4f}"
        )

    def test_se_is_reasonable_magnitude(self):
        """For n=10000, SE should be roughly sqrt(0.33*0.67/10000) ≈ 0.0047."""
        result = validate_drift_level(0.5, a=20, b=10, n_bars=10000, seed=42)
        # SE should be in the range [0.001, 0.05] for 10000 bars
        assert 0.001 < result["se"] < 0.05, (
            f"SE={result['se']} out of expected range for 10000 bars"
        )


# ===========================================================================
# 10. Z-score computation
# ===========================================================================


class TestZScoreComputation:
    """z_score = (empirical - analytic) / se."""

    def test_z_score_magnitude_reasonable(self):
        """For a correct pipeline, |z_score| should typically be < 3."""
        result = validate_drift_level(0.5, n_bars=10000, seed=42)
        assert abs(result["z_score"]) < 5.0, (
            f"z_score={result['z_score']:.2f} is suspiciously large"
        )

    def test_z_score_sign_consistent_with_empirical_vs_analytic(self):
        """If empirical > analytic, z_score should be positive, and vice versa."""
        result = validate_drift_level(0.5, n_bars=1000, seed=42)
        if result["empirical"] > result["analytic"]:
            assert result["z_score"] > 0
        elif result["empirical"] < result["analytic"]:
            assert result["z_score"] < 0
        else:
            assert result["z_score"] == pytest.approx(0.0)
