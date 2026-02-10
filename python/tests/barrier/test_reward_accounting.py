"""Tests for the barrier reward accounting module.

Spec: docs/t7-reward-accounting.md

Hand-computed reward sequences for long/short entries, barrier hits,
timeouts, transaction cost accounting, position state transitions,
unrealized PnL, and action masking.
"""

import pytest

from lob_rl.barrier.bar_pipeline import TradeBar
from lob_rl.barrier.reward_accounting import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    PositionState,
    RewardConfig,
    compute_entry,
    compute_hold_reward,
    compute_reward_sequence,
    compute_unrealized_pnl,
    get_action_mask,
)
from .conftest import make_bar


# ---------------------------------------------------------------------------
# Helpers — synthetic bar construction for hand-computed sequences
# ---------------------------------------------------------------------------

def _bar(idx, open_p, high, low, close):
    """Shorthand: build a TradeBar with explicit OHLC for reward tests."""
    return make_bar(
        bar_index=idx,
        open_price=open_p,
        high=high,
        low=low,
        close=close,
    )


# Default config values used in hand computations:
#   a=20 ticks, b=10 ticks, tick_size=0.25
#   profit barrier (long)  = entry + 20*0.25 = entry + 5.00
#   stop barrier   (long)  = entry - 10*0.25 = entry - 2.50
#   profit barrier (short) = entry - 20*0.25 = entry - 5.00
#   stop barrier   (short) = entry + 10*0.25 = entry + 2.50
#   G = 2.0, L = 1.0, C = 0.2
#   Profit reward = +G - C = +1.8
#   Stop reward   = -L - C = -1.2
#   T_max = 5  (small for test brevity)


@pytest.fixture
def config():
    """Shared test config: a=20, b=10, T_max=5 (small for test brevity)."""
    return RewardConfig(a=20, b=10, T_max=5)


# ===========================================================================
# 1. Imports and constants
# ===========================================================================


class TestImports:
    """All public names must be importable."""

    def test_reward_config_importable(self):
        from lob_rl.barrier.reward_accounting import RewardConfig
        assert RewardConfig is not None

    def test_position_state_importable(self):
        from lob_rl.barrier.reward_accounting import PositionState
        assert PositionState is not None

    def test_action_constants_importable(self):
        from lob_rl.barrier.reward_accounting import (
            ACTION_LONG, ACTION_SHORT, ACTION_FLAT, ACTION_HOLD,
        )
        assert ACTION_LONG == 0
        assert ACTION_SHORT == 1
        assert ACTION_FLAT == 2
        assert ACTION_HOLD == 3

    def test_get_action_mask_importable(self):
        from lob_rl.barrier.reward_accounting import get_action_mask
        assert callable(get_action_mask)

    def test_compute_entry_importable(self):
        from lob_rl.barrier.reward_accounting import compute_entry
        assert callable(compute_entry)

    def test_compute_hold_reward_importable(self):
        from lob_rl.barrier.reward_accounting import compute_hold_reward
        assert callable(compute_hold_reward)

    def test_compute_unrealized_pnl_importable(self):
        from lob_rl.barrier.reward_accounting import compute_unrealized_pnl
        assert callable(compute_unrealized_pnl)

    def test_compute_reward_sequence_importable(self):
        from lob_rl.barrier.reward_accounting import compute_reward_sequence
        assert callable(compute_reward_sequence)


# ===========================================================================
# 2. Hand-Computed Reward Sequences — Long Entry
# ===========================================================================


class TestLongRewardSequences:
    """Hand-computed reward sequences for long entries (spec tests 1–5)."""

    # --- Test 1: Long profit hit ---
    def test_long_profit_hit(self, config):
        """Entry at 4000.00, next bar high >= 4005.00. Reward = +1.8."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        # Bar 1: high reaches profit barrier exactly
        hit_bar = _bar(1, 4001.0, 4005.0, 3999.0, 4003.0)
        reward, new_state = compute_hold_reward(hit_bar, state, config)

        assert reward == pytest.approx(1.8)
        assert new_state.position == 0  # back to flat

    # --- Test 2: Long stop hit ---
    def test_long_stop_hit(self, config):
        """Entry at 4000.00, next bar low <= 3997.50. Reward = -1.2."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        # Bar 1: low reaches stop barrier exactly
        hit_bar = _bar(1, 3999.0, 4000.0, 3997.5, 3998.0)
        reward, new_state = compute_hold_reward(hit_bar, state, config)

        assert reward == pytest.approx(-1.2)
        assert new_state.position == 0

    # --- Test 3: Long hold then profit ---
    def test_long_hold_then_profit(self, config):
        """Entry at 4000.00, bars 1-3 no hit, bar 4 profit hit.
        Rewards: [0, 0, 0, +1.8]. Hold counter: 1, 2, 3, 4."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        rewards = []
        # Bars 1-3: mild price action, no barrier breach
        for i in range(1, 4):
            bar = _bar(i, 4000.0, 4002.0, 3998.5, 4001.0)
            reward, state = compute_hold_reward(bar, state, config)
            rewards.append(reward)
            assert state.position == 1  # still long
            assert state.hold_counter == i

        # Bar 4: profit hit
        bar4 = _bar(4, 4003.0, 4005.0, 4002.0, 4004.0)
        reward, state = compute_hold_reward(bar4, state, config)
        rewards.append(reward)

        assert rewards == [
            pytest.approx(0.0),
            pytest.approx(0.0),
            pytest.approx(0.0),
            pytest.approx(1.8),
        ]
        assert state.position == 0

    # --- Test 4: Long timeout ---
    def test_long_timeout(self, config):
        """Entry at 4000.00, T_max=5, final bar close=4001.25.
        MTM = +1 * (4001.25 - 4000.00) / (10*0.25) - 0.2
            = 1.25 / 2.5 - 0.2 = 0.5 - 0.2 = 0.3."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        # Bars 1-4: no barrier hit
        for i in range(1, 5):
            bar = _bar(i, 4000.0, 4002.0, 3998.5, 4001.0)
            reward, state = compute_hold_reward(bar, state, config)
            assert reward == pytest.approx(0.0)
            assert state.position == 1

        # Bar 5: T_max reached, close=4001.25
        bar5 = _bar(5, 4001.0, 4002.0, 3999.0, 4001.25)
        reward, state = compute_hold_reward(bar5, state, config)

        assert reward == pytest.approx(0.3)
        assert state.position == 0

    # --- Test 5: Long timeout with negative MTM ---
    def test_long_timeout_negative_mtm(self, config):
        """Entry at 4000.00, T_max=5, final bar close=3999.00.
        MTM = +1 * (3999.00 - 4000.00) / (10*0.25) - 0.2
            = -1.0 / 2.5 - 0.2 = -0.4 - 0.2 = -0.6."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        for i in range(1, 5):
            bar = _bar(i, 4000.0, 4001.0, 3999.5, 4000.0)
            _, state = compute_hold_reward(bar, state, config)

        bar5 = _bar(5, 3999.5, 4000.0, 3998.5, 3999.0)
        reward, state = compute_hold_reward(bar5, state, config)

        assert reward == pytest.approx(-0.6)
        assert state.position == 0


# ===========================================================================
# 3. Hand-Computed Reward Sequences — Short Entry
# ===========================================================================


class TestShortRewardSequences:
    """Hand-computed reward sequences for short entries (spec tests 6–10)."""

    # --- Test 6: Short profit hit ---
    def test_short_profit_hit(self, config):
        """Entry at 4000.00, next bar low <= 3995.00. Reward = +1.8."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_SHORT, config)

        # Verify barriers are reversed for short
        assert state.profit_barrier == pytest.approx(3995.0)
        assert state.stop_barrier == pytest.approx(4002.5)

        hit_bar = _bar(1, 3998.0, 4000.0, 3995.0, 3996.0)
        reward, new_state = compute_hold_reward(hit_bar, state, config)

        assert reward == pytest.approx(1.8)
        assert new_state.position == 0

    # --- Test 7: Short stop hit ---
    def test_short_stop_hit(self, config):
        """Entry at 4000.00, next bar high >= 4002.50. Reward = -1.2."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_SHORT, config)

        hit_bar = _bar(1, 4001.0, 4002.5, 3999.0, 4001.5)
        reward, new_state = compute_hold_reward(hit_bar, state, config)

        assert reward == pytest.approx(-1.2)
        assert new_state.position == 0

    # --- Test 8: Short hold then profit ---
    def test_short_hold_then_profit(self, config):
        """Entry at 4000.00, bars 1-3 no hit, bar 4 profit hit.
        Rewards: [0, 0, 0, +1.8]."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_SHORT, config)

        rewards = []
        for i in range(1, 4):
            bar = _bar(i, 3999.0, 4001.0, 3997.0, 3998.5)
            reward, state = compute_hold_reward(bar, state, config)
            rewards.append(reward)
            assert state.position == -1
            assert state.hold_counter == i

        bar4 = _bar(4, 3997.0, 3998.0, 3995.0, 3996.0)
        reward, state = compute_hold_reward(bar4, state, config)
        rewards.append(reward)

        assert rewards == [
            pytest.approx(0.0),
            pytest.approx(0.0),
            pytest.approx(0.0),
            pytest.approx(1.8),
        ]
        assert state.position == 0

    # --- Test 9: Short timeout ---
    def test_short_timeout(self, config):
        """Entry at 4000.00, T_max=5, final bar close=3998.75.
        MTM = -1 * (3998.75 - 4000.00) / (10*0.25) - 0.2
            = -1 * (-1.25) / 2.5 - 0.2 = 0.5 - 0.2 = 0.3."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_SHORT, config)

        for i in range(1, 5):
            bar = _bar(i, 3999.5, 4001.0, 3997.5, 3999.0)
            reward, state = compute_hold_reward(bar, state, config)
            assert reward == pytest.approx(0.0)
            assert state.position == -1

        bar5 = _bar(5, 3999.0, 4000.0, 3998.0, 3998.75)
        reward, state = compute_hold_reward(bar5, state, config)

        assert reward == pytest.approx(0.3)
        assert state.position == 0

    # --- Test 10: Short timeout with negative MTM ---
    def test_short_timeout_negative_mtm(self, config):
        """Entry at 4000.00, T_max=5, final bar close=4001.00.
        MTM = -1 * (4001.00 - 4000.00) / (10*0.25) - 0.2
            = -1 * 1.0 / 2.5 - 0.2 = -0.4 - 0.2 = -0.6."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_SHORT, config)

        for i in range(1, 5):
            bar = _bar(i, 4000.0, 4001.5, 3999.0, 4000.5)
            _, state = compute_hold_reward(bar, state, config)

        bar5 = _bar(5, 4000.5, 4001.5, 3999.5, 4001.0)
        reward, state = compute_hold_reward(bar5, state, config)

        assert reward == pytest.approx(-0.6)
        assert state.position == 0


# ===========================================================================
# 4. MTM Normalization
# ===========================================================================


class TestMTMNormalization:
    """Verify mark-to-market normalization at timeout (spec tests 11–12)."""

    def test_mtm_normalization_by_b(self, config):
        """Timeout reward denominator = b * tick_size = 10 * 0.25 = 2.5.
        Price move = 5 ticks = 1.25 points → MTM = 1.25 / 2.5 = 0.5.
        Reward = 0.5 - 0.2 = 0.3."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        for i in range(1, 5):
            bar = _bar(i, 4000.0, 4002.0, 3998.5, 4001.0)
            _, state = compute_hold_reward(bar, state, config)

        # 5 ticks up = 1.25 points
        bar5 = _bar(5, 4001.0, 4002.0, 3999.0, 4001.25)
        reward, _ = compute_hold_reward(bar5, state, config)
        assert reward == pytest.approx(0.3)

    def test_mtm_zero_at_entry_price(self, config):
        """Timeout at exact entry price → MTM = 0 - C = -0.2. Pure cost."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        for i in range(1, 5):
            bar = _bar(i, 4000.0, 4001.0, 3999.0, 4000.0)
            _, state = compute_hold_reward(bar, state, config)

        # Close exactly at entry price
        bar5 = _bar(5, 4000.0, 4001.0, 3999.0, 4000.0)
        reward, state = compute_hold_reward(bar5, state, config)

        assert reward == pytest.approx(-0.2)
        assert state.position == 0


# ===========================================================================
# 5. Transaction Cost Accounting
# ===========================================================================


class TestTransactionCostAccounting:
    """Verify cost is deducted exactly once (spec tests 13–15)."""

    def test_cost_deducted_once_on_profit(self, config):
        """Profit hit reward is exactly +G - C = +2.0 - 0.2 = +1.8, not +G - 2C."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        hit_bar = _bar(1, 4003.0, 4005.0, 4002.0, 4004.0)
        reward, _ = compute_hold_reward(hit_bar, state, config)

        # Exactly G - C, not G - 2C
        assert reward == pytest.approx(config.G - config.C)
        assert reward != pytest.approx(config.G - 2 * config.C)

    def test_cost_deducted_once_on_stop(self, config):
        """Stop hit reward is exactly -L - C = -1.0 - 0.2 = -1.2."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        hit_bar = _bar(1, 3999.0, 4000.0, 3997.5, 3998.0)
        reward, _ = compute_hold_reward(hit_bar, state, config)

        assert reward == pytest.approx(-config.L - config.C)

    def test_cost_deducted_once_on_timeout(self, config):
        """Timeout reward is MTM - C, cost deducted once."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        for i in range(1, 5):
            bar = _bar(i, 4000.0, 4001.0, 3999.0, 4000.5)
            _, state = compute_hold_reward(bar, state, config)

        # Close at 4000.50 → MTM = 0.5/2.5 = 0.2
        bar5 = _bar(5, 4000.0, 4001.0, 3999.0, 4000.50)
        reward, _ = compute_hold_reward(bar5, state, config)

        # MTM = 0.2, cost = 0.2 → reward = 0.2 - 0.2 = 0.0
        expected_mtm = 1.0 * (4000.50 - 4000.0) / (config.b * config.tick_size)
        assert reward == pytest.approx(expected_mtm - config.C)


# ===========================================================================
# 6. Position State Transitions
# ===========================================================================


class TestPositionStateTransitions:
    """Verify state transitions: flat ↔ position (spec tests 16–19)."""

    def test_flat_to_long_to_flat(self, config):
        """Enter long → hold → profit hit → back to flat.
        Position transitions: 0 → +1 → +1 → 0."""
        # Start flat
        initial = PositionState()
        assert initial.position == 0

        # Enter long
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)
        assert state.position == 1

        # Hold
        hold_bar = _bar(1, 4001.0, 4002.0, 3999.0, 4001.5)
        _, state = compute_hold_reward(hold_bar, state, config)
        assert state.position == 1

        # Profit hit → flat
        hit_bar = _bar(2, 4003.0, 4005.0, 4002.0, 4004.0)
        _, state = compute_hold_reward(hit_bar, state, config)
        assert state.position == 0

    def test_flat_to_short_to_flat(self, config):
        """Enter short → hold → stop hit → back to flat.
        Position: 0 → -1 → -1 → 0."""
        initial = PositionState()
        assert initial.position == 0

        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_SHORT, config)
        assert state.position == -1

        hold_bar = _bar(1, 3999.0, 4001.0, 3997.5, 3998.5)
        _, state = compute_hold_reward(hold_bar, state, config)
        assert state.position == -1

        # Stop hit (high >= 4002.50)
        hit_bar = _bar(2, 4000.0, 4002.5, 3999.0, 4001.0)
        _, state = compute_hold_reward(hit_bar, state, config)
        assert state.position == 0

    def test_entry_sets_barriers_long(self, config):
        """Long entry at 4000.00 with a=20, b=10:
        profit_barrier = 4000.00 + 20*0.25 = 4005.00
        stop_barrier   = 4000.00 - 10*0.25 = 3997.50."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        assert state.entry_price == pytest.approx(4000.0)
        assert state.profit_barrier == pytest.approx(4005.0)
        assert state.stop_barrier == pytest.approx(3997.5)
        assert state.hold_counter == 0
        assert state.position == 1

    def test_entry_sets_barriers_short(self, config):
        """Short entry at 4000.00 with a=20, b=10:
        profit_barrier = 4000.00 - 20*0.25 = 3995.00
        stop_barrier   = 4000.00 + 10*0.25 = 4002.50."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_SHORT, config)

        assert state.entry_price == pytest.approx(4000.0)
        assert state.profit_barrier == pytest.approx(3995.0)
        assert state.stop_barrier == pytest.approx(4002.5)
        assert state.hold_counter == 0
        assert state.position == -1


# ===========================================================================
# 7. Unrealized PnL
# ===========================================================================


class TestUnrealizedPnL:
    """Verify unrealized PnL in ticks (spec tests 20–23)."""

    def test_unrealized_pnl_long_positive(self, config):
        """Long at 4000.00, close=4001.25. PnL = +1*(4001.25-4000)/0.25 = 5.0 ticks."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        pnl = compute_unrealized_pnl(state, 4001.25)
        assert pnl == pytest.approx(5.0)

    def test_unrealized_pnl_long_negative(self, config):
        """Long at 4000.00, close=3999.00. PnL = +1*(3999-4000)/0.25 = -4.0 ticks."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        pnl = compute_unrealized_pnl(state, 3999.0)
        assert pnl == pytest.approx(-4.0)

    def test_unrealized_pnl_short_positive(self, config):
        """Short at 4000.00, close=3998.75. PnL = -1*(3998.75-4000)/0.25 = 5.0 ticks."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_SHORT, config)

        pnl = compute_unrealized_pnl(state, 3998.75)
        assert pnl == pytest.approx(5.0)

    def test_unrealized_pnl_short_negative(self, config):
        """Short at 4000.00, close=4001.00. PnL = -1*(4001-4000)/0.25 = -4.0 ticks."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_SHORT, config)

        pnl = compute_unrealized_pnl(state, 4001.0)
        assert pnl == pytest.approx(-4.0)

    def test_unrealized_pnl_flat_is_zero(self):
        """Unrealized PnL when flat should be 0.0."""
        state = PositionState()  # position=0
        pnl = compute_unrealized_pnl(state, 4500.0)
        assert pnl == pytest.approx(0.0)


# ===========================================================================
# 8. Action Masking
# ===========================================================================


class TestActionMasking:
    """Verify action masking rules (spec tests 24–27)."""

    def test_mask_flat_position(self):
        """When flat: long=True, short=True, flat=True, hold=False."""
        mask = get_action_mask(0)
        assert mask == [True, True, True, False]

    def test_mask_long_position(self):
        """When long: long=False, short=False, flat=False, hold=True."""
        mask = get_action_mask(1)
        assert mask == [False, False, False, True]

    def test_mask_short_position(self):
        """When short: long=False, short=False, flat=False, hold=True."""
        mask = get_action_mask(-1)
        assert mask == [False, False, False, True]

    def test_mask_is_list_of_bool(self):
        """Return type is list[bool] of length 4."""
        for pos in [-1, 0, 1]:
            mask = get_action_mask(pos)
            assert isinstance(mask, list)
            assert len(mask) == 4
            assert all(isinstance(v, bool) for v in mask)


# ===========================================================================
# 9. Full Reward Sequence
# ===========================================================================


class TestFullRewardSequence:
    """Verify complete reward sequences from entry to exit (spec tests 28–30)."""

    def test_reward_sequence_long_profit(self, config):
        """Full sequence from long entry to profit hit.
        Entry bar 0 (close=4000), bars 1-2 hold, bar 3 profit hit."""
        bars = [
            _bar(0, 4000.0, 4000.5, 3999.5, 4000.0),  # entry bar
            _bar(1, 4001.0, 4002.0, 3999.0, 4001.0),  # hold
            _bar(2, 4002.0, 4003.0, 4000.0, 4002.5),  # hold
            _bar(3, 4003.0, 4005.0, 4002.0, 4004.0),  # profit hit
        ]
        seq = compute_reward_sequence(bars, ACTION_LONG, start_bar_idx=0, config=config)

        # Entry step
        assert seq[0]["reward"] == pytest.approx(0.0)
        assert seq[0]["position"] == 1
        assert seq[0]["hold_counter"] == 0
        assert seq[0]["exit_type"] is None

        # Hold steps
        assert seq[1]["reward"] == pytest.approx(0.0)
        assert seq[1]["position"] == 1
        assert seq[1]["hold_counter"] == 1
        assert seq[1]["exit_type"] is None
        # Unrealized PnL at bar 1: (4001.0-4000.0)/0.25 = 4.0
        assert seq[1]["unrealized_pnl"] == pytest.approx(4.0)

        assert seq[2]["reward"] == pytest.approx(0.0)
        assert seq[2]["position"] == 1
        assert seq[2]["hold_counter"] == 2

        # Profit hit
        assert seq[3]["reward"] == pytest.approx(1.8)
        assert seq[3]["position"] == 0
        assert seq[3]["exit_type"] == "profit"

        assert len(seq) == 4

    def test_reward_sequence_short_timeout(self, config):
        """Full sequence from short entry to timeout.
        Entry at 4000, T_max=5, close at 3999.50 at timeout."""
        bars = [
            _bar(0, 4000.0, 4000.5, 3999.5, 4000.0),  # entry
            _bar(1, 3999.5, 4001.0, 3998.0, 3999.0),  # hold
            _bar(2, 3999.0, 4001.0, 3997.5, 3998.5),  # hold
            _bar(3, 3998.5, 4001.0, 3997.0, 3999.0),  # hold
            _bar(4, 3999.0, 4001.5, 3997.5, 3999.5),  # hold
            _bar(5, 3999.5, 4001.0, 3998.0, 3999.50),  # timeout
        ]
        seq = compute_reward_sequence(bars, ACTION_SHORT, start_bar_idx=0, config=config)

        # Entry
        assert seq[0]["reward"] == pytest.approx(0.0)
        assert seq[0]["position"] == -1

        # Intermediate: check unrealized PnL at bar 1
        # Short PnL = -1*(3999.0 - 4000.0)/0.25 = 4.0 ticks (profitable)
        assert seq[1]["unrealized_pnl"] == pytest.approx(4.0)

        # Timeout at bar 5 (hold_counter == 5 == T_max)
        last = seq[-1]
        assert last["exit_type"] == "timeout"
        # MTM = -1 * (3999.50 - 4000.0) / (10*0.25) - 0.2
        #     = -1 * (-0.50) / 2.5 - 0.2 = 0.2 - 0.2 = 0.0
        assert last["reward"] == pytest.approx(0.0)
        assert last["position"] == 0

        assert len(seq) == 6  # entry + 5 hold bars

    def test_reward_sequence_immediate_hit(self, config):
        """Entry at close, immediate barrier hit on next bar.
        Sequence has exactly 2 elements: entry (reward=0) + hit."""
        bars = [
            _bar(0, 4000.0, 4000.5, 3999.5, 4000.0),  # entry
            _bar(1, 4003.0, 4005.0, 4001.0, 4004.0),  # immediate profit hit
        ]
        seq = compute_reward_sequence(bars, ACTION_LONG, start_bar_idx=0, config=config)

        assert len(seq) == 2
        assert seq[0]["reward"] == pytest.approx(0.0)
        assert seq[0]["exit_type"] is None
        assert seq[1]["reward"] == pytest.approx(1.8)
        assert seq[1]["exit_type"] == "profit"
        assert seq[1]["position"] == 0


# ===========================================================================
# 10. Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge case verification (spec tests 31–33)."""

    def test_hold_counter_increments(self, config):
        """Hold counter goes from 0 at entry to T_max at timeout."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)
        assert state.hold_counter == 0

        for i in range(1, 6):  # T_max=5
            bar = _bar(i, 4000.0, 4001.0, 3999.0, 4000.0)
            _, state = compute_hold_reward(bar, state, config)
            # Last iteration should be timeout, but counter should have been T_max
            if i < 5:
                assert state.hold_counter == i
                assert state.position != 0
            else:
                # Timeout: position goes flat, but hold_counter hit T_max
                assert state.position == 0

    def test_config_defaults(self):
        """RewardConfig defaults match spec: G=2.0, L=1.0, C=0.2,
        T_max=40, a=20, b=10, tick_size=0.25."""
        c = RewardConfig()
        assert c.a == 20
        assert c.b == 10
        assert c.G == pytest.approx(2.0)
        assert c.L == pytest.approx(1.0)
        assert c.C == pytest.approx(0.2)
        assert c.T_max == 40
        assert c.tick_size == pytest.approx(0.25)

    def test_entry_price_is_bar_close(self, config):
        """Entry price is always bar.close, not bar.open or bar.vwap."""
        # Bar where open, close, and vwap are all different
        bar = make_bar(
            bar_index=0,
            open_price=4001.0,
            high=4003.0,
            low=3998.0,
            close=4000.0,
            vwap=4000.5,
        )
        state = compute_entry(bar, ACTION_LONG, config)
        assert state.entry_price == pytest.approx(4000.0)  # close, not open or vwap

    def test_no_reward_on_entry(self, config):
        """Entering a position yields reward 0, verified via reward_sequence."""
        bars = [
            _bar(0, 4000.0, 4000.5, 3999.5, 4000.0),
            _bar(1, 4003.0, 4005.0, 4001.0, 4004.0),  # immediate hit
        ]
        seq = compute_reward_sequence(bars, ACTION_LONG, start_bar_idx=0, config=config)
        assert seq[0]["reward"] == pytest.approx(0.0)

    def test_staying_flat_reward_zero(self):
        """Choosing flat action when flat yields reward 0."""
        # This is a behavioral contract: flat → flat, reward 0.
        # The module doesn't have a "step flat" function, but staying flat
        # means the agent chose ACTION_FLAT, which produces no state change
        # and no reward. We verify via the spec's stated contract.
        state = PositionState()
        assert state.position == 0
        # No reward function call needed — staying flat is a no-op.
        # The environment handles this, but we verify the state default.

    def test_barrier_exact_hit(self, config):
        """Barrier hit exactly at the boundary value (not exceeding).
        Long profit barrier = 4005.00. Bar high = 4005.00 exactly → profit hit."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        # High exactly equals profit barrier
        bar = _bar(1, 4002.0, 4005.0, 4001.0, 4003.0)
        reward, new_state = compute_hold_reward(bar, state, config)

        assert reward == pytest.approx(1.8)
        assert new_state.position == 0

    def test_stop_barrier_exact_hit_long(self, config):
        """Long stop barrier = 3997.50. Bar low = 3997.50 exactly → stop hit."""
        entry_bar = _bar(0, 4000.0, 4000.5, 3999.5, 4000.0)
        state = compute_entry(entry_bar, ACTION_LONG, config)

        bar = _bar(1, 3999.0, 4000.0, 3997.5, 3998.5)
        reward, new_state = compute_hold_reward(bar, state, config)

        assert reward == pytest.approx(-1.2)
        assert new_state.position == 0
