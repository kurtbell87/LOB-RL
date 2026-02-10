"""Barrier reward accounting module.

Implements reward computation for the barrier-hit RL agent.
Hand-computed reward sequences for long/short entries with profit barriers,
stop barriers, timeout (mark-to-market), and transaction cost accounting.
"""

from dataclasses import dataclass, field

from lob_rl.barrier import TICK_SIZE


# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

ACTION_LONG = 0
ACTION_SHORT = 1
ACTION_FLAT = 2
ACTION_HOLD = 3


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Reward computation parameters.

    Parameters
    ----------
    a : int
        Profit barrier width in ticks (default 20).
    b : int
        Stop barrier width in ticks (default 10).
    G : float
        Profit gain reward (default 2.0).
    L : float
        Stop loss penalty magnitude (default 1.0).
    C : float
        Round-trip transaction cost (default 0.2).
    T_max : int
        Maximum holding period in bars before timeout (default 40).
    tick_size : float
        Instrument tick size in price units (default 0.25 for /MES).
    """
    a: int = 20
    b: int = 10
    G: float = 2.0
    L: float = 1.0
    C: float = 0.2
    T_max: int = 40
    tick_size: float = TICK_SIZE


# ---------------------------------------------------------------------------
# Position state
# ---------------------------------------------------------------------------

@dataclass
class PositionState:
    """Tracks current position and barrier levels.

    Attributes
    ----------
    position : int
        0 = flat, +1 = long, -1 = short.
    entry_price : float
        Price at which position was entered.
    profit_barrier : float
        Price level for profit exit.
    stop_barrier : float
        Price level for stop exit.
    hold_counter : int
        Number of bars held since entry.
    """
    position: int = 0
    entry_price: float = 0.0
    profit_barrier: float = 0.0
    stop_barrier: float = 0.0
    hold_counter: int = 0


# ---------------------------------------------------------------------------
# Action masking
# ---------------------------------------------------------------------------

def get_action_mask(position):
    """Return valid action mask for the given position.

    Parameters
    ----------
    position : int
        Current position: 0 (flat), +1 (long), -1 (short).

    Returns
    -------
    list[bool]
        Length-4 mask: [long, short, flat, hold].
        When flat: can go long, short, or stay flat. Cannot hold.
        When in position: can only hold.
    """
    if position == 0:
        return [True, True, True, False]
    else:
        return [False, False, False, True]


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def compute_entry(bar, action, config):
    """Compute position state after entering a trade.

    Entry price is always bar.close.

    Parameters
    ----------
    bar : TradeBar
        The bar on which entry occurs.
    action : int
        ACTION_LONG or ACTION_SHORT.
    config : RewardConfig
        Barrier parameters.

    Returns
    -------
    PositionState
        New state with barriers set.
    """
    entry_price = bar.close
    if action == ACTION_LONG:
        direction = 1
        profit_barrier = entry_price + config.a * config.tick_size
        stop_barrier = entry_price - config.b * config.tick_size
    else:  # ACTION_SHORT
        direction = -1
        profit_barrier = entry_price - config.a * config.tick_size
        stop_barrier = entry_price + config.b * config.tick_size

    return PositionState(
        position=direction,
        entry_price=entry_price,
        profit_barrier=profit_barrier,
        stop_barrier=stop_barrier,
        hold_counter=0,
    )


# ---------------------------------------------------------------------------
# Hold / barrier check
# ---------------------------------------------------------------------------

def compute_hold_reward(bar, state, config):
    """Compute reward for holding through one bar.

    Checks barriers (profit/stop), then timeout.

    Parameters
    ----------
    bar : TradeBar
        Current bar.
    state : PositionState
        Current position state.
    config : RewardConfig
        Reward parameters.

    Returns
    -------
    tuple[float, PositionState]
        (reward, new_state). If barrier hit or timeout, new_state.position == 0.
    """
    new_counter = state.hold_counter + 1

    if state.position == 1:  # Long
        # Profit: high >= profit_barrier
        if bar.high >= state.profit_barrier:
            reward = config.G - config.C
            return reward, PositionState(position=0)
        # Stop: low <= stop_barrier
        if bar.low <= state.stop_barrier:
            reward = -(config.L + config.C)
            return reward, PositionState(position=0)
    else:  # Short (position == -1)
        # Profit: low <= profit_barrier
        if bar.low <= state.profit_barrier:
            reward = config.G - config.C
            return reward, PositionState(position=0)
        # Stop: high >= stop_barrier
        if bar.high >= state.stop_barrier:
            reward = -(config.L + config.C)
            return reward, PositionState(position=0)

    # Timeout check
    if new_counter >= config.T_max:
        mtm = state.position * (bar.close - state.entry_price) / (config.b * config.tick_size)
        reward = mtm - config.C
        return reward, PositionState(position=0)

    # No event — hold continues
    return 0.0, PositionState(
        position=state.position,
        entry_price=state.entry_price,
        profit_barrier=state.profit_barrier,
        stop_barrier=state.stop_barrier,
        hold_counter=new_counter,
    )


# ---------------------------------------------------------------------------
# Unrealized PnL
# ---------------------------------------------------------------------------

def compute_unrealized_pnl(state, current_price):
    """Compute unrealized PnL in ticks.

    Parameters
    ----------
    state : PositionState
        Current position state.
    current_price : float
        Current market price.

    Returns
    -------
    float
        Unrealized PnL in ticks. Positive = profitable.
    """
    if state.position == 0:
        return 0.0
    return state.position * (current_price - state.entry_price) / TICK_SIZE


# ---------------------------------------------------------------------------
# Full reward sequence
# ---------------------------------------------------------------------------

def compute_reward_sequence(bars, action, start_bar_idx, config):
    """Compute the complete reward sequence from entry to exit.

    Parameters
    ----------
    bars : list[TradeBar]
        Sequence of bars starting from the entry bar.
    action : int
        ACTION_LONG or ACTION_SHORT.
    start_bar_idx : int
        Index of the entry bar in the bars list.
    config : RewardConfig
        Reward parameters.

    Returns
    -------
    list[dict]
        One dict per bar with keys: reward, position, hold_counter,
        exit_type, unrealized_pnl.
    """
    entry_bar = bars[start_bar_idx]
    state = compute_entry(entry_bar, action, config)

    sequence = [{
        "reward": 0.0,
        "position": state.position,
        "hold_counter": state.hold_counter,
        "exit_type": None,
        "unrealized_pnl": 0.0,
    }]

    for i in range(start_bar_idx + 1, len(bars)):
        bar = bars[i]
        reward, new_state = compute_hold_reward(bar, state, config)

        # Determine exit type
        if new_state.position == 0:
            if state.hold_counter + 1 >= config.T_max:
                exit_type = "timeout"
            elif reward > 0:
                exit_type = "profit"
            else:
                exit_type = "stop"
        else:
            exit_type = None

        # Unrealized PnL (use bar close for intermediate, 0 if exited)
        if new_state.position != 0:
            pnl = compute_unrealized_pnl(new_state, bar.close)
        else:
            pnl = 0.0

        sequence.append({
            "reward": reward,
            "position": new_state.position,
            "hold_counter": state.hold_counter + 1,
            "exit_type": exit_type,
            "unrealized_pnl": pnl,
        })

        state = new_state
        if state.position == 0:
            break

    return sequence
