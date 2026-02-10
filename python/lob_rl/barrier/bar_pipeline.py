"""Barrier bar construction pipeline.

Offline batch processor that reads raw MBO data, extracts matched trades,
and produces fixed-count trade bars with full trade sequence retention.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass
class TradeBar:
    """A fixed-count trade bar with full trade sequence retention."""
    bar_index: int
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float
    t_start: int
    t_end: int
    session_date: str
    trade_prices: np.ndarray
    trade_sizes: np.ndarray


def build_bars_from_trades(trades, n=500, session_date=""):
    """Build fixed-count trade bars from a sequence of trades.

    Groups trades into bars of exactly n trades. Discards the last
    incomplete bar if len(trades) is not a multiple of n.

    Parameters
    ----------
    trades : structured numpy array
        Must have fields: price (float64), size (int32), timestamp (int64).
    n : int
        Number of trades per bar (default 500).
    session_date : str
        Date string for the session (e.g. "2022-06-15").

    Returns
    -------
    list[TradeBar]
    """
    total = len(trades)
    n_bars = total // n
    bars = []

    for k in range(n_bars):
        start_idx = k * n
        end_idx = start_idx + n
        chunk = trades[start_idx:end_idx]

        prices = np.asarray(chunk["price"], dtype=np.float64)
        sizes = np.asarray(chunk["size"], dtype=np.int32)
        timestamps = chunk["timestamp"]

        open_price = float(prices[0])
        close_price = float(prices[-1])
        high_price = float(np.max(prices))
        low_price = float(np.min(prices))
        volume = int(np.sum(sizes))
        vwap = float(np.sum(prices * sizes) / np.sum(sizes))
        t_start = int(timestamps[0])
        t_end = int(timestamps[-1])

        bars.append(TradeBar(
            bar_index=k,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            vwap=vwap,
            t_start=t_start,
            t_end=t_end,
            session_date=session_date,
            trade_prices=prices.copy(),
            trade_sizes=sizes.copy(),
        ))

    return bars


_CHICAGO = ZoneInfo("America/Chicago")

# RTH hours in Central Time
_RTH_OPEN_HOUR = 8
_RTH_OPEN_MINUTE = 30
_RTH_CLOSE_HOUR = 15
_RTH_CLOSE_MINUTE = 0


def _rth_bounds_ns(ct_date):
    """Return (open_ns, close_ns) for RTH on the given Chicago-time date."""
    open_ct = datetime(
        ct_date.year, ct_date.month, ct_date.day,
        _RTH_OPEN_HOUR, _RTH_OPEN_MINUTE, 0,
        tzinfo=_CHICAGO,
    )
    close_ct = datetime(
        ct_date.year, ct_date.month, ct_date.day,
        _RTH_CLOSE_HOUR, _RTH_CLOSE_MINUTE, 0,
        tzinfo=_CHICAGO,
    )
    return int(open_ct.timestamp() * 1e9), int(close_ct.timestamp() * 1e9)


def filter_rth_trades(trades):
    """Filter trades to RTH session hours only.

    RTH = 8:30 AM - 3:00 PM Central Time (inclusive of open, exclusive of close).
    Handles CDT/CST transitions via America/Chicago timezone.

    Parameters
    ----------
    trades : structured numpy array
        Must have field: timestamp (int64, nanoseconds since epoch).

    Returns
    -------
    structured numpy array
        Filtered trades within RTH hours.
    """
    if len(trades) == 0:
        return trades

    timestamps = trades["timestamp"]
    mask = np.zeros(len(trades), dtype=bool)

    # Cache RTH boundaries per calendar date to avoid redundant datetime work.
    bounds_cache = {}  # ct_date -> (open_ns, close_ns)

    for i in range(len(trades)):
        ts_ns = int(timestamps[i])
        ts_seconds = ts_ns / 1e9
        dt_utc = datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
        ct_date = dt_utc.astimezone(_CHICAGO).date()

        if ct_date not in bounds_cache:
            bounds_cache[ct_date] = _rth_bounds_ns(ct_date)
        open_ns, close_ns = bounds_cache[ct_date]

        if open_ns <= ts_ns < close_ns:
            mask[i] = True

    return trades[mask]


def extract_trades_from_mbo(filepath, instrument_id=None):
    """Read a .dbn.zst file and extract matched trades.

    Parameters
    ----------
    filepath : str
        Path to .dbn.zst file.
    instrument_id : int, optional
        Filter to specific contract.

    Returns
    -------
    structured numpy array
        Fields: price (float64), size (int32), timestamp (int64), side (int8).
    """
    import databento as db

    store = db.DBNStore.from_file(filepath)
    df = store.to_df()

    # Filter to trades only (action == 'T')
    df = df[df["action"] == "T"]

    if instrument_id is not None:
        df = df[df["instrument_id"] == instrument_id]

    n = len(df)
    trades = np.zeros(n, dtype=[
        ("price", np.float64),
        ("size", np.int32),
        ("timestamp", np.int64),
        ("side", np.int8),
    ])

    if n > 0:
        trades["price"] = df["price"].values.astype(np.float64)
        trades["size"] = df["size"].values.astype(np.int32)
        # ts_event is the exchange timestamp
        if "ts_event" in df.columns:
            ts_col = df["ts_event"]
        else:
            # Fallback to index (ts_recv)
            ts_col = df.index
        trades["timestamp"] = ts_col.astype(np.int64).values

        # Map side: 'B' -> 1, 'A' -> -1, 'N' -> 0
        side_map = {"B": 1, "A": -1, "N": 0}
        trades["side"] = df["side"].map(side_map).fillna(0).values.astype(np.int8)

    return trades


def build_session_bars(filepath, n=500, instrument_id=None):
    """End-to-end: read MBO file, extract trades, filter RTH, build bars.

    Parameters
    ----------
    filepath : str
        Path to .dbn.zst file.
    n : int
        Number of trades per bar (default 500).
    instrument_id : int, optional
        Filter to specific contract.

    Returns
    -------
    list[TradeBar]
    """
    trades = extract_trades_from_mbo(filepath, instrument_id=instrument_id)
    filtered = filter_rth_trades(trades)

    # Derive session date from the filepath (e.g. "2022-06-15.mbo.dbn.zst")
    stem = Path(filepath).name
    # Try to extract date from filename
    date_str = stem.split(".")[0]  # e.g. "2022-06-15"
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        # If filename doesn't contain date, use first trade timestamp
        if len(filtered) > 0:
            ts_ns = int(filtered["timestamp"][0])
            dt = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")
        else:
            date_str = ""

    return build_bars_from_trades(filtered, n=n, session_date=date_str)


def build_dataset(filepaths, n=500, roll_calendar=None, output_path=None):
    """Batch process multiple days into a bar dataset.

    Parameters
    ----------
    filepaths : list[str]
        Paths to .dbn.zst files.
    n : int
        Number of trades per bar (default 500).
    roll_calendar : dict, optional
        Maps date string to instrument_id.
    output_path : str, optional
        If provided, writes bars.parquet and trade_sequences/ to this directory.

    Returns
    -------
    pd.DataFrame
        Columns: bar_index, open, high, low, close, volume, vwap, t_start, t_end, session_date.
    """
    all_bars = []
    all_trade_seqs = {}  # session_date -> list of (trade_prices, trade_sizes)

    for filepath in filepaths:
        # Determine instrument_id from roll calendar if provided
        instrument_id = None
        if roll_calendar is not None:
            stem = Path(filepath).name
            date_str = stem.split(".")[0]
            instrument_id = roll_calendar.get(date_str)

        bars = build_session_bars(filepath, n=n, instrument_id=instrument_id)
        all_bars.extend(bars)

        # Collect trade sequences per session
        for bar in bars:
            key = bar.session_date
            if key not in all_trade_seqs:
                all_trade_seqs[key] = []
            all_trade_seqs[key].append({
                "trade_prices": bar.trade_prices,
                "trade_sizes": bar.trade_sizes,
            })

    # Build DataFrame (without trade arrays)
    records = []
    for bar in all_bars:
        records.append({
            "bar_index": bar.bar_index,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "vwap": bar.vwap,
            "t_start": bar.t_start,
            "t_end": bar.t_end,
            "session_date": bar.session_date,
        })

    df = pd.DataFrame(records)

    if output_path is not None:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write parquet
        df.to_parquet(out_dir / "bars.parquet", index=False)

        # Write trade sequences
        seq_dir = out_dir / "trade_sequences"
        seq_dir.mkdir(exist_ok=True)
        for session_date, seqs in all_trade_seqs.items():
            prices = np.array([s["trade_prices"] for s in seqs])
            sizes = np.array([s["trade_sizes"] for s in seqs])
            np.save(seq_dir / f"{session_date}_prices.npy", prices)
            np.save(seq_dir / f"{session_date}_sizes.npy", sizes)

    return df
