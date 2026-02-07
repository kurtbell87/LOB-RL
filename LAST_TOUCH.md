# Last Touch — Cold-Start Briefing

## What to do next

### Immediate next step

**Fix the C++ precompute event handling.** Root cause of the broken spread data has been fully diagnosed. Write a TDD spec (`docs/fix-precompute-events.md`) and run the cycle.

#### Root cause (fully verified)

The `spread` array from `precompute()` is negative 100% of the time because of TWO interacting bugs in how we handle Databento MBO event semantics:

**Bug 1 — Snapshotting mid-event.** Databento MBO events are multi-message: a single exchange event (e.g., a trade match) produces several messages (Trade, Fill, Cancel) that must ALL be processed before the book is consistent. The `F_LAST` flag (bit 7, `0x80`) marks the final message in each event. Our `precompute()` ignores this flag entirely — the `Message` struct doesn't even have a `flags` field — and snapshots after EVERY message that changes BBO. This captures transient crossed book states (`bid > ask`) mid-event.

**Bug 2 — Applying Trade/Fill to the book.** Databento's spec explicitly states Trade and Fill "do not affect the book" — book changes are communicated entirely through Add, Cancel, and Modify. But `Book::apply_trade()` actively removes quantities from the book, creating intermediate state corruption mid-event.

**Proof:** Python simulation of the full data file with both fixes (skip trades + only snapshot on `F_LAST`) reduces crossed-book snapshots from 100% to 0.007% (331 / 4.8M). The final book state shows correct `spread = 0.25`.

#### What the spec should require (4 changes)

1. **Add `uint8_t flags = 0` to `Message` struct** (`include/lob/message.h`)
2. **Copy `rec.flags` in `BinaryFileSource::convert()`** (`src/data/binary_file_source.cpp:62`)
3. **Skip `Action::Trade` in `Book::apply()`** — make it a no-op (or remove the case entirely). Databento says Trade/Fill don't affect the book. Our converter already remaps Fill→Trade, so both are handled.
4. **Only snapshot in `precompute()` when:**
   - `msg.flags & 0x80` (F_LAST — event complete)
   - `!(msg.flags & 0x20)` (not F_SNAPSHOT — not a synthetic replay record)
   - `spread > 0` (defensive filter for remaining 0.007% edge cases)

#### Databento flag reference (from `references/dbn/rust/dbn/src/flags.rs`)

```
Bit 7 (0x80): F_LAST              — last record in event for this instrument_id
Bit 6 (0x40): F_TOB               — top-of-book record, not individual order
Bit 5 (0x20): F_SNAPSHOT           — sourced from replay/snapshot server
Bit 4 (0x10): F_MBP               — aggregated price level
Bit 3 (0x08): BAD_TS_RECV
Bit 2 (0x04): MAYBE_BAD_BOOK
Bit 1 (0x02): PUBLISHER_SPECIFIC
```

Flag distribution in real data (20241226.bin, 5.5M records):
- `0x82` (LAST | PUBLISHER_SPECIFIC): 4,800,127 — event-terminal records
- `0x00` (none): 698,555 — mid-event records (mostly Trade/Fill)
- `0x28` (SNAPSHOT | BAD_TS_RECV): 4,535 — initial book snapshot at file start
- `0x80` (LAST): 2 records
- `0xa8` (LAST | SNAPSHOT | BAD_TS_RECV): 1 record

#### What NOT to change

- **Python converter (`convert_dbn.py`)** — already correctly preserves `flags` into binary files. No changes needed.
- **Binary file format** — the `FlatRecord` struct already has a `flags` field at the correct offset. No format change.
- **Reward calculation in `precomputed_env.py`** — `reward = pos * (mid[t+1] - mid[t])` is the standard PnL-delta formulation, NOT lookahead. Agent decides position at time t, earns PnL from price move during [t, t+1].
- **Walk-forward analysis** — train/val/test split is clean. VecNormalize stats frozen at eval. Temporal features use trailing windows only. No lookahead anywhere.

#### Snapshot messages and warmup

Each day's DBN file begins with snapshot messages (`F_SNAPSHOT = 0x20`) that recreate the initial book state. These are correctly processed during warmup (pre-market phase) to build the book. They should NOT generate RL observations. The `!(flags & 0x20)` check in precompute handles this defensively for any snapshot messages that might appear during RTH.

### After the fix

1. **Re-run precompute** and verify spread is positive (~0.25 for /MES) and mid prices are realistic
2. **Re-train with `--execution-cost`** — with correct spread, execution costs will properly penalize the bid-ask bounce strategy
3. **Add coarser time sampling** — current ~4.6 steps/sec gives autocorr=-0.75 (bid-ask bounce). Need step_interval or time-based sampling.

### Training history

| Run | Config | Result |
|-----|--------|--------|
| Baseline (old pipeline) | 500k steps, ent_coef=0.0, 1 env | Sortino -1.05 val. Entropy collapsed. |
| v2 + exec cost | 2M steps, ent_coef=0.01, 8 envs, VecNormalize, exec_cost | Entropy stable (0.70). Agent learned to stay flat (mean return ~0). |
| v2 no exec cost | 2M steps, ent_coef=0.01, 8 envs, VecNormalize | Entropy collapsed (0.09). Sortino -1.05 val. Consistently negative. |
| v2 + participation bonus | 2M steps, participation_bonus=0.01 | Sortino -0.91 val. Entropy 0.17. Agent trades but picks wrong direction. |
| **Temporal features** | 2M steps, 54-dim obs, participation_bonus=0.01 | Sortino inf val. **But PnL is bid-ask bounce mean reversion, not real alpha.** Autocorr(1)=-0.75. Caused by broken spread/precompute. |

## Key files for current task

| File | Role | What to change |
|---|---|---|
| `include/lob/message.h` | `Message` struct | Add `uint8_t flags = 0` field |
| `src/data/binary_file_source.cpp` | Reads `.bin` files into `Message` | Copy `rec.flags` to `msg.flags` in `convert()` (line 62) |
| `src/data/binary_file_source.h` | `FlatRecord` struct | Already has `flags` field — no change needed |
| `src/engine/book.cpp` | `Book::apply()` and `apply_trade()` | Make `apply_trade()` a no-op (or skip Trade case in `apply()`) |
| `src/env/precompute.cpp` | `precompute()` — snapshots on BBO change | Add flag checks: only snapshot on `F_LAST`, skip `F_SNAPSHOT`, require `spread > 0` |
| `python/lob_rl/convert_dbn.py` | Python DBN→bin converter | **No changes needed** — already preserves flags correctly |
| `python/lob_rl/precomputed_env.py` | Python RL env | **No changes needed** — reward formula is correct |

## Reference material

- **Databento DBN spec cloned to `references/dbn/`** — authoritative source for MBO record layout, flag definitions, and event semantics
- Key files: `references/dbn/rust/dbn/src/flags.rs` (flag constants), `references/dbn/rust/dbn/src/record.rs` (MboMsg struct), `references/dbn/rust/dbn/src/enums.rs` (Action/Side enums)
- Databento Python library: `rec.price` returns raw int64 nanodollars (divide by 1e9 for dollars). `str(Action.ADD)` returns `'A'` (single char). Verified empirically.

## Don't waste time on

- **Build verification** — `build-release/` is current, 460 C++ tests pass.
- **Dependency checks** — SB3, gymnasium, numpy, tensorboard all installed.
- **Reading PRD.md** — everything relevant is in this file.
- **Codebase exploration** — read directory `README.md` files instead.
- **Investigating the Python converter** — it's correct. The bugs are in C++ only.
- **Investigating lookahead in reward or temporal features** — fully audited, all clean.
- **Investigating walk-forward / VecNormalize leakage** — fully audited, all clean.

## Architecture overview

```
data/mes/*.bin  →  BinaryFileSource (C++)  →  Book (C++)  →  LOBEnv (C++)
                                                                   ↓
                                                           precompute() (C++)
                                                                   ↓
                                                         numpy arrays (Python)
                                                                   ↓
                                              PrecomputedEnv + temporal features (Python)
                                                                   ↓
                                                    MultiDayEnv (Python/Gym)
                                                                   ↓
                                                      SB3 PPO (scripts/train.py)
```

## Test coverage

- **460 C++ tests** — `cd build-release && ./lob_tests`
- **696 Python tests** — `cd build-release && PYTHONPATH=.:../python uv run pytest ../python/tests/`
- **1156 total**, all passing.

## Remaining work

| Item | Priority | Notes |
|---|---|---|
| **Fix precompute event handling** | **Critical** | See "Immediate next step" above. 4 C++ changes. |
| **Coarser time sampling** | **High** | After spread fix. Step_interval or time-based sampling to reduce autocorr from -0.75. |
| Re-train with correct spread + exec cost | High | After spread fix, verify mean reversion no longer dominates |
| Hyperparameter sweep | Medium | ent_coef, participation_bonus size, LR, network arch |
| Inventory penalty experiment | Medium | `--reward-mode pnl_delta_penalized --lambda 0.01` |
| `binary_file_source.cpp:62` int64→double precision loss | Low | Low-impact for real financial data |
| DST handling for session boundaries | Low | Current dataset is entirely non-DST |

---

For project history, see `git log`.
