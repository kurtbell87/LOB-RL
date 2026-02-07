# Fix Precompute Event Handling

## Problem

The `spread` array from `precompute()` is negative 100% of the time because of two interacting bugs in how we handle Databento MBO event semantics. This corrupts all downstream RL training — mid-prices are wrong, execution costs can't penalize correctly, and the agent learns bid-ask bounce artifacts instead of real alpha.

## Root Cause

### Bug 1 — Snapshotting mid-event

Databento MBO events are multi-message: a single exchange event (e.g., a trade match) produces several messages (Trade, Fill, Cancel) that must ALL be processed before the book is consistent. The `F_LAST` flag (bit 7, `0x80`) marks the final message in each event. Our `precompute()` ignores this flag entirely — the `Message` struct doesn't even have a `flags` field — and snapshots after EVERY message that changes BBO. This captures transient crossed book states (`bid > ask`) mid-event.

### Bug 2 — Applying Trade/Fill to the book

Databento's spec explicitly states Trade and Fill "do not affect the book" — book changes are communicated entirely through Add, Cancel, and Modify. But `Book::apply_trade()` actively removes quantities from the book, creating intermediate state corruption mid-event.

## Required Changes (4 total)

### Change 1: Add `flags` field to `Message` struct

**File:** `include/lob/message.h`

Add `uint8_t flags = 0` to the `Message` struct. This carries the Databento event flags through the pipeline.

### Change 2: Copy `rec.flags` in `BinaryFileSource::convert()`

**File:** `src/data/binary_file_source.cpp` (around line 62)

In the `convert()` function that reads `FlatRecord` into `Message`, copy `rec.flags` to `msg.flags`. The `FlatRecord` struct already has a `flags` field at the correct offset — it's just not being copied into `Message`.

### Change 3: Skip `Action::Trade` in `Book::apply()`

**File:** `src/engine/book.cpp`

Make `Action::Trade` a no-op in `Book::apply()`. Databento says Trade/Fill don't affect the book. Our converter already remaps Fill→Trade, so both are handled by this single change. The `apply_trade()` method can remain (for backwards compat) but `apply()` should not call it for Trade actions.

### Change 4: Only snapshot on event-terminal messages in `precompute()`

**File:** `src/env/precompute.cpp`

Change the snapshot logic in `precompute()` to only capture a snapshot when ALL of:
- `msg.flags & 0x80` is set (`F_LAST` — event is complete, book is consistent)
- `!(msg.flags & 0x20)` (`F_SNAPSHOT` is NOT set — not a synthetic replay/snapshot record)
- `spread > 0` (defensive filter for the remaining ~0.007% edge cases with crossed books)

## Databento Flag Reference

```
Bit 7 (0x80): F_LAST       — last record in event for this instrument_id
Bit 6 (0x40): F_TOB        — top-of-book record, not individual order
Bit 5 (0x20): F_SNAPSHOT   — sourced from replay/snapshot server
Bit 4 (0x10): F_MBP        — aggregated price level
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

## What NOT to Change

- **Python converter (`convert_dbn.py`)** — already correctly preserves `flags` into binary files
- **Binary file format** — `FlatRecord` already has a `flags` field at the correct offset
- **Reward calculation in `precomputed_env.py`** — formula is correct, audited clean
- **Walk-forward analysis** — train/val/test split is clean, no lookahead

## Test Strategy

### C++ Unit Tests

1. **Message struct has flags field** — construct a `Message`, set `flags = 0x82`, verify it's stored and retrievable.

2. **BinaryFileSource copies flags** — create a binary file with known `FlatRecord` data including `flags = 0x82`. Read via `BinaryFileSource`, verify `msg.flags == 0x82`.

3. **Book::apply() ignores Trade actions** — build a book with known orders. Apply a Trade message. Verify the book is unchanged (best bid/ask quantities unaffected).

4. **Book::apply() still processes Add/Cancel/Modify** — verify these actions still work correctly after the Trade change.

5. **precompute() skips mid-event messages** — feed a sequence where:
   - Message 1: flags=0x00 (mid-event), changes BBO → should NOT produce a snapshot
   - Message 2: flags=0x82 (event-terminal), changes BBO → SHOULD produce a snapshot
   Verify only 1 snapshot is produced, not 2.

6. **precompute() skips snapshot messages** — feed a message with flags=0x28 (F_SNAPSHOT set). Verify it does NOT produce a snapshot row even if it changes BBO.

7. **precompute() requires positive spread** — feed an event-terminal message (flags=0x82) that results in spread ≤ 0. Verify no snapshot is produced.

8. **precompute() produces positive spreads** — feed a realistic sequence of messages with proper flags. Verify all output spread values are > 0.

### Integration Tests

9. **End-to-end with real-ish data** — construct a synthetic multi-event sequence (e.g., Add orders to build a book, then a Trade+Cancel event with proper F_LAST flags). Run through full precompute pipeline. Verify:
   - All spreads > 0
   - Mid prices are between best bid and best ask
   - Number of snapshots equals number of event-terminal messages that change BBO with positive spread

## Acceptance Criteria

1. All existing C++ tests continue to pass (no regressions)
2. New tests cover all 4 changes
3. `precompute()` output has 0% negative spreads (was 100%)
4. Mid-event messages (flags without F_LAST) never generate snapshots
5. Trade actions are no-ops in `Book::apply()`
6. Snapshot messages (F_SNAPSHOT set) never generate observation rows
