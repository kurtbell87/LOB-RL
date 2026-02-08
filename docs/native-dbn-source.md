# Native DBN Source

## Goal

Replace the custom `.bin` format with direct `.dbn.zst` reading via `databento-cpp`. Rip out `BinaryFileSource` and `convert_dbn.py` entirely. Design the data interface so live exchange data (Phase 4) drops in with minimal changes.

## What to build

### 1. `map_mbo_to_message()` — shared mapping function

New file: `src/data/dbn_message_map.h` (header-only or with .cpp)

```cpp
#include "lob/message.h"
#include <databento/record.hpp>

// Returns true if the record produced a valid Message, false if it should be skipped.
// Skips: Action::Clear ('R'), Action::None ('N'), non-matching instrument_id.
// Remaps: Action::Fill ('F') → Message::Action::Trade.
// Side::None → Side::Bid (default for trades with unspecified aggressor).
// Price: i64 fixed-point (1e-9) → double.
bool map_mbo_to_message(const databento::MboMsg& mbo, Message& msg,
                        uint32_t instrument_id = 0);
```

This function is shared by `DbnFileSource` now and `LiveSource` in Phase 4.

### 2. `DbnFileSource` — replaces `BinaryFileSource`

New files: `src/data/dbn_file_source.h`, `src/data/dbn_file_source.cpp`

```cpp
#include "lob/source.h"
#include <databento/dbn_file_store.hpp>

class DbnFileSource : public IMessageSource {
public:
    DbnFileSource(const std::string& path, uint32_t instrument_id = 0);
    bool next(Message& msg) override;  // uses map_mbo_to_message()
    void reset() override;             // reconstructs store_ from path_
private:
    std::string path_;
    uint32_t instrument_id_;
    databento::DbnFileStore store_;
};
```

### 3. Delete `.bin` infrastructure

Remove entirely:
- `src/data/binary_file_source.h` and `src/data/binary_file_source.cpp`
- `python/lob_rl/convert_dbn.py`
- `tests/test_binary_file_source.cpp`
- Any tests in `tests/test_data_integrity.cpp` that depend on `.bin` format

Update all references:
- `CMakeLists.txt` — remove `binary_file_source.cpp` from `lob_core`, add `dbn_file_source.cpp` and `dbn_message_map.cpp`. Link `lob_core` against `databento::databento`.
- `src/env/precompute.cpp` — replace `#include "binary_file_source.h"` with `#include "dbn_file_source.h"`. Path-based overload creates `DbnFileSource`.
- `src/bindings/bindings.cpp` — all file-based LOBEnv constructors and `precompute()` binding use `DbnFileSource`. Add `instrument_id` parameter (default 0).
- `python/lob_rl/README.md` — remove `convert_dbn.py` entry.
- `src/data/README.md` — replace `BinaryFileSource` with `DbnFileSource`.
- `data/README.md` — update to reflect `.dbn.zst` format, remove `.bin` references.

### 4. CMake changes

```cmake
cmake_minimum_required(VERSION 3.24)  # bump for databento-cpp

FetchContent_Declare(
  databento
  URL https://github.com/databento/databento-cpp/archive/refs/tags/v0.47.0.tar.gz
)
FetchContent_MakeAvailable(databento)
```

`lob_core` sources: replace `binary_file_source.cpp` with `dbn_file_source.cpp` and `dbn_message_map.cpp`. Link against `databento::databento`.

### 5. Update `precompute_cache.py`

Remove manifest-based `.bin` mode entirely. New behavior:

- `--data-dir` points to a directory of `.dbn.zst` files
- `--instrument-id` required (uint32)
- Globs for `*.mbo.dbn.zst`, extracts dates from filenames (`glbx-mdp3-YYYYMMDD.mbo.dbn.zst`)
- Calls `lob_rl_core.precompute(dbn_path, config, instrument_id)` per file
- Output `.npz` cache identical to before

### 6. `precompute()` signature update

```cpp
// Only overload — takes instrument_id for DbnFileSource
PrecomputedDay precompute(const std::string& path, const SessionConfig& cfg,
                          uint32_t instrument_id = 0);

// IMessageSource overload stays unchanged
PrecomputedDay precompute(IMessageSource& source, const SessionConfig& cfg);
```

Update `include/lob/precompute.h` accordingly.

### 7. Python bindings update

`precompute()` binding gains `instrument_id` parameter:
```python
obs, mid, spread, num_steps = lob_rl_core.precompute(path, config, instrument_id)
```

LOBEnv file-based constructors gain `instrument_id` parameter:
```python
env = lob_rl_core.LOBEnv(path, config, steps=0, instrument_id=12345, ...)
```

## Edge cases

- **Holidays/weekends:** Tiny `.dbn.zst` files. `precompute()` already handles `num_steps < 2` → skip.
- **Contract rolls:** User provides correct `instrument_id` per contract period.
- **Side::None:** Map to `Side::Bid`. Trades with unspecified aggressor still update Book correctly.
- **Non-MBO records:** `GetIf<MboMsg>()` returns nullptr → skip.
- **SyntheticSource:** Untouched. Still used for unit tests that don't need real data.

## Acceptance criteria

1. `map_mbo_to_message()` correctly maps all action/side/price fields, filters by instrument_id, skips R/N actions.
2. `DbnFileSource` reads `.dbn.zst` files and produces correct `Message` objects.
3. `DbnFileSource::reset()` re-reads from the start.
4. `LOBEnv` constructs from `.dbn.zst` path + instrument_id.
5. `precompute()` works with `.dbn.zst` paths.
6. `precompute_cache.py` builds cache from `.dbn.zst` directory.
7. `BinaryFileSource`, `convert_dbn.py`, and their tests are deleted.
8. All remaining C++ and Python tests pass.
9. New tests cover: `map_mbo_to_message()` mapping logic, `DbnFileSource` iteration/reset/filtering, precompute with `.dbn.zst`, LOBEnv with `.dbn.zst`.

## Out of scope

- Auto-resolving instrument IDs from `symbology.json`.
- Live data streaming (Phase 4 — but `map_mbo_to_message()` is ready for it).
- Deleting existing `.bin` data files from `data/mes/` (can clean up manually).
