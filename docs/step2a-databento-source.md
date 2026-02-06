# Step 2a: Databento Data Pipeline

## Goal

Read real /MES MBO data from Databento `.dbn.zst` files and feed it into our C++ Book/LOBEnv pipeline via the `IMessageSource` interface.

## Approach

Two-stage pipeline:
1. **Python preprocessing** (`python/lob_rl/convert_dbn.py`): Uses `databento` Python library to read `.dbn.zst` files and write a simple flat binary format.
2. **C++ BinaryFileSource** (`src/data/binary_file_source.h/cpp`): Memory-maps the flat binary file and iterates records as `Message` objects via the `IMessageSource` interface.

This avoids adding zstd/DBN parsing dependencies to the C++ build.

## Component 1: Flat Binary Format

### File Layout

```
[Header: 16 bytes]
[Record 0: 40 bytes]
[Record 1: 40 bytes]
...
[Record N: 40 bytes]
```

### Header (16 bytes)

```c
struct FlatHeader {
    char     magic[4];        // "LOBR"
    uint32_t version;         // 1
    uint32_t record_count;    // number of records
    uint32_t instrument_id;   // filtered instrument_id
};
```

### Record (40 bytes, packed)

```c
struct FlatRecord {
    uint64_t ts_ns;       // ts_event nanoseconds since epoch
    uint64_t order_id;    // order ID
    int64_t  price_raw;   // price in fixed-point (raw from Databento, divide by 1e9)
    uint32_t qty;         // size
    uint8_t  action;      // 'A'=Add, 'C'=Cancel, 'M'=Modify, 'T'=Trade, 'F'=Fill, 'R'=Clear
    uint8_t  side;        // 'B'=Bid, 'A'=Ask, 'N'=None
    uint8_t  flags;       // Databento flags byte
    uint8_t  _pad;        // padding to 8-byte alignment
    uint32_t _reserved;   // future use, set to 0
};
// sizeof(FlatRecord) == 40
// static_assert(sizeof(FlatRecord) == 40)
```

## Component 2: Python Converter

**File:** `python/lob_rl/convert_dbn.py`

```
python -m lob_rl.convert_dbn \
    --input-dir /path/to/GLBX-20250125-7EGB4J87MC \
    --output-dir data/ \
    --symbol MESH5 \
    --instrument-id 42005347
```

**Behavior:**
- Reads all `.dbn.zst` files from `--input-dir`, sorted by date
- Filters to the specified `--instrument-id`
- Skips records with action='R' (clear/reset — these are snapshots, not order events)
- Writes one `.bin` file per input date file: `data/mes_20241226.bin`, `data/mes_20241227.bin`, etc.
- Also writes a `data/manifest.json` listing all output files with metadata (date, record count, first/last timestamp)

**Action mapping from Databento:**
| DBN action | FlatRecord action byte | Our Message::Action |
|------------|----------------------|---------------------|
| 'A' (Add)  | 'A' (0x41)          | Add                 |
| 'C' (Cancel) | 'C' (0x43)        | Cancel              |
| 'M' (Modify) | 'M' (0x4D)        | Modify              |
| 'T' (Trade)  | 'T' (0x54)         | Trade               |
| 'F' (Fill)   | 'F' (0x46)         | Trade (treat same)  |
| 'R' (Clear)  | skipped             | N/A                 |

**Side mapping:**
| DBN side | FlatRecord side byte |
|----------|---------------------|
| 'B'      | 'B' (0x42)         |
| 'A'      | 'A' (0x41)         |
| 'N'      | 'N' (0x4E)         |

## Component 3: C++ BinaryFileSource

**Files:** `src/data/binary_file_source.h`, `src/data/binary_file_source.cpp`

```cpp
class BinaryFileSource : public IMessageSource {
public:
    explicit BinaryFileSource(const std::string& filepath);
    ~BinaryFileSource();

    bool next(Message& msg) override;
    void reset() override;

    uint32_t record_count() const;
    uint32_t instrument_id() const;
};
```

**Behavior:**
- Constructor opens and memory-maps the file (or reads into buffer)
- Validates header magic ("LOBR") and version (1)
- `next()` reads the next `FlatRecord`, converts to `Message`:
  - `price = price_raw * 1e-9` (fixed-point to double)
  - Maps action/side bytes to `Message::Action` / `Message::Side` enums
  - Fill ('F') maps to `Message::Action::Trade`
  - Side 'N' maps to `Message::Side::Bid` (doesn't matter for clears)
- `reset()` rewinds to the first record
- Throws `std::runtime_error` on invalid header

## Update to Message

Add `Fill` handling — no enum change needed since Fill maps to Trade.

No changes to `Message` struct required; BinaryFileSource maps DBN actions internally.

## Edge Cases & Error Conditions

- File not found: throw `std::runtime_error`
- Invalid magic bytes: throw `std::runtime_error`
- Unsupported version: throw `std::runtime_error`
- Zero records: valid file, `next()` returns false immediately
- Side='N' records: map to Bid side (these are typically clear events which we skip in conversion)
- Very large files (200MB+): memory-map for efficiency, don't load all into memory

## Acceptance Criteria

1. **Python converter tests:**
   - Converts a small test DBN file successfully
   - Output binary file has correct header (magic, version, count)
   - Records match expected values from the DBN source
   - Filters by instrument_id correctly
   - Skips 'R' (clear) action records
   - Manifest JSON is written with correct metadata

2. **C++ BinaryFileSource tests:**
   - Opens a valid binary file and reads header
   - Iterates all records via `next()`
   - Returns false after last record
   - `reset()` rewinds to first record
   - Prices are correctly converted from fixed-point
   - Actions/sides are correctly mapped
   - Throws on invalid/missing file
   - Throws on bad magic bytes

3. **Integration test:**
   - Convert one day of real data, load in C++, replay through Book
   - Book has valid BBO after processing initial messages
   - No crashes processing entire day

4. **Test data:**
   - Create a small test fixture binary file (10-20 records) with known values for deterministic C++ tests
   - The Python converter tests can use the smallest real DBN file (Dec 25, ~78K records) or mock data
