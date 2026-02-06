# src/data — Data Sources

Message sources that feed the Book engine. All implement `IMessageSource` (see `include/lob/source.h`).

## Files

| File | Role |
|---|---|
| `synthetic_source.h/cpp` | `SyntheticSource` — deterministic ~100 message generator for testing. Timestamps are not realistic (start at ~1ms epoch). Cannot be used with session-aware constructors. |
| `binary_file_source.h/cpp` | `BinaryFileSource(path)` — reads flat binary files (16-byte header + 36-byte records). Created by `convert_dbn.py`. Loads entire file into memory. |

## Binary file format

- Magic: "LOBR", version 1
- Header: 16 bytes (magic + version + instrument_id + record_count)
- Records: 36 bytes each (timestamp, order_id, price, qty, action, side)
