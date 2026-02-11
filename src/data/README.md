# src/data — Data Sources

Message sources that feed the Book engine. All implement `IMessageSource` (see `include/lob/source.h`).

## Files

| File | Role |
|---|---|
| `synthetic_source.h/cpp` | `SyntheticSource` — deterministic ~100 message generator for testing. Timestamps are not realistic (start at ~1ms epoch). Cannot be used with session-aware constructors. |
| `dbn_file_source.h/cpp` | `DbnFileSource(path, instrument_id)` — reads native `.mbo.dbn.zst` and legacy `.bin` files via databento-cpp. Uses shared `map_action()`/`map_side()` from `dbn_message_map`. |
| `dbn_message_map.h/cpp` | `map_mbo_to_message(MboMsg)` — converts Databento MBO records to internal `Message` format. Also exports `map_action(char, Action&)` and `map_side(char)` for shared char-to-enum mapping. Maps Databento actions (Add/Cancel/Modify/Trade/Fill/Clear→Cancel) to engine actions. |

## Dependencies

- **Depends on:** `include/lob/source.h`, `include/lob/message.h`, databento-cpp (FetchContent)
- **Depended on by:** `src/env/precompute.cpp`, `src/bindings/bindings.cpp`, C++ tests
