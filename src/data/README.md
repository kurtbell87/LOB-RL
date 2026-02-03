# Data Module

## Purpose
Message sources that feed MBO data to the Book.

## Interface
- `IMessageSource` — abstract base in `include/lob/source.h`
- `SyntheticSource` — deterministic test data
- `DatabentoSource` — real Databento MBO files (later)

## Dependencies
- Depends on: `message.h` only
- Depended on by: `LOBEnv`

## Status
Not implemented.
