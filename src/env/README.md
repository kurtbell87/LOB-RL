# Environment Module

## Purpose
Gym-style RL environment wrapping the Book and MessageSource.

## Interface
- `LOBEnv` — defined in `include/lob/env.h`
- `reset()` — start new episode
- `step(action)` — take action, advance simulation, return observation

## Dependencies
- Depends on: `Book`, `IMessageSource`
- Depended on by: Python bindings

## Status
Not implemented.
