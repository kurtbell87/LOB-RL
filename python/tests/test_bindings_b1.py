"""Tests for B1: Expose BinaryFileSource and SessionConfig in Python Bindings.

Spec: docs/b1-python-bindings.md

These tests verify that:
- SessionConfig is exposed with readable/writable fields and a default_rth() factory
- LOBEnv accepts file paths (BinaryFileSource-backed)
- LOBEnv accepts SessionConfig for session-aware construction
- LOBEnv exposes steps_per_episode as a read-only property
- Invalid file paths raise exceptions
- Default LOBEnv() constructor still works (backward compatibility)
"""

import os
import pytest

import lob_rl_core

from conftest import FIXTURE_DIR, EPISODE_FILE, SESSION_FILE


# ===========================================================================
# Helpers
# ===========================================================================

# C++ test fixtures for error cases
CPP_FIXTURE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "tests", "fixtures"
)


def cpp_fixture_path(name: str) -> str:
    return os.path.join(CPP_FIXTURE_DIR, name)


# ===========================================================================
# SessionConfig: Existence & Construction
# ===========================================================================


class TestSessionConfigExists:
    """SessionConfig should be exposed as a Python class."""

    def test_session_config_class_exists(self):
        """lob_rl_core should have a SessionConfig class."""
        assert hasattr(lob_rl_core, "SessionConfig")

    def test_session_config_constructible(self):
        """SessionConfig() should be constructible with no arguments."""
        cfg = lob_rl_core.SessionConfig()
        assert cfg is not None


# ===========================================================================
# SessionConfig: Field Read/Write
# ===========================================================================


class TestSessionConfigFields:
    """SessionConfig fields should be readable and writable from Python."""

    def test_rth_open_ns_readable(self):
        """rth_open_ns should be readable."""
        cfg = lob_rl_core.SessionConfig()
        _ = cfg.rth_open_ns  # Should not raise

    def test_rth_open_ns_writable(self):
        """rth_open_ns should be writable."""
        cfg = lob_rl_core.SessionConfig()
        cfg.rth_open_ns = 48_600_000_000_000
        assert cfg.rth_open_ns == 48_600_000_000_000

    def test_rth_close_ns_readable(self):
        """rth_close_ns should be readable."""
        cfg = lob_rl_core.SessionConfig()
        _ = cfg.rth_close_ns

    def test_rth_close_ns_writable(self):
        """rth_close_ns should be writable."""
        cfg = lob_rl_core.SessionConfig()
        cfg.rth_close_ns = 72_000_000_000_000
        assert cfg.rth_close_ns == 72_000_000_000_000

    def test_warmup_messages_readable(self):
        """warmup_messages should be readable."""
        cfg = lob_rl_core.SessionConfig()
        _ = cfg.warmup_messages

    def test_warmup_messages_writable(self):
        """warmup_messages should be writable."""
        cfg = lob_rl_core.SessionConfig()
        cfg.warmup_messages = 500
        assert cfg.warmup_messages == 500

    def test_warmup_messages_accepts_negative(self):
        """warmup_messages should accept -1 (use all pre-RTH)."""
        cfg = lob_rl_core.SessionConfig()
        cfg.warmup_messages = -1
        assert cfg.warmup_messages == -1

    def test_field_values_persist(self):
        """Setting multiple fields should all persist independently."""
        cfg = lob_rl_core.SessionConfig()
        cfg.rth_open_ns = 11111
        cfg.rth_close_ns = 22222
        cfg.warmup_messages = 33
        assert cfg.rth_open_ns == 11111
        assert cfg.rth_close_ns == 22222
        assert cfg.warmup_messages == 33


# ===========================================================================
# SessionConfig: default_rth() Factory
# ===========================================================================


class TestSessionConfigDefaultRTH:
    """SessionConfig.default_rth() should return standard US RTH values."""

    def test_default_rth_exists(self):
        """SessionConfig should have a default_rth() static/class method."""
        assert hasattr(lob_rl_core.SessionConfig, "default_rth")
        assert callable(lob_rl_core.SessionConfig.default_rth)

    def test_default_rth_returns_session_config(self):
        """default_rth() should return a SessionConfig instance."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        assert isinstance(cfg, lob_rl_core.SessionConfig)

    def test_default_rth_open_time(self):
        """default_rth() should set rth_open_ns to 13:30 UTC in nanoseconds."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        expected_ns = 13 * 3_600_000_000_000 + 30 * 60_000_000_000
        assert cfg.rth_open_ns == expected_ns

    def test_default_rth_close_time(self):
        """default_rth() should set rth_close_ns to 20:00 UTC in nanoseconds."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        expected_ns = 20 * 3_600_000_000_000
        assert cfg.rth_close_ns == expected_ns

    def test_default_rth_warmup(self):
        """default_rth() should set warmup_messages to -1."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        assert cfg.warmup_messages == -1


# ===========================================================================
# LOBEnv: Default Constructor (Backward Compatibility)
# ===========================================================================


class TestLOBEnvDefaultConstructor:
    """The existing default LOBEnv() constructor must still work."""

    def test_default_constructor_still_works(self):
        """LOBEnv() with no args should create a SyntheticSource env."""
        env = lob_rl_core.LOBEnv()
        assert env is not None

    def test_default_constructor_reset_works(self):
        """Default LOBEnv should support reset()."""
        env = lob_rl_core.LOBEnv()
        obs = env.reset()
        assert isinstance(obs, (list, tuple))
        assert len(obs) > 0

    def test_default_constructor_step_works(self):
        """Default LOBEnv should support step() after reset."""
        env = lob_rl_core.LOBEnv()
        env.reset()
        obs, reward, done = env.step(1)
        assert isinstance(obs, (list, tuple))
        assert isinstance(reward, float)
        assert isinstance(done, bool)


# ===========================================================================
# LOBEnv: File-Based Constructor
# ===========================================================================


class TestLOBEnvFileConstructor:
    """LOBEnv(file_path) should create an env backed by BinaryFileSource."""

    def test_file_constructor_accepts_string_path(self):
        """LOBEnv(file_path: str) should accept a file path string."""
        env = lob_rl_core.LOBEnv(EPISODE_FILE)
        assert env is not None

    def test_file_constructor_reset_returns_obs(self):
        """LOBEnv from file should return observation on reset()."""
        env = lob_rl_core.LOBEnv(EPISODE_FILE)
        obs = env.reset()
        assert isinstance(obs, (list, tuple))
        assert len(obs) > 0

    def test_file_constructor_step_works(self):
        """LOBEnv from file should support step() after reset."""
        env = lob_rl_core.LOBEnv(EPISODE_FILE)
        env.reset()
        obs, reward, done = env.step(1)
        assert isinstance(obs, (list, tuple))
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_file_constructor_full_episode(self):
        """LOBEnv from file should run a complete episode without crash."""
        env = lob_rl_core.LOBEnv(EPISODE_FILE)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            obs, reward, done = env.step(steps % 3)
            steps += 1
        assert steps > 0, "Should have taken at least one step"

    def test_file_constructor_with_steps_per_episode(self):
        """LOBEnv(file_path, steps_per_episode=N) should configure step count."""
        env = lob_rl_core.LOBEnv(EPISODE_FILE, 20)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            _, _, done = env.step(1)
            steps += 1
        # With 20 steps_per_episode, episode should end around step 20
        assert steps <= 25, f"Expected ~20 steps, got {steps}"


# ===========================================================================
# LOBEnv: File + SessionConfig Constructor
# ===========================================================================


class TestLOBEnvSessionConstructor:
    """LOBEnv(file_path, session_config, steps_per_episode) should create
    a session-aware env."""

    def test_session_constructor_accepts_args(self):
        """LOBEnv(path, config, steps) should be constructible."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        env = lob_rl_core.LOBEnv(SESSION_FILE, cfg, 50)
        assert env is not None

    def test_session_constructor_reset_works(self):
        """Session-aware LOBEnv should support reset()."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        env = lob_rl_core.LOBEnv(SESSION_FILE, cfg, 50)
        obs = env.reset()
        assert isinstance(obs, (list, tuple))
        assert len(obs) > 0

    def test_session_constructor_step_works(self):
        """Session-aware LOBEnv should support step() after reset."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        env = lob_rl_core.LOBEnv(SESSION_FILE, cfg, 50)
        env.reset()
        obs, reward, done = env.step(1)
        assert isinstance(obs, (list, tuple))
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_session_constructor_full_episode(self):
        """Session-aware LOBEnv should run a complete episode."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        env = lob_rl_core.LOBEnv(SESSION_FILE, cfg, 30)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            obs, reward, done = env.step(steps % 3)
            steps += 1
        assert steps > 0, "Should have taken at least one step"

    def test_session_constructor_with_custom_config(self):
        """LOBEnv should accept a custom SessionConfig (not just default)."""
        cfg = lob_rl_core.SessionConfig()
        cfg.rth_open_ns = 13 * 3_600_000_000_000 + 30 * 60_000_000_000
        cfg.rth_close_ns = 20 * 3_600_000_000_000
        cfg.warmup_messages = 10
        env = lob_rl_core.LOBEnv(SESSION_FILE, cfg, 30)
        obs = env.reset()
        assert isinstance(obs, (list, tuple))


# ===========================================================================
# LOBEnv: SyntheticSource with steps_per_episode
# ===========================================================================


class TestLOBEnvSyntheticSteps:
    """LOBEnv(steps_per_episode: int) should allow configuring step count
    with synthetic data."""

    def test_synthetic_with_steps_per_episode(self):
        """LOBEnv(steps_per_episode=10) with synthetic data should work."""
        env = lob_rl_core.LOBEnv(10)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            _, _, done = env.step(1)
            steps += 1
        # Episode should end around step 10
        assert steps <= 15, f"Expected ~10 steps, got {steps}"

    def test_synthetic_with_large_steps_per_episode(self):
        """LOBEnv(steps_per_episode=5) should produce short episodes."""
        env = lob_rl_core.LOBEnv(5)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            _, _, done = env.step(1)
            steps += 1
        assert steps <= 10, f"Expected ~5 steps, got {steps}"


# ===========================================================================
# LOBEnv: steps_per_episode Read-Only Property
# ===========================================================================


class TestStepsPerEpisodeProperty:
    """LOBEnv should expose steps_per_episode as a read-only property."""

    def test_steps_per_episode_exists(self):
        """LOBEnv should have a steps_per_episode attribute."""
        env = lob_rl_core.LOBEnv()
        assert hasattr(env, "steps_per_episode")

    def test_steps_per_episode_default_value(self):
        """Default LOBEnv should have steps_per_episode == 50."""
        env = lob_rl_core.LOBEnv()
        assert env.steps_per_episode == 50

    def test_steps_per_episode_reflects_constructor_arg(self):
        """steps_per_episode should match what was passed to constructor."""
        env = lob_rl_core.LOBEnv(EPISODE_FILE, 100)
        assert env.steps_per_episode == 100

    def test_steps_per_episode_from_synthetic_constructor(self):
        """steps_per_episode should match synthetic constructor arg."""
        env = lob_rl_core.LOBEnv(25)
        assert env.steps_per_episode == 25

    def test_steps_per_episode_from_session_constructor(self):
        """steps_per_episode should match session constructor arg."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        env = lob_rl_core.LOBEnv(SESSION_FILE, cfg, 75)
        assert env.steps_per_episode == 75

    def test_steps_per_episode_is_int(self):
        """steps_per_episode should return an int."""
        env = lob_rl_core.LOBEnv()
        assert isinstance(env.steps_per_episode, int)

    def test_steps_per_episode_read_only(self):
        """steps_per_episode should be read-only (raise on write)."""
        env = lob_rl_core.LOBEnv()
        with pytest.raises(AttributeError):
            env.steps_per_episode = 999


# ===========================================================================
# Error Handling: Invalid File Paths
# ===========================================================================


class TestErrorHandling:
    """Invalid inputs should raise appropriate Python exceptions."""

    def test_nonexistent_file_raises(self):
        """LOBEnv with a non-existent file path should raise an exception."""
        with pytest.raises(Exception):
            lob_rl_core.LOBEnv("/nonexistent/path/no_such_file.bin")

    def test_empty_path_raises(self):
        """LOBEnv with an empty string path should raise an exception."""
        with pytest.raises(Exception):
            lob_rl_core.LOBEnv("")

    def test_nonexistent_file_with_session_raises(self):
        """LOBEnv(bad_path, config, steps) should raise an exception."""
        cfg = lob_rl_core.SessionConfig.default_rth()
        with pytest.raises(Exception):
            lob_rl_core.LOBEnv("/nonexistent/path.bin", cfg, 50)

    def test_directory_as_path_raises(self):
        """LOBEnv with a directory path (not a file) should raise."""
        with pytest.raises(Exception):
            lob_rl_core.LOBEnv(FIXTURE_DIR)
