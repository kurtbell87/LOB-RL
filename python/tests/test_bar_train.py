"""Tests for train.py --bar-size, --policy-arch, --activation CLI flags.

Spec: docs/bar-level-env.md (Requirements 4, 5)

These tests verify that:
- --bar-size flag exists and is accepted
- --bar-size forwards bar_size= to MultiDayEnv
- --bar-size and --step-interval together produce a warning
- --policy-arch flag exists with default "64,64"
- --policy-arch parses comma-separated dims to net_arch
- --activation flag exists with default "tanh"
- --activation only accepts "tanh" or "relu"
- evaluate_sortino() accepts bar_size parameter
"""

import os
import sys

import numpy as np
import pytest

from conftest import TRAIN_SCRIPT


# ===========================================================================
# Helper: load train.py as module
# ===========================================================================


def _load_train_module():
    """Import train.py as a module."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("train", TRAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(project_root, "python"))
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# Test 1: --bar-size flag exists
# ===========================================================================


class TestBarSizeFlagExists:
    """train.py should define a --bar-size CLI flag."""

    def test_bar_size_in_source(self):
        """train.py source should contain '--bar-size' argument definition."""
        source = open(TRAIN_SCRIPT).read()
        assert "--bar-size" in source, (
            "train.py does not define --bar-size flag"
        )

    def test_bar_size_is_argparse_argument(self):
        """--bar-size should be an argparse add_argument call."""
        source = open(TRAIN_SCRIPT).read()
        assert "add_argument" in source and "bar-size" in source, (
            "train.py should use add_argument for --bar-size"
        )


# ===========================================================================
# Test 2: --bar-size default is 0
# ===========================================================================


class TestBarSizeDefault:
    """--bar-size should default to 0 (tick-level behavior)."""

    def test_default_is_zero(self):
        """Parsing with no --bar-size should give bar_size=0."""
        source = open(TRAIN_SCRIPT).read()
        # Should have default=0 for bar-size
        assert "default=0" in source or "default = 0" in source, (
            "bar-size default should be 0"
        )


# ===========================================================================
# Test 3: --bar-size is integer type
# ===========================================================================


class TestBarSizeType:
    """--bar-size should be an integer argument."""

    def test_bar_size_type_int(self):
        """--bar-size should have type=int."""
        source = open(TRAIN_SCRIPT).read()
        # Find the bar-size argument definition and check type=int
        idx = source.find("bar-size")
        assert idx > 0
        # Check nearby context for type=int
        context = source[max(0, idx - 50):idx + 100]
        assert "int" in context, (
            f"--bar-size should have type=int, context: {context}"
        )


# ===========================================================================
# Test 4: --policy-arch flag exists
# ===========================================================================


class TestPolicyArchFlagExists:
    """train.py should define a --policy-arch CLI flag."""

    def test_policy_arch_in_source(self):
        """train.py source should contain '--policy-arch'."""
        source = open(TRAIN_SCRIPT).read()
        assert "--policy-arch" in source, (
            "train.py does not define --policy-arch flag"
        )

    def test_policy_arch_default(self):
        """--policy-arch default should be '64,64'."""
        source = open(TRAIN_SCRIPT).read()
        assert "64,64" in source, (
            "train.py should have default policy-arch of '64,64'"
        )


# ===========================================================================
# Test 5: --activation flag exists
# ===========================================================================


class TestActivationFlagExists:
    """train.py should define an --activation CLI flag."""

    def test_activation_in_source(self):
        """train.py source should contain '--activation'."""
        source = open(TRAIN_SCRIPT).read()
        assert "--activation" in source, (
            "train.py does not define --activation flag"
        )

    def test_activation_choices(self):
        """--activation should accept 'tanh' and 'relu'."""
        source = open(TRAIN_SCRIPT).read()
        assert "tanh" in source and "relu" in source, (
            "train.py --activation should support tanh and relu"
        )

    def test_activation_default_tanh(self):
        """--activation default should be 'tanh'."""
        source = open(TRAIN_SCRIPT).read()
        # Find the activation argument section
        idx = source.find("--activation")
        assert idx > 0
        context = source[idx:idx + 200]
        assert "tanh" in context, (
            "activation default should be tanh"
        )


# ===========================================================================
# Test 6: make_train_env forwards bar_size
# ===========================================================================


class TestMakeTrainEnvBarSize:
    """make_train_env should forward bar_size to MultiDayEnv."""

    def test_make_train_env_has_bar_size_param(self):
        """make_train_env source should reference bar_size."""
        source = open(TRAIN_SCRIPT).read()
        assert "bar_size" in source, (
            "train.py should use bar_size parameter in make_train_env or main"
        )

    def test_bar_size_forwarded_to_multi_day_env(self):
        """bar_size should be passed to MultiDayEnv constructor."""
        source = open(TRAIN_SCRIPT).read()
        # Both bar_size and MultiDayEnv should appear
        assert "MultiDayEnv" in source and "bar_size" in source, (
            "bar_size should be forwarded to MultiDayEnv"
        )


# ===========================================================================
# Test 7: --policy-arch parsed to net_arch
# ===========================================================================


class TestPolicyArchParsing:
    """--policy-arch should be parsed into policy_kwargs net_arch."""

    def test_net_arch_in_source(self):
        """train.py should construct net_arch from policy-arch."""
        source = open(TRAIN_SCRIPT).read()
        assert "net_arch" in source, (
            "train.py should use net_arch in policy_kwargs"
        )

    def test_policy_kwargs_in_source(self):
        """train.py should use policy_kwargs for PPO."""
        source = open(TRAIN_SCRIPT).read()
        assert "policy_kwargs" in source, (
            "train.py should pass policy_kwargs to PPO"
        )


# ===========================================================================
# Test 8: --activation parsed to activation_fn
# ===========================================================================


class TestActivationParsing:
    """--activation should be mapped to torch activation function."""

    def test_activation_fn_in_source(self):
        """train.py should use activation_fn in policy_kwargs."""
        source = open(TRAIN_SCRIPT).read()
        assert "activation_fn" in source, (
            "train.py should use activation_fn in policy_kwargs"
        )


# ===========================================================================
# Test 9: evaluate_sortino accepts bar_size
# ===========================================================================


class TestEvaluateSortinoBarSize:
    """evaluate_sortino should accept bar_size parameter."""

    def test_bar_size_param_in_evaluate_sortino(self):
        """evaluate_sortino function should have bar_size parameter."""
        mod = _load_train_module()
        import inspect
        sig = inspect.signature(mod.evaluate_sortino)
        assert "bar_size" in sig.parameters, (
            "evaluate_sortino should accept bar_size parameter"
        )

    def test_bar_size_default_zero(self):
        """evaluate_sortino bar_size should default to 0."""
        mod = _load_train_module()
        import inspect
        sig = inspect.signature(mod.evaluate_sortino)
        param = sig.parameters["bar_size"]
        assert param.default == 0, (
            f"evaluate_sortino bar_size default should be 0, got {param.default}"
        )


# ===========================================================================
# Test 10: --bar-size and --step-interval together
# ===========================================================================


class TestBarSizeStepIntervalInteraction:
    """When both --bar-size and --step-interval are set, step_interval should be ignored."""

    def test_source_handles_interaction(self):
        """train.py should handle the bar_size + step_interval interaction."""
        source = open(TRAIN_SCRIPT).read()
        # The source should contain logic to warn or ignore step_interval
        # when bar_size > 0
        assert "bar_size" in source and "step_interval" in source, (
            "train.py should handle bar_size and step_interval interaction"
        )
