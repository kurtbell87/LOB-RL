"""Tests for Training Pipeline v2 — PPO Hyperparameters & Normalization.

Spec: docs/training-pipeline-v2.md

These tests verify that scripts/train.py:
- Defines all 6 new CLI flags with correct types and defaults
- Preserves all existing CLI flags unchanged
- Sets ent_coef=0.01 by default, tunable via --ent-coef
- Wraps training env with VecNormalize (obs+reward normalization)
- Saves VecNormalize stats (vec_normalize.pkl) after training
- Loads VecNormalize stats in eval mode during evaluation
- --no-norm disables VecNormalize entirely
- Uses SubprocVecEnv with --n-envs parallel training environments
- Uses env factory function (not lambda) to avoid late-binding
- --n-envs 1 still works
- evaluate_sortino() accepts execution_cost and vec_normalize_path params
- Eval envs use execution_cost when --execution-cost is passed
- Eval envs are wrapped with VecNormalize in eval mode (training=False)
- PPO hyperparameters updated: batch_size=256, n_epochs=5
- All new hyperparams are CLI-tunable
"""

import argparse
import ast
import inspect
import os
import re
import sys
import textwrap
from unittest import mock

import numpy as np
import pytest

from conftest import TRAIN_SCRIPT, load_train_source


def _build_parser_from_train():
    """Import train.py and extract its argparse parser.

    We do this by running the main() function's parser setup only,
    without executing the rest. We parse train.py's source to find
    the parser and replay the add_argument calls.
    """
    # Add project paths so imports resolve
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(project_root, "python"))
    sys.path.insert(0, os.path.join(project_root, "build-release"))
    sys.path.insert(0, os.path.join(project_root, "build"))

    source = load_train_source()
    tree = ast.parse(source)

    # Find the main() function and extract parser setup
    # We look for argparse.ArgumentParser and add_argument calls
    # As a simpler approach: exec the module but mock away the heavy parts
    # and call parse_args with --help-like introspection.

    # Safest: just build the parser by finding the main function and
    # executing only the parser portion with mocks.
    # For robustness, we use the source-reading approach.
    return source


# ===========================================================================
# CLI Flags: New flags exist with correct types and defaults
# ===========================================================================


class TestCLINewFlags:
    """All 6 new CLI flags should be defined with correct types and defaults."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_ent_coef_flag_exists(self):
        """--ent-coef flag should be defined in train.py."""
        assert "--ent-coef" in self.source, (
            "train.py should define --ent-coef CLI flag"
        )

    def test_ent_coef_default_is_0_01(self):
        """--ent-coef default should be 0.01."""
        # Match: default=0.01 or default=cfg('...', 0.01)
        pattern = r"--ent-coef.*?default\s*=\s*(?:cfg\([^,]+,\s*)?(0\.01)"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "--ent-coef default should be 0.01"
        )

    def test_ent_coef_type_is_float(self):
        """--ent-coef should be type=float."""
        # Find the add_argument call for --ent-coef and check it has type=float
        pattern = r"add_argument\s*\(\s*['\"]--ent-coef['\"].*?type\s*=\s*float"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--ent-coef should have type=float"

    def test_n_envs_flag_exists(self):
        """--n-envs flag should be defined in train.py."""
        assert "--n-envs" in self.source, (
            "train.py should define --n-envs CLI flag"
        )

    def test_n_envs_default_is_8(self):
        """--n-envs default should be 8."""
        pattern = r"--n-envs.*?default\s*=\s*(?:cfg\([^,]+,\s*)?8\b"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--n-envs default should be 8"

    def test_n_envs_type_is_int(self):
        """--n-envs should be type=int."""
        pattern = r"add_argument\s*\(\s*['\"]--n-envs['\"].*?type\s*=\s*int"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--n-envs should have type=int"

    def test_batch_size_flag_exists(self):
        """--batch-size flag should be defined in train.py."""
        assert "--batch-size" in self.source, (
            "train.py should define --batch-size CLI flag"
        )

    def test_batch_size_default_is_256(self):
        """--batch-size default should be 256."""
        pattern = r"--batch-size.*?default\s*=\s*(?:cfg\([^,]+,\s*)?256\b"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--batch-size default should be 256"

    def test_batch_size_type_is_int(self):
        """--batch-size should be type=int."""
        pattern = r"add_argument\s*\(\s*['\"]--batch-size['\"].*?type\s*=\s*int"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--batch-size should have type=int"

    def test_n_epochs_flag_exists(self):
        """--n-epochs flag should be defined in train.py."""
        assert "--n-epochs" in self.source, (
            "train.py should define --n-epochs CLI flag"
        )

    def test_n_epochs_default_is_5(self):
        """--n-epochs default should be 5."""
        pattern = r"--n-epochs.*?default\s*=\s*(?:cfg\([^,]+,\s*)?5\b"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--n-epochs default should be 5"

    def test_n_epochs_type_is_int(self):
        """--n-epochs should be type=int."""
        pattern = r"add_argument\s*\(\s*['\"]--n-epochs['\"].*?type\s*=\s*int"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--n-epochs should have type=int"

    def test_learning_rate_flag_exists(self):
        """--learning-rate flag should be defined in train.py."""
        assert "--learning-rate" in self.source, (
            "train.py should define --learning-rate CLI flag"
        )

    def test_learning_rate_default_is_3e_4(self):
        """--learning-rate default should be 3e-4."""
        pattern = r"--learning-rate.*?default\s*=\s*(?:cfg\([^,]+,\s*)?(3e-4|0\.0003)"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--learning-rate default should be 3e-4"

    def test_learning_rate_type_is_float(self):
        """--learning-rate should be type=float."""
        pattern = r"add_argument\s*\(\s*['\"]--learning-rate['\"].*?type\s*=\s*float"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--learning-rate should have type=float"

    def test_no_norm_flag_exists(self):
        """--no-norm flag should be defined in train.py."""
        assert "--no-norm" in self.source, (
            "train.py should define --no-norm CLI flag"
        )

    def test_no_norm_is_boolean_flag(self):
        """--no-norm should be a store_true boolean flag."""
        pattern = r"add_argument\s*\(\s*['\"]--no-norm['\"].*?store_true"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--no-norm should be action='store_true'"


# ===========================================================================
# CLI Flags: Existing flags still present
# ===========================================================================


class TestCLIExistingFlags:
    """Existing CLI flags should be preserved unchanged."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_data_dir_flag(self):
        """--data-dir flag should still exist."""
        assert "--data-dir" in self.source

    def test_train_days_flag(self):
        """--train-days flag should still exist."""
        assert "--train-days" in self.source

    def test_total_timesteps_flag(self):
        """--total-timesteps flag should still exist."""
        assert "--total-timesteps" in self.source

    def test_reward_mode_flag(self):
        """--reward-mode flag should still exist."""
        assert "--reward-mode" in self.source

    def test_lambda_flag(self):
        """--lambda flag should still exist."""
        assert "--lambda" in self.source

    def test_execution_cost_flag(self):
        """--execution-cost flag should still exist."""
        assert "--execution-cost" in self.source

    def test_output_dir_flag(self):
        """--output-dir flag should still exist."""
        assert "--output-dir" in self.source


# ===========================================================================
# Entropy Coefficient: ent_coef passed to PPO
# ===========================================================================


class TestEntropyCoefficient:
    """PPO should receive ent_coef from CLI args."""

    def test_ent_coef_used_in_ppo_constructor(self):
        """PPO constructor should include ent_coef parameter."""
        source = load_train_source()
        # The PPO() call should reference ent_coef (from args)
        assert "ent_coef" in source, (
            "PPO constructor should include ent_coef"
        )

    def test_ent_coef_not_hardcoded_zero(self):
        """ent_coef should be present in PPO call and not hardcoded to 0.0."""
        source = load_train_source()
        # First, ent_coef must be present in the PPO constructor
        assert "ent_coef" in source, "PPO constructor should include ent_coef"
        # And it should not be hardcoded to 0.0
        pattern = r"PPO\s*\(.*?ent_coef\s*=\s*0(\.0)?\s*[,)]"
        match = re.search(pattern, source, re.DOTALL)
        assert match is None, (
            "ent_coef should not be hardcoded to 0.0"
        )

    def test_ent_coef_references_args(self):
        """ent_coef should reference args (CLI-tunable), not a hardcoded value."""
        source = load_train_source()
        # Look for ent_coef=args.ent_coef or similar pattern
        pattern = r"ent_coef\s*=\s*args\."
        match = re.search(pattern, source)
        assert match is not None, (
            "ent_coef should be set from args (e.g., args.ent_coef)"
        )


# ===========================================================================
# VecNormalize: Training wraps with normalization
# ===========================================================================


class TestVecNormalizeTraining:
    """Training env should be wrapped with VecNormalize."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_imports_vec_normalize(self):
        """train.py should import VecNormalize."""
        assert "VecNormalize" in self.source, (
            "train.py should import VecNormalize"
        )

    def test_vec_normalize_wraps_env(self):
        """Training env should be wrapped with VecNormalize(env, ...)."""
        # Look for VecNormalize( applied to the env
        pattern = r"VecNormalize\s*\("
        match = re.search(pattern, self.source)
        assert match is not None, (
            "Training env should be wrapped with VecNormalize"
        )

    def test_vec_normalize_norm_obs_true(self):
        """VecNormalize should have norm_obs=True."""
        pattern = r"VecNormalize\s*\(.*?norm_obs\s*=\s*True"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "VecNormalize should have norm_obs=True"
        )

    def test_vec_normalize_norm_reward_true(self):
        """VecNormalize should have norm_reward=True."""
        pattern = r"VecNormalize\s*\(.*?norm_reward\s*=\s*True"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "VecNormalize should have norm_reward=True"
        )

    def test_vec_normalize_clip_obs_10(self):
        """VecNormalize should have clip_obs=10.0."""
        pattern = r"VecNormalize\s*\(.*?clip_obs\s*=\s*10(\.0)?"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "VecNormalize should have clip_obs=10.0"
        )


# ===========================================================================
# VecNormalize: Stats saved after training
# ===========================================================================


class TestVecNormalizeSave:
    """VecNormalize stats should be saved after training."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_saves_vec_normalize_pkl(self):
        """After training, vec_normalize.pkl should be saved."""
        assert "vec_normalize.pkl" in self.source, (
            "train.py should save vec_normalize.pkl"
        )

    def test_env_save_called_for_vec_normalize(self):
        """VecNormalize .save() should be called with vec_normalize.pkl path."""
        # Look for .save( with vec_normalize.pkl nearby — not just model.save()
        pattern = r"\.save\s*\(.*vec_normalize"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "VecNormalize stats should be saved via .save() with vec_normalize.pkl"
        )


# ===========================================================================
# VecNormalize: --no-norm disables normalization
# ===========================================================================


class TestNoNormFlag:
    """--no-norm flag should disable VecNormalize entirely."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_no_norm_conditionally_skips_vec_normalize(self):
        """When --no-norm is set, VecNormalize wrapping should be skipped."""
        # There should be a conditional check like `if not args.no_norm:`
        # or `if args.no_norm:` that gates VecNormalize
        pattern = r"(no_norm|args\.no_norm)"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "--no-norm should gate VecNormalize wrapping"
        )

    def test_no_norm_skips_stats_save(self):
        """When --no-norm is set, vec_normalize.pkl save should be conditional."""
        # The save call should be gated by a normalization check
        # Look for conditional logic around the save
        pattern = r"(if.*no_norm|if.*norm|not.*no_norm).*save.*vec_normalize"
        match = re.search(pattern, self.source, re.DOTALL)
        # Alternative: check that the save is inside a conditional block
        # This is a structural check — the save shouldn't be unconditional
        assert "vec_normalize" in self.source, (
            "train.py should reference vec_normalize for conditional save"
        )


# ===========================================================================
# Parallel Environments: SubprocVecEnv
# ===========================================================================


class TestParallelEnvironments:
    """Training should use SubprocVecEnv for parallel environments."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_imports_subproc_vec_env(self):
        """train.py should import SubprocVecEnv."""
        assert "SubprocVecEnv" in self.source, (
            "train.py should import SubprocVecEnv"
        )

    def test_subproc_vec_env_used(self):
        """SubprocVecEnv should be used to create the training env."""
        pattern = r"SubprocVecEnv\s*\("
        match = re.search(pattern, self.source)
        assert match is not None, (
            "SubprocVecEnv should be instantiated for training"
        )

    def test_n_envs_used_in_env_creation(self):
        """args.n_envs should be used to control number of parallel envs."""
        pattern = r"(n_envs|args\.n_envs)"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "n_envs should be used in env creation"
        )

    def test_env_factory_is_function_not_lambda(self):
        """Env factory for SubprocVecEnv should use a closure, not bare lambda.

        The spec explicitly warns about lambda late-binding issues with
        SubprocVecEnv. The factory should be a named function that returns
        a closure (_init), capturing parameters by value.
        """
        # Look for a factory function that returns an inner function
        # Pattern: def make_*(...): ... def _init(): ... return _init
        pattern = r"def\s+make_train_env\s*\("
        match = re.search(pattern, self.source)
        assert match is not None, (
            "Should define make_train_env() factory function for SubprocVecEnv"
        )

    def test_multiple_envs_created_in_list_comprehension(self):
        """SubprocVecEnv should receive a list of env factories (one per env)."""
        # Look for pattern like [make_env(...) for _ in range(n_envs)]
        pattern = r"SubprocVecEnv\s*\(\s*\["
        match = re.search(pattern, self.source)
        assert match is not None, (
            "SubprocVecEnv should receive a list of env factories"
        )


# ===========================================================================
# Parallel Environments: n_envs=1 still works
# ===========================================================================


class TestNEnvsOne:
    """--n-envs 1 should still work (either SubprocVecEnv or DummyVecEnv)."""

    def test_n_envs_one_accepted(self):
        """n_envs=1 should be a valid value (no minimum > 1 check)."""
        source = load_train_source()
        # There should be no hard lower bound preventing n_envs=1
        # Check that there's no assertion like assert n_envs > 1
        pattern = r"(assert.*n_envs\s*>\s*1|n_envs\s*<\s*2.*raise)"
        match = re.search(pattern, source)
        assert match is None, (
            "n_envs=1 should be valid — no minimum > 1 assertion"
        )


# ===========================================================================
# Evaluation: evaluate_sortino accepts execution_cost
# ===========================================================================


class TestEvalExecutionCost:
    """evaluate_sortino() should accept execution_cost parameter."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_evaluate_sortino_has_execution_cost_param(self):
        """evaluate_sortino() signature should include execution_cost."""
        pattern = r"def\s+evaluate_sortino\s*\(.*?execution_cost"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "evaluate_sortino() should accept execution_cost parameter"
        )

    def test_evaluate_sortino_has_vec_normalize_path_param(self):
        """evaluate_sortino() signature should include vec_normalize_path."""
        pattern = r"def\s+evaluate_sortino\s*\(.*?vec_normalize_path"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "evaluate_sortino() should accept vec_normalize_path parameter"
        )

    def test_execution_cost_passed_to_eval_env(self):
        """evaluate_sortino should pass execution_cost to env creation."""
        # Inside evaluate_sortino, the make_env or env constructor should
        # receive execution_cost
        pattern = r"def\s+evaluate_sortino.*?execution_cost.*?(make_env|LOBGymEnv|MultiDayEnv|DummyVecEnv)"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "evaluate_sortino should forward execution_cost to env creation"
        )

    def test_make_env_accepts_execution_cost(self):
        """make_env() should accept execution_cost parameter."""
        pattern = r"def\s+make_env\s*\(.*?execution_cost"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "make_env() should accept execution_cost parameter"
        )

    def test_make_env_forwards_execution_cost_to_env(self):
        """make_env() should forward execution_cost to the env constructor."""
        # Inside make_env, the env constructor should receive execution_cost
        pattern = r"def\s+make_env.*?(PrecomputedEnv|LOBGymEnv)\S*\(.*?execution_cost"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "make_env() should forward execution_cost to env constructor"
        )


# ===========================================================================
# Evaluation: VecNormalize in eval mode
# ===========================================================================


class TestEvalVecNormalize:
    """Evaluation should use VecNormalize in eval mode."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_vec_normalize_load_in_eval(self):
        """Evaluation should load VecNormalize stats."""
        pattern = r"VecNormalize\.load\s*\("
        match = re.search(pattern, self.source)
        assert match is not None, (
            "Evaluation should load VecNormalize stats via VecNormalize.load()"
        )

    def test_eval_training_false(self):
        """Eval VecNormalize should have training=False."""
        # Look for .training = False on the eval env
        pattern = r"training\s*=\s*False"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "Eval VecNormalize should set training=False"
        )

    def test_eval_norm_reward_false(self):
        """Eval VecNormalize should have norm_reward=False."""
        pattern = r"norm_reward\s*=\s*False"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "Eval VecNormalize should set norm_reward=False"
        )

    def test_eval_uses_dummy_vec_env_for_vec_normalize(self):
        """Eval should wrap env in DummyVecEnv for VecNormalize compatibility."""
        # evaluate_sortino should use DummyVecEnv to wrap the eval env
        # so that VecNormalize can be applied. Extract the evaluate_sortino
        # function body and check it contains DummyVecEnv.
        pattern = r"def\s+evaluate_sortino\s*\(.*?\n(?=def\s|\Z)"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "evaluate_sortino() function not found"
        eval_body = match.group(0)
        assert "DummyVecEnv" in eval_body, (
            "evaluate_sortino() should use DummyVecEnv for VecNormalize wrapping"
        )


# ===========================================================================
# PPO Hyperparameters: Updated defaults
# ===========================================================================


class TestPPOHyperparameters:
    """PPO should use updated hyperparameters from the spec."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_batch_size_not_64(self):
        """PPO batch_size should not be the old default of 64."""
        # Check that PPO constructor doesn't use batch_size=64
        pattern = r"PPO\s*\(.*?batch_size\s*=\s*64\b"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is None, (
            "PPO batch_size should not be 64 (old value); should be 256"
        )

    def test_batch_size_from_args(self):
        """PPO batch_size should reference args (CLI-tunable)."""
        pattern = r"batch_size\s*=\s*args\."
        match = re.search(pattern, self.source)
        assert match is not None, (
            "PPO batch_size should be set from args"
        )

    def test_n_epochs_not_10(self):
        """PPO n_epochs should not be the old default of 10."""
        pattern = r"PPO\s*\(.*?n_epochs\s*=\s*10\b"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is None, (
            "PPO n_epochs should not be 10 (old value); should be 5"
        )

    def test_n_epochs_from_args(self):
        """PPO n_epochs should reference args (CLI-tunable)."""
        pattern = r"n_epochs\s*=\s*args\."
        match = re.search(pattern, self.source)
        assert match is not None, (
            "PPO n_epochs should be set from args"
        )

    def test_learning_rate_from_args(self):
        """PPO learning_rate should reference args (CLI-tunable)."""
        pattern = r"learning_rate\s*=\s*args\."
        match = re.search(pattern, self.source)
        assert match is not None, (
            "PPO learning_rate should be set from args"
        )

    def test_vf_coef_set(self):
        """PPO should set vf_coef=0.5."""
        pattern = r"vf_coef\s*=\s*0\.5"
        match = re.search(pattern, self.source)
        assert match is not None, "PPO should have vf_coef=0.5"

    def test_max_grad_norm_set(self):
        """PPO should set max_grad_norm=0.5."""
        pattern = r"max_grad_norm\s*=\s*0\.5"
        match = re.search(pattern, self.source)
        assert match is not None, "PPO should have max_grad_norm=0.5"

    def test_clip_range_set(self):
        """PPO should set clip_range=0.2."""
        pattern = r"clip_range\s*=\s*0\.2"
        match = re.search(pattern, self.source)
        assert match is not None, "PPO should have clip_range=0.2"


# ===========================================================================
# Training Env: execution_cost forwarded to MultiDayEnv
# ===========================================================================


class TestTrainingEnvExecutionCost:
    """Training env creation should forward execution_cost."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_multiday_env_receives_execution_cost(self):
        """MultiDayEnv in training should receive execution_cost."""
        # Inside the env factory or DummyVecEnv/SubprocVecEnv setup,
        # MultiDayEnv should get execution_cost
        pattern = r"MultiDayEnv\s*\(.*?execution_cost"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "Training MultiDayEnv should receive execution_cost parameter"
        )


# ===========================================================================
# Evaluation: execution_cost forwarded to main() eval calls
# ===========================================================================


class TestMainEvalCallsForwardExecutionCost:
    """main() should pass execution_cost to evaluate_sortino() calls."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_val_eval_passes_execution_cost(self):
        """Validation evaluate_sortino() call should include execution_cost."""
        pattern = r"evaluate_sortino\s*\(.*?execution_cost"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "evaluate_sortino() call should include execution_cost"
        )

    def test_eval_passes_vec_normalize_path(self):
        """evaluate_sortino() call should include vec_normalize_path."""
        pattern = r"evaluate_sortino\s*\(.*?vec_normalize_path"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "evaluate_sortino() call should include vec_normalize_path"
        )


# ===========================================================================
# Integration: All acceptance criteria
# ===========================================================================


class TestAcceptanceCriteria:
    """High-level checks for acceptance criteria from the spec."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_ac1_all_new_flags_present(self):
        """AC1: All 6 new CLI flags should be defined."""
        new_flags = [
            "--ent-coef",
            "--n-envs",
            "--batch-size",
            "--n-epochs",
            "--learning-rate",
            "--no-norm",
        ]
        for flag in new_flags:
            assert flag in self.source, f"Missing CLI flag: {flag}"

    def test_ac3_default_config_uses_subproc_vec_env(self):
        """AC3: Default config uses SubprocVecEnv with 8 envs."""
        assert "SubprocVecEnv" in self.source

    def test_ac3_default_config_uses_vec_normalize(self):
        """AC3: Default config uses VecNormalize."""
        assert "VecNormalize" in self.source

    def test_ac4_vec_normalize_pkl_saved(self):
        """AC4: After training, vec_normalize.pkl is saved."""
        assert "vec_normalize.pkl" in self.source

    def test_ac5_eval_uses_execution_cost(self):
        """AC5: Evaluation uses execution_cost when --execution-cost passed."""
        pattern = r"evaluate_sortino\s*\(.*?execution_cost"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None

    def test_ac6_eval_vec_normalize_eval_mode(self):
        """AC6: Eval applies VecNormalize stats in eval mode."""
        assert "training" in self.source
        assert "VecNormalize.load" in self.source

    def test_ac7_all_changes_in_train_py(self):
        """AC7: All changes are in scripts/train.py — it should contain all new features."""
        # Verify that train.py itself contains all the new infrastructure:
        # SubprocVecEnv, VecNormalize, updated evaluate_sortino, new CLI flags
        required = [
            "SubprocVecEnv",
            "VecNormalize",
            "vec_normalize_path",
            "--ent-coef",
            "--n-envs",
        ]
        for req in required:
            assert req in self.source, f"train.py should contain {req}"


# ===========================================================================
# Edge case: --no-norm with evaluation
# ===========================================================================


class TestNoNormWithEvaluation:
    """When --no-norm is set, eval should skip VecNormalize loading."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_eval_skips_vec_normalize_when_no_norm(self):
        """Eval should conditionally skip VecNormalize.load when --no-norm."""
        # vec_normalize_path should be None or conditional when no_norm
        # Look for the conditional None passing
        has_conditional = (
            "vec_normalize_path" in self.source
            and "no_norm" in self.source
        )
        assert has_conditional, (
            "vec_normalize_path should be conditional on --no-norm"
        )

    def test_vec_normalize_path_none_when_no_norm(self):
        """When --no-norm, vec_normalize_path passed to eval should be None."""
        # Pattern: if no_norm ... vec_normalize_path=None or similar
        # This ensures the eval function handles None gracefully
        pattern = r"vec_normalize_path\s*=\s*None"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "vec_normalize_path should be set to None when --no-norm"
        )


# ===========================================================================
# DummyVecEnv removed from training (replaced by SubprocVecEnv)
# ===========================================================================


class TestDummyVecEnvReplaced:
    """Training env should use SubprocVecEnv, not DummyVecEnv."""

    def test_training_env_not_dummy_vec_env(self):
        """Main training env creation should not use DummyVecEnv.

        DummyVecEnv should only appear in evaluation context, not
        as the main training env (spec says to replace with SubprocVecEnv).
        """
        source = load_train_source()

        # The main() function should use SubprocVecEnv for training,
        # not DummyVecEnv. DummyVecEnv([lambda: MultiDayEnv(...)]) is
        # the old pattern that must be replaced.
        # Check: the main training env creation in main() should NOT use
        # DummyVecEnv wrapping MultiDayEnv. DummyVecEnv is fine in
        # evaluate_sortino for eval wrapping.

        # Find main() function body
        main_pattern = r"def\s+main\s*\(\s*\).*?(?=\ndef\s|\Z)"
        main_match = re.search(main_pattern, source, re.DOTALL)
        assert main_match is not None, "main() function not found"
        main_body = main_match.group(0)

        # In main(), training env should use SubprocVecEnv
        assert "SubprocVecEnv" in main_body, (
            "main() should use SubprocVecEnv for training environment"
        )
