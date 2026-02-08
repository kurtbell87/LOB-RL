"""Tests for frame stacking feature in train.py.

Spec: docs/frame-stacking.md

These tests verify that scripts/train.py:
- Defines --frame-stack CLI flag with type=int, default=1
- Imports VecFrameStack from stable_baselines3.common.vec_env
- When frame_stack > 1, training env chain is SubprocVecEnv → VecFrameStack → VecNormalize
- When frame_stack == 1, no VecFrameStack wrapper (default behavior unchanged)
- evaluate_sortino() accepts frame_stack parameter (default 1)
- Eval wraps DummyVecEnv with VecFrameStack before VecNormalize.load() when frame_stack > 1
- Both val and test eval calls forward frame_stack
- No other files modified
"""

import os
import re

import pytest

from conftest import load_train_source


# ===========================================================================
# CLI Flag: --frame-stack exists with correct type and default
# ===========================================================================


class TestFrameStackFlag:
    """--frame-stack flag should be defined with correct properties."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_frame_stack_flag_exists(self):
        """--frame-stack flag should be defined in train.py."""
        assert "--frame-stack" in self.source, (
            "train.py should define --frame-stack CLI flag"
        )

    def test_frame_stack_type_is_int(self):
        """--frame-stack should have type=int."""
        pattern = r"add_argument\s*\(\s*['\"]--frame-stack['\"].*?type\s*=\s*int"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--frame-stack should have type=int"

    def test_frame_stack_default_is_1(self):
        """--frame-stack default should be 1."""
        pattern = r"add_argument\s*\(\s*['\"]--frame-stack['\"].*?default\s*=\s*1\b"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--frame-stack default should be 1"


# ===========================================================================
# Import: VecFrameStack imported from stable_baselines3.common.vec_env
# ===========================================================================


class TestVecFrameStackImport:
    """VecFrameStack should be imported from stable_baselines3.common.vec_env."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_vec_frame_stack_imported(self):
        """train.py should import VecFrameStack."""
        assert "VecFrameStack" in self.source, (
            "train.py should import VecFrameStack"
        )

    def test_vec_frame_stack_from_sb3_vec_env(self):
        """VecFrameStack should be imported from stable_baselines3.common.vec_env."""
        pattern = r"from\s+stable_baselines3\.common\.vec_env\s+import\s+.*VecFrameStack"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "VecFrameStack should be imported from stable_baselines3.common.vec_env"
        )

    def test_vec_frame_stack_on_same_import_line_as_others(self):
        """VecFrameStack should be on the same import line as DummyVecEnv, SubprocVecEnv, VecNormalize."""
        # All four should appear in a single from ... import line
        pattern = r"from\s+stable_baselines3\.common\.vec_env\s+import\s+[^\n]*DummyVecEnv[^\n]*SubprocVecEnv[^\n]*VecFrameStack[^\n]*VecNormalize"
        match = re.search(pattern, self.source)
        if match is None:
            # Try alternative orderings — just check all four are on a line that starts with the import
            import_line_pattern = r"from\s+stable_baselines3\.common\.vec_env\s+import\s+([^\n]+)"
            import_match = re.search(import_line_pattern, self.source)
            assert import_match is not None, "vec_env import line not found"
            import_text = import_match.group(1)
            for name in ["DummyVecEnv", "SubprocVecEnv", "VecFrameStack", "VecNormalize"]:
                assert name in import_text, (
                    f"{name} should be on the same import line from stable_baselines3.common.vec_env"
                )


# ===========================================================================
# Training Env Chain: VecFrameStack inserted between SubprocVecEnv and VecNormalize
# ===========================================================================


class TestTrainingEnvChain:
    """When frame_stack > 1, training env chain should be SubprocVecEnv → VecFrameStack → VecNormalize."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_vec_frame_stack_conditional_on_frame_stack_gt_1(self):
        """VecFrameStack wrapping should be conditional on frame_stack > 1."""
        pattern = r"if\s+args\.frame_stack\s*>\s*1\s*:"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "VecFrameStack wrapping should be gated by 'if args.frame_stack > 1:'"
        )

    def test_vec_frame_stack_wraps_env_in_main(self):
        """In main(), VecFrameStack should be called with env and n_stack."""
        # Find main() function body
        main_pattern = r"def\s+main\s*\(\s*\).*?(?=\ndef\s|\Z)"
        main_match = re.search(main_pattern, self.source, re.DOTALL)
        assert main_match is not None, "main() function not found"
        main_body = main_match.group(0)

        pattern = r"VecFrameStack\s*\(\s*env\s*,\s*n_stack\s*=\s*args\.frame_stack\s*\)"
        match = re.search(pattern, main_body)
        assert match is not None, (
            "main() should call VecFrameStack(env, n_stack=args.frame_stack)"
        )

    def test_vec_frame_stack_before_vec_normalize(self):
        """VecFrameStack should appear BEFORE VecNormalize in main()."""
        # Find main() function body
        main_pattern = r"def\s+main\s*\(\s*\).*?(?=\ndef\s|\Z)"
        main_match = re.search(main_pattern, self.source, re.DOTALL)
        assert main_match is not None, "main() function not found"
        main_body = main_match.group(0)

        # VecFrameStack should appear before VecNormalize( wrapping
        frame_stack_pos = main_body.find("VecFrameStack")
        vec_normalize_pos = main_body.find("VecNormalize(")
        assert frame_stack_pos != -1, "VecFrameStack not found in main()"
        assert vec_normalize_pos != -1, "VecNormalize( not found in main()"
        assert frame_stack_pos < vec_normalize_pos, (
            "VecFrameStack should appear BEFORE VecNormalize in main()"
        )

    def test_vec_frame_stack_after_subproc_vec_env(self):
        """VecFrameStack should appear AFTER SubprocVecEnv creation in main()."""
        main_pattern = r"def\s+main\s*\(\s*\).*?(?=\ndef\s|\Z)"
        main_match = re.search(main_pattern, self.source, re.DOTALL)
        assert main_match is not None, "main() function not found"
        main_body = main_match.group(0)

        # Find last SubprocVecEnv( call (there are two — cache_dir and data_dir branches)
        subproc_positions = [m.start() for m in re.finditer(r"SubprocVecEnv\s*\(", main_body)]
        assert len(subproc_positions) > 0, "SubprocVecEnv not found in main()"
        last_subproc_pos = max(subproc_positions)

        frame_stack_pos = main_body.find("VecFrameStack")
        assert frame_stack_pos != -1, "VecFrameStack not found in main()"
        assert frame_stack_pos > last_subproc_pos, (
            "VecFrameStack should appear AFTER SubprocVecEnv in main()"
        )

    def test_vec_frame_stack_assigns_to_env(self):
        """VecFrameStack result should be assigned back to env."""
        main_pattern = r"def\s+main\s*\(\s*\).*?(?=\ndef\s|\Z)"
        main_match = re.search(main_pattern, self.source, re.DOTALL)
        assert main_match is not None, "main() function not found"
        main_body = main_match.group(0)

        pattern = r"env\s*=\s*VecFrameStack\s*\("
        match = re.search(pattern, main_body)
        assert match is not None, (
            "VecFrameStack result should be assigned to env"
        )


# ===========================================================================
# Default Behavior: frame_stack == 1 means no VecFrameStack
# ===========================================================================


class TestDefaultNoFrameStack:
    """When frame_stack == 1, no VecFrameStack wrapper should be applied."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_frame_stack_1_skips_wrapping(self):
        """The condition 'if args.frame_stack > 1' ensures frame_stack=1 skips wrapping."""
        # If the condition is > 1, then default=1 naturally skips.
        pattern = r"if\s+args\.frame_stack\s*>\s*1\s*:"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "Condition should be 'args.frame_stack > 1' so default=1 skips VecFrameStack"
        )

    def test_no_unconditional_vec_frame_stack(self):
        """VecFrameStack should NOT be applied unconditionally in main()."""
        main_pattern = r"def\s+main\s*\(\s*\).*?(?=\ndef\s|\Z)"
        main_match = re.search(main_pattern, self.source, re.DOTALL)
        assert main_match is not None, "main() function not found"
        main_body = main_match.group(0)

        # Find all VecFrameStack calls in main
        frame_stack_calls = list(re.finditer(r"VecFrameStack\s*\(", main_body))
        # Each call should be inside a conditional block (indented under if)
        for call in frame_stack_calls:
            # Get the line containing this call
            line_start = main_body.rfind("\n", 0, call.start()) + 1
            line = main_body[line_start:call.start()]
            # Should be indented more than the function body (at least 8 spaces for if block)
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            assert indent >= 8, (
                f"VecFrameStack call should be inside a conditional block (found indent={indent})"
            )


# ===========================================================================
# Evaluate: evaluate_sortino() accepts frame_stack parameter
# ===========================================================================


class TestEvalFrameStackParam:
    """evaluate_sortino() should accept a frame_stack parameter."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_evaluate_sortino_has_frame_stack_param(self):
        """evaluate_sortino() signature should include frame_stack."""
        pattern = r"def\s+evaluate_sortino\s*\(.*?frame_stack"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "evaluate_sortino() should accept frame_stack parameter"
        )

    def test_evaluate_sortino_frame_stack_default_is_1(self):
        """evaluate_sortino() frame_stack parameter should default to 1."""
        pattern = r"def\s+evaluate_sortino\s*\(.*?frame_stack\s*=\s*1\b"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "evaluate_sortino() frame_stack should default to 1"
        )


# ===========================================================================
# Evaluate: VecFrameStack in eval wrapping chain
# ===========================================================================


class TestEvalVecFrameStack:
    """Eval should wrap DummyVecEnv with VecFrameStack before VecNormalize.load()."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()
        # Extract evaluate_sortino function body
        pattern = r"def\s+evaluate_sortino\s*\(.*?\n(?=def\s|\Z)"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "evaluate_sortino() not found"
        self.eval_body = match.group(0)

    def test_eval_has_vec_frame_stack_conditional(self):
        """evaluate_sortino should check frame_stack > 1 before wrapping."""
        pattern = r"if\s+frame_stack\s*>\s*1\s*:"
        match = re.search(pattern, self.eval_body)
        assert match is not None, (
            "evaluate_sortino should have 'if frame_stack > 1:' conditional"
        )

    def test_eval_vec_frame_stack_wraps_venv(self):
        """evaluate_sortino should call VecFrameStack on venv."""
        pattern = r"VecFrameStack\s*\(\s*venv\s*,\s*n_stack\s*=\s*frame_stack\s*\)"
        match = re.search(pattern, self.eval_body)
        assert match is not None, (
            "evaluate_sortino should call VecFrameStack(venv, n_stack=frame_stack)"
        )

    def test_eval_vec_frame_stack_assigns_to_venv(self):
        """VecFrameStack result should be assigned back to venv in eval."""
        pattern = r"venv\s*=\s*VecFrameStack\s*\("
        match = re.search(pattern, self.eval_body)
        assert match is not None, (
            "VecFrameStack result should be assigned to venv in evaluate_sortino"
        )

    def test_eval_vec_frame_stack_before_vec_normalize_load(self):
        """VecFrameStack should appear before VecNormalize.load() in eval."""
        frame_pos = self.eval_body.find("VecFrameStack")
        norm_load_pos = self.eval_body.find("VecNormalize.load")
        assert frame_pos != -1, "VecFrameStack not found in evaluate_sortino"
        assert norm_load_pos != -1, "VecNormalize.load not found in evaluate_sortino"
        assert frame_pos < norm_load_pos, (
            "VecFrameStack should appear BEFORE VecNormalize.load in evaluate_sortino"
        )

    def test_eval_vec_frame_stack_after_dummy_vec_env(self):
        """VecFrameStack should appear after DummyVecEnv creation in eval."""
        dummy_pos = self.eval_body.find("DummyVecEnv")
        frame_pos = self.eval_body.find("VecFrameStack")
        assert dummy_pos != -1, "DummyVecEnv not found in evaluate_sortino"
        assert frame_pos != -1, "VecFrameStack not found in evaluate_sortino"
        assert frame_pos > dummy_pos, (
            "VecFrameStack should appear AFTER DummyVecEnv in evaluate_sortino"
        )


# ===========================================================================
# Eval Calls: Both val and test forward frame_stack
# ===========================================================================


class TestEvalCallsForwardFrameStack:
    """Both val and test evaluate_sortino() calls should forward frame_stack."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()
        # Extract main() function body
        main_pattern = r"def\s+main\s*\(\s*\).*?(?=\ndef\s|\Z)"
        main_match = re.search(main_pattern, self.source, re.DOTALL)
        assert main_match is not None, "main() function not found"
        self.main_body = main_match.group(0)

    def test_val_eval_call_forwards_frame_stack(self):
        """Validation evaluate_sortino() call should include frame_stack=."""
        # Find evaluate_sortino calls that include frame_stack
        eval_calls = list(re.finditer(r"evaluate_sortino\s*\(", self.main_body))
        assert len(eval_calls) >= 2, (
            "main() should have at least 2 evaluate_sortino calls (val and test)"
        )
        # Check first call (validation) includes frame_stack
        first_call_start = eval_calls[0].start()
        # Find the closing paren (approximate — look for the next line that ends the call)
        call_region = self.main_body[first_call_start:first_call_start + 500]
        assert "frame_stack" in call_region, (
            "Validation evaluate_sortino() call should include frame_stack"
        )

    def test_test_eval_call_forwards_frame_stack(self):
        """Test evaluate_sortino() call should include frame_stack=."""
        eval_calls = list(re.finditer(r"evaluate_sortino\s*\(", self.main_body))
        assert len(eval_calls) >= 2, (
            "main() should have at least 2 evaluate_sortino calls (val and test)"
        )
        # Check second call (test) includes frame_stack
        second_call_start = eval_calls[1].start()
        call_region = self.main_body[second_call_start:second_call_start + 500]
        assert "frame_stack" in call_region, (
            "Test evaluate_sortino() call should include frame_stack"
        )

    def test_frame_stack_comes_from_args(self):
        """frame_stack argument should reference args.frame_stack."""
        pattern = r"frame_stack\s*=\s*args\.frame_stack"
        matches = re.findall(pattern, self.main_body)
        assert len(matches) >= 2, (
            "Both eval calls should use frame_stack=args.frame_stack (found %d)" % len(matches)
        )


# ===========================================================================
# Scope: No other files modified
# ===========================================================================


class TestNoOtherFilesModified:
    """frame_stack logic should only exist in scripts/train.py."""

    def test_no_frame_stack_in_lob_rl_modules(self):
        """No lob_rl/ Python modules should reference frame_stack."""
        lob_rl_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "lob_rl")
        )
        if not os.path.isdir(lob_rl_dir):
            pytest.skip("lob_rl directory not found")

        for fname in os.listdir(lob_rl_dir):
            if fname.endswith(".py"):
                fpath = os.path.join(lob_rl_dir, fname)
                with open(fpath) as f:
                    content = f.read()
                assert "frame_stack" not in content, (
                    f"frame_stack logic should not be in lob_rl/{fname}"
                )

    def test_no_vec_frame_stack_in_lob_rl_modules(self):
        """No lob_rl/ Python modules should import VecFrameStack."""
        lob_rl_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "lob_rl")
        )
        if not os.path.isdir(lob_rl_dir):
            pytest.skip("lob_rl directory not found")

        for fname in os.listdir(lob_rl_dir):
            if fname.endswith(".py"):
                fpath = os.path.join(lob_rl_dir, fname)
                with open(fpath) as f:
                    content = f.read()
                assert "VecFrameStack" not in content, (
                    f"VecFrameStack should not be in lob_rl/{fname}"
                )


# ===========================================================================
# Existing Flags: Preserved unchanged
# ===========================================================================


class TestExistingFlagsPreserved:
    """Existing CLI flags should still be present and unchanged."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_cache_dir_flag_preserved(self):
        assert "--cache-dir" in self.source

    def test_data_dir_flag_preserved(self):
        assert "--data-dir" in self.source

    def test_bar_size_flag_preserved(self):
        assert "--bar-size" in self.source

    def test_n_envs_flag_preserved(self):
        assert "--n-envs" in self.source

    def test_no_norm_flag_preserved(self):
        assert "--no-norm" in self.source

    def test_execution_cost_flag_preserved(self):
        assert "--execution-cost" in self.source

    def test_shuffle_split_flag_preserved(self):
        assert "--shuffle-split" in self.source


# ===========================================================================
# Acceptance Criteria: High-level checks
# ===========================================================================


class TestAcceptanceCriteria:
    """High-level checks for all acceptance criteria from the spec."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_ac1_frame_stack_flag_int_default_1(self):
        """AC1: --frame-stack flag exists with type=int, default=1."""
        pattern = r"add_argument\s*\(\s*['\"]--frame-stack['\"].*?type\s*=\s*int"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "AC1: --frame-stack should have type=int"
        pattern = r"add_argument\s*\(\s*['\"]--frame-stack['\"].*?default\s*=\s*1\b"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "AC1: --frame-stack should have default=1"

    def test_ac2_vec_frame_stack_imported(self):
        """AC2: VecFrameStack imported from stable_baselines3.common.vec_env."""
        pattern = r"from\s+stable_baselines3\.common\.vec_env\s+import\s+.*VecFrameStack"
        match = re.search(pattern, self.source)
        assert match is not None

    def test_ac3_training_chain_order(self):
        """AC3: Training env chain is SubprocVecEnv → VecFrameStack → VecNormalize."""
        main_pattern = r"def\s+main\s*\(\s*\).*?(?=\ndef\s|\Z)"
        main_match = re.search(main_pattern, self.source, re.DOTALL)
        assert main_match is not None
        main_body = main_match.group(0)

        # All three should exist in order
        subproc_positions = [m.start() for m in re.finditer(r"SubprocVecEnv\s*\(", main_body)]
        frame_pos = main_body.find("VecFrameStack")
        normalize_pos = main_body.find("VecNormalize(")
        assert len(subproc_positions) > 0
        assert frame_pos > max(subproc_positions), "VecFrameStack must come after SubprocVecEnv"
        assert normalize_pos > frame_pos, "VecNormalize must come after VecFrameStack"

    def test_ac4_default_no_wrapping(self):
        """AC4: When frame_stack == 1, no VecFrameStack wrapper."""
        pattern = r"if\s+args\.frame_stack\s*>\s*1\s*:"
        match = re.search(pattern, self.source)
        assert match is not None

    def test_ac5_evaluate_sortino_has_frame_stack(self):
        """AC5: evaluate_sortino() accepts frame_stack parameter."""
        pattern = r"def\s+evaluate_sortino\s*\(.*?frame_stack\s*=\s*1"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None

    def test_ac6_eval_wraps_with_frame_stack(self):
        """AC6: Eval wraps DummyVecEnv with VecFrameStack before VecNormalize.load()."""
        eval_pattern = r"def\s+evaluate_sortino\s*\(.*?\n(?=def\s|\Z)"
        eval_match = re.search(eval_pattern, self.source, re.DOTALL)
        assert eval_match is not None
        eval_body = eval_match.group(0)
        assert "VecFrameStack" in eval_body

    def test_ac7_both_eval_calls_forward_frame_stack(self):
        """AC7: Both val and test eval calls forward frame_stack."""
        main_pattern = r"def\s+main\s*\(\s*\).*?(?=\ndef\s|\Z)"
        main_match = re.search(main_pattern, self.source, re.DOTALL)
        assert main_match is not None
        main_body = main_match.group(0)
        frame_stack_in_eval = re.findall(r"evaluate_sortino\s*\([^)]*frame_stack", main_body, re.DOTALL)
        assert len(frame_stack_in_eval) >= 2, (
            f"Both val and test eval calls should forward frame_stack (found {len(frame_stack_in_eval)})"
        )

    def test_ac8_only_train_py_modified(self):
        """AC8: No other files should contain VecFrameStack."""
        lob_rl_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "lob_rl")
        )
        if not os.path.isdir(lob_rl_dir):
            pytest.skip("lob_rl directory not found")

        for fname in os.listdir(lob_rl_dir):
            if fname.endswith(".py"):
                fpath = os.path.join(lob_rl_dir, fname)
                with open(fpath) as f:
                    content = f.read()
                assert "VecFrameStack" not in content, (
                    f"VecFrameStack should not appear in lob_rl/{fname}"
                )
