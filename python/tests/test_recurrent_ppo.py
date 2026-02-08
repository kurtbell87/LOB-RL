"""Tests for RecurrentPPO support in train.py.

Spec: docs/recurrent-ppo.md

These tests verify that scripts/train.py:
- Defines --recurrent CLI flag with action='store_true', default False
- --recurrent + --frame-stack > 1 raises parser error (mutually exclusive)
- When --recurrent, RecurrentPPO is imported from sb3_contrib (conditional import)
- sb3_contrib is NOT imported at the top level
- When --recurrent, model is RecurrentPPO('MlpLstmPolicy', ...)
- When not --recurrent, model is PPO('MlpPolicy', ...) (unchanged default)
- evaluate_sortino() accepts is_recurrent parameter (default False)
- When is_recurrent=True, eval uses LSTM state tracking (lstm_states, episode_start)
- Both val and test eval calls forward is_recurrent=args.recurrent
- All other PPO hyperparameters stay the same for RecurrentPPO
- No other files are modified
"""

import os
import re

import pytest

# Path to the train.py script
TRAIN_SCRIPT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "train.py")
)


def _load_train_source():
    """Read train.py source as a string."""
    with open(TRAIN_SCRIPT) as f:
        return f.read()


def _extract_main_body(source):
    """Extract the main() function body from source."""
    pattern = r"def\s+main\s*\(\s*\).*?(?=\ndef\s|\Z)"
    match = re.search(pattern, source, re.DOTALL)
    assert match is not None, "main() function not found"
    return match.group(0)


def _extract_evaluate_sortino_body(source):
    """Extract the evaluate_sortino() function body from source."""
    pattern = r"def\s+evaluate_sortino\s*\(.*?\n(?=def\s|\Z)"
    match = re.search(pattern, source, re.DOTALL)
    assert match is not None, "evaluate_sortino() function not found"
    return match.group(0)


# ===========================================================================
# CLI Flag: --recurrent exists with correct properties
# ===========================================================================


class TestRecurrentFlag:
    """--recurrent flag should be defined with correct properties."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _load_train_source()

    def test_recurrent_flag_exists(self):
        """--recurrent flag should be defined in train.py."""
        assert "--recurrent" in self.source, (
            "train.py should define --recurrent CLI flag"
        )

    def test_recurrent_is_store_true(self):
        """--recurrent should have action='store_true'."""
        pattern = r"add_argument\s*\(\s*['\"]--recurrent['\"].*?store_true"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--recurrent should have action='store_true'"

    def test_recurrent_default_is_false(self):
        """--recurrent should default to False (store_true implies this)."""
        # store_true defaults to False, but verify no explicit default=True
        pattern = r"add_argument\s*\(\s*['\"]--recurrent['\"].*?default\s*=\s*True"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is None, "--recurrent should not have default=True"


# ===========================================================================
# Mutual Exclusivity: --recurrent + --frame-stack > 1
# ===========================================================================


class TestMutualExclusivity:
    """--recurrent and --frame-stack > 1 should be mutually exclusive."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _load_train_source()
        self.main_body = _extract_main_body(self.source)

    def test_mutual_exclusivity_check_exists(self):
        """There should be a check for --recurrent + --frame-stack > 1."""
        # Look for the validation pattern
        pattern = r"args\.recurrent.*args\.frame_stack\s*>\s*1|args\.frame_stack\s*>\s*1.*args\.recurrent"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "Should check for --recurrent + --frame-stack > 1 mutual exclusivity"
        )

    def test_mutual_exclusivity_raises_parser_error(self):
        """The mutual exclusivity check should raise a parser error."""
        # Look for parser.error() in the context of the recurrent/frame-stack check
        pattern = r"(args\.recurrent.*args\.frame_stack|args\.frame_stack.*args\.recurrent).*?parser\.error"
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "Mutual exclusivity check should call parser.error()"
        )

    def test_mutual_exclusivity_error_message_mentions_both(self):
        """The parser error message should mention both flags."""
        # Find the parser.error call related to recurrent/frame-stack
        pattern = r"parser\.error\s*\(\s*['\"].*recurrent.*frame.stack|parser\.error\s*\(\s*['\"].*frame.stack.*recurrent"
        match = re.search(pattern, self.main_body, re.DOTALL | re.IGNORECASE)
        assert match is not None, (
            "parser.error message should mention both --recurrent and --frame-stack"
        )

    def test_recurrent_with_frame_stack_1_allowed(self):
        """The check should be > 1, so --recurrent with --frame-stack 1 is fine."""
        # Verify the condition uses > 1 (not >= 1 or != 1)
        pattern = r"frame_stack\s*>\s*1"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "Mutual exclusivity condition should use frame_stack > 1 (not >= 1)"
        )


# ===========================================================================
# Conditional Import: RecurrentPPO from sb3_contrib
# ===========================================================================


class TestConditionalImport:
    """RecurrentPPO should be conditionally imported from sb3_contrib."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _load_train_source()
        self.main_body = _extract_main_body(self.source)

    def test_sb3_contrib_not_imported_at_top_level(self):
        """sb3_contrib should NOT be imported at the top level of train.py."""
        # Get lines before the first function definition — these are top-level imports
        first_def = re.search(r"^def\s+", self.source, re.MULTILINE)
        assert first_def is not None, "No function definitions found"
        top_level = self.source[:first_def.start()]
        assert "sb3_contrib" not in top_level, (
            "sb3_contrib should NOT be imported at the top level"
        )

    def test_recurrent_ppo_imported_from_sb3_contrib(self):
        """RecurrentPPO should be imported from sb3_contrib."""
        pattern = r"from\s+sb3_contrib\s+import\s+RecurrentPPO"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "RecurrentPPO should be imported from sb3_contrib"
        )

    def test_import_is_conditional_on_recurrent_flag(self):
        """The sb3_contrib import should be inside a conditional block (not top-level)."""
        # Find the import and check it's inside main() or gated by args.recurrent
        pattern = r"if\s+args\.recurrent.*?from\s+sb3_contrib\s+import\s+RecurrentPPO"
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "sb3_contrib import should be conditional on args.recurrent"
        )

    def test_import_inside_main_not_top_level(self):
        """The RecurrentPPO import should be inside main(), not at module level."""
        # The import should appear inside the main() function body
        pattern = r"from\s+sb3_contrib\s+import\s+RecurrentPPO"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "RecurrentPPO import should be inside main() function"
        )


# ===========================================================================
# Model Creation: RecurrentPPO vs PPO
# ===========================================================================


class TestModelCreation:
    """Model creation should switch between PPO and RecurrentPPO based on --recurrent."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _load_train_source()
        self.main_body = _extract_main_body(self.source)

    def test_recurrent_ppo_uses_mlp_lstm_policy(self):
        """When --recurrent, model should use 'MlpLstmPolicy'."""
        pattern = r"RecurrentPPO\s*\(\s*['\"]MlpLstmPolicy['\"]"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "RecurrentPPO should be created with 'MlpLstmPolicy'"
        )

    def test_ppo_still_uses_mlp_policy(self):
        """When not --recurrent, model should still use 'MlpPolicy'."""
        pattern = r"PPO\s*\(\s*['\"]MlpPolicy['\"]"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "PPO should still use 'MlpPolicy' when not recurrent"
        )

    def test_model_creation_is_conditional(self):
        """Model creation should be conditional on args.recurrent."""
        # There should be an if args.recurrent block that contains RecurrentPPO
        # AND an else block that contains PPO — both must be present
        pattern = r"if\s+args\.recurrent.*?RecurrentPPO.*?else.*?PPO\s*\("
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "Model creation should have if args.recurrent: RecurrentPPO ... else: PPO"
        )

    def test_recurrent_ppo_receives_same_hyperparams(self):
        """RecurrentPPO should receive the same hyperparameters as PPO."""
        # Find the RecurrentPPO constructor call and check key hyperparameters
        # Extract the RecurrentPPO(...) call
        pattern = r"RecurrentPPO\s*\(.*?\)"
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, "RecurrentPPO constructor call not found"
        rppo_call = match.group(0)

        required_params = [
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "clip_range",
            "policy_kwargs",
            "tensorboard_log",
        ]
        for param in required_params:
            assert param in rppo_call, (
                f"RecurrentPPO should receive {param} (same as PPO)"
            )

    def test_recurrent_ppo_learning_rate_from_args(self):
        """RecurrentPPO learning_rate should reference args (not hardcoded)."""
        pattern = r"RecurrentPPO\s*\(.*?learning_rate\s*=\s*args\."
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "RecurrentPPO learning_rate should reference args"
        )

    def test_recurrent_ppo_ent_coef_from_args(self):
        """RecurrentPPO ent_coef should reference args (not hardcoded)."""
        pattern = r"RecurrentPPO\s*\(.*?ent_coef\s*=\s*args\."
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "RecurrentPPO ent_coef should reference args"
        )

    def test_recurrent_ppo_batch_size_from_args(self):
        """RecurrentPPO batch_size should reference args (not hardcoded)."""
        pattern = r"RecurrentPPO\s*\(.*?batch_size\s*=\s*args\."
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "RecurrentPPO batch_size should reference args"
        )


# ===========================================================================
# Evaluation: evaluate_sortino() accepts is_recurrent parameter
# ===========================================================================


class TestEvalIsRecurrentParam:
    """evaluate_sortino() should accept an is_recurrent parameter."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _load_train_source()
        self.eval_body = _extract_evaluate_sortino_body(self.source)

    def test_evaluate_sortino_has_is_recurrent_param(self):
        """evaluate_sortino() signature should include is_recurrent."""
        pattern = r"def\s+evaluate_sortino\s*\(.*?is_recurrent"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "evaluate_sortino() should accept is_recurrent parameter"
        )

    def test_evaluate_sortino_is_recurrent_default_false(self):
        """evaluate_sortino() is_recurrent parameter should default to False."""
        pattern = r"def\s+evaluate_sortino\s*\(.*?is_recurrent\s*=\s*False"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "evaluate_sortino() is_recurrent should default to False"
        )


# ===========================================================================
# Evaluation: LSTM state tracking when is_recurrent=True
# ===========================================================================


class TestEvalLSTMStateTracking:
    """When is_recurrent=True, eval should use LSTM state tracking."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _load_train_source()
        self.eval_body = _extract_evaluate_sortino_body(self.source)

    def test_lstm_states_variable_exists(self):
        """evaluate_sortino should initialize lstm_states variable."""
        assert "lstm_states" in self.eval_body, (
            "evaluate_sortino should have lstm_states variable"
        )

    def test_episode_start_variable_exists(self):
        """evaluate_sortino should initialize episode_start variable."""
        assert "episode_start" in self.eval_body, (
            "evaluate_sortino should have episode_start variable"
        )

    def test_lstm_states_initialized_to_none(self):
        """lstm_states should be initialized to None."""
        pattern = r"lstm_states\s*=\s*None"
        match = re.search(pattern, self.eval_body)
        assert match is not None, (
            "lstm_states should be initialized to None"
        )

    def test_episode_start_initialized_as_ones(self):
        """episode_start should be initialized as np.ones((1,), dtype=bool)."""
        pattern = r"episode_start\s*=\s*np\.ones\s*\(\s*\(\s*1\s*,?\s*\)\s*,\s*dtype\s*=\s*bool\s*\)"
        match = re.search(pattern, self.eval_body)
        assert match is not None, (
            "episode_start should be initialized as np.ones((1,), dtype=bool)"
        )

    def test_predict_uses_state_kwarg(self):
        """When recurrent, model.predict should use state= kwarg."""
        pattern = r"model\.predict\s*\(.*?state\s*=\s*lstm_states"
        match = re.search(pattern, self.eval_body, re.DOTALL)
        assert match is not None, (
            "Recurrent predict should pass state=lstm_states"
        )

    def test_predict_uses_episode_start_kwarg(self):
        """When recurrent, model.predict should use episode_start= kwarg."""
        pattern = r"model\.predict\s*\(.*?episode_start\s*=\s*episode_start"
        match = re.search(pattern, self.eval_body, re.DOTALL)
        assert match is not None, (
            "Recurrent predict should pass episode_start=episode_start"
        )

    def test_predict_returns_lstm_states(self):
        """Recurrent predict should unpack action and lstm_states."""
        pattern = r"action\s*,\s*lstm_states\s*=\s*model\.predict"
        match = re.search(pattern, self.eval_body)
        assert match is not None, (
            "Recurrent predict should unpack as action, lstm_states = model.predict(...)"
        )

    def test_episode_start_updated_from_dones(self):
        """episode_start should be updated from dones after each step."""
        pattern = r"episode_start\s*=\s*dones"
        match = re.search(pattern, self.eval_body)
        assert match is not None, (
            "episode_start should be updated: episode_start = dones"
        )

    def test_lstm_tracking_conditional_on_is_recurrent(self):
        """LSTM state tracking should be conditional on is_recurrent."""
        # The predict with state= should be inside an if is_recurrent block
        pattern = r"if\s+is_recurrent.*?state\s*=\s*lstm_states"
        match = re.search(pattern, self.eval_body, re.DOTALL)
        assert match is not None, (
            "LSTM state tracking should be gated by is_recurrent"
        )

    def test_non_recurrent_predict_unchanged(self):
        """When not recurrent, predict should still use the simple two-value unpack."""
        # There should be an else branch or the existing simple predict pattern
        # action, _ = model.predict(obs, deterministic=True)
        pattern = r"action\s*,\s*_\s*=\s*model\.predict\s*\(\s*obs\s*,\s*deterministic\s*=\s*True\s*\)"
        match = re.search(pattern, self.eval_body)
        assert match is not None, (
            "Non-recurrent predict should still use action, _ = model.predict(obs, deterministic=True)"
        )


# ===========================================================================
# Eval Calls: Both val and test forward is_recurrent
# ===========================================================================


class TestEvalCallsForwardIsRecurrent:
    """Both val and test evaluate_sortino() calls should forward is_recurrent."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _load_train_source()
        self.main_body = _extract_main_body(self.source)

    def test_val_eval_call_forwards_is_recurrent(self):
        """Validation evaluate_sortino() call should include is_recurrent=."""
        eval_calls = list(re.finditer(r"evaluate_sortino\s*\(", self.main_body))
        assert len(eval_calls) >= 2, (
            "main() should have at least 2 evaluate_sortino calls (val and test)"
        )
        first_call_start = eval_calls[0].start()
        call_region = self.main_body[first_call_start:first_call_start + 600]
        assert "is_recurrent" in call_region, (
            "Validation evaluate_sortino() call should include is_recurrent"
        )

    def test_test_eval_call_forwards_is_recurrent(self):
        """Test evaluate_sortino() call should include is_recurrent=."""
        eval_calls = list(re.finditer(r"evaluate_sortino\s*\(", self.main_body))
        assert len(eval_calls) >= 2, (
            "main() should have at least 2 evaluate_sortino calls (val and test)"
        )
        second_call_start = eval_calls[1].start()
        call_region = self.main_body[second_call_start:second_call_start + 600]
        assert "is_recurrent" in call_region, (
            "Test evaluate_sortino() call should include is_recurrent"
        )

    def test_is_recurrent_comes_from_args(self):
        """is_recurrent argument should reference args.recurrent."""
        pattern = r"is_recurrent\s*=\s*args\.recurrent"
        matches = re.findall(pattern, self.main_body)
        assert len(matches) >= 2, (
            "Both eval calls should use is_recurrent=args.recurrent (found %d)" % len(matches)
        )


# ===========================================================================
# Dependency: sb3_contrib importable
# ===========================================================================


class TestSb3ContribDependency:
    """sb3_contrib should be installed and importable."""

    def test_sb3_contrib_importable(self):
        """sb3_contrib package should be importable."""
        try:
            import sb3_contrib
        except ImportError:
            pytest.fail("sb3_contrib is not installed — run: uv pip install sb3-contrib")

    def test_recurrent_ppo_importable(self):
        """RecurrentPPO should be importable from sb3_contrib."""
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError:
            pytest.fail("RecurrentPPO not importable from sb3_contrib")


# ===========================================================================
# Scope: No other files modified
# ===========================================================================


class TestNoOtherFilesModified:
    """Recurrent PPO logic should only exist in scripts/train.py."""

    def test_no_recurrent_ppo_in_lob_rl_modules(self):
        """No lob_rl/ Python modules should reference RecurrentPPO."""
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
                assert "RecurrentPPO" not in content, (
                    f"RecurrentPPO should not be in lob_rl/{fname}"
                )

    def test_no_sb3_contrib_in_lob_rl_modules(self):
        """No lob_rl/ Python modules should import sb3_contrib."""
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
                assert "sb3_contrib" not in content, (
                    f"sb3_contrib should not be in lob_rl/{fname}"
                )

    def test_no_lstm_states_in_lob_rl_modules(self):
        """No lob_rl/ Python modules should reference lstm_states."""
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
                assert "lstm_states" not in content, (
                    f"lstm_states should not be in lob_rl/{fname}"
                )


# ===========================================================================
# Existing Flags: Preserved unchanged
# ===========================================================================


class TestExistingFlagsPreserved:
    """Existing CLI flags should still be present and unchanged."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _load_train_source()

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

    def test_frame_stack_flag_preserved(self):
        assert "--frame-stack" in self.source


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge case handling for --recurrent flag."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _load_train_source()
        self.main_body = _extract_main_body(self.source)

    def test_recurrent_without_frame_stack_works(self):
        """--recurrent without --frame-stack should work (frame_stack defaults to 1).

        The mutual exclusivity check should use frame_stack > 1, so default=1 is fine.
        """
        pattern = r"frame_stack\s*>\s*1"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "Mutual exclusivity should check frame_stack > 1 (allowing default=1)"
        )

    def test_recurrent_with_frame_stack_1_works(self):
        """--recurrent with --frame-stack 1 should work (1 means no stacking)."""
        # Condition is > 1, not >= 1, so frame_stack=1 passes
        pattern = r"frame_stack\s*>=\s*1"
        match = re.search(pattern, self.main_body)
        assert match is None, (
            "Condition should be > 1, NOT >= 1 (frame_stack=1 should be allowed)"
        )

    def test_default_behavior_unchanged_without_recurrent(self):
        """Without --recurrent, PPO('MlpPolicy', ...) should still be the default."""
        pattern = r"PPO\s*\(\s*['\"]MlpPolicy['\"]"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "Default model creation should still be PPO('MlpPolicy', ...)"
        )


# ===========================================================================
# Acceptance Criteria: High-level checks
# ===========================================================================


class TestAcceptanceCriteria:
    """High-level checks for all acceptance criteria from the spec."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = _load_train_source()
        self.main_body = _extract_main_body(self.source)
        self.eval_body = _extract_evaluate_sortino_body(self.source)

    def test_ac1_recurrent_flag_store_true(self):
        """AC1: --recurrent flag exists with action='store_true'."""
        pattern = r"add_argument\s*\(\s*['\"]--recurrent['\"].*?store_true"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "AC1: --recurrent should have action='store_true'"

    def test_ac2_mutual_exclusivity(self):
        """AC2: --recurrent + --frame-stack > 1 raises parser error."""
        pattern = r"args\.recurrent.*frame_stack.*parser\.error"
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, "AC2: Should raise parser error for --recurrent + --frame-stack > 1"

    def test_ac3_recurrent_ppo_import(self):
        """AC3: When --recurrent, RecurrentPPO is imported from sb3_contrib."""
        pattern = r"from\s+sb3_contrib\s+import\s+RecurrentPPO"
        match = re.search(pattern, self.source)
        assert match is not None, "AC3: Should import RecurrentPPO from sb3_contrib"

    def test_ac4_recurrent_uses_mlp_lstm_policy(self):
        """AC4: When --recurrent, model is RecurrentPPO('MlpLstmPolicy', ...)."""
        pattern = r"RecurrentPPO\s*\(\s*['\"]MlpLstmPolicy['\"]"
        match = re.search(pattern, self.main_body)
        assert match is not None, "AC4: Should use RecurrentPPO('MlpLstmPolicy', ...)"

    def test_ac5_default_is_ppo_mlp_policy(self):
        """AC5: When not --recurrent, model is PPO('MlpPolicy', ...) unchanged."""
        pattern = r"PPO\s*\(\s*['\"]MlpPolicy['\"]"
        match = re.search(pattern, self.main_body)
        assert match is not None, "AC5: Default should still be PPO('MlpPolicy', ...)"

    def test_ac6_evaluate_sortino_has_is_recurrent(self):
        """AC6: evaluate_sortino() accepts is_recurrent parameter (default False)."""
        pattern = r"def\s+evaluate_sortino\s*\(.*?is_recurrent\s*=\s*False"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "AC6: evaluate_sortino() should have is_recurrent=False"

    def test_ac7_eval_uses_lstm_tracking(self):
        """AC7: When is_recurrent=True, eval uses LSTM state tracking."""
        assert "lstm_states" in self.eval_body, "AC7: Eval should use lstm_states"
        assert "episode_start" in self.eval_body, "AC7: Eval should use episode_start"
        pattern = r"state\s*=\s*lstm_states"
        match = re.search(pattern, self.eval_body)
        assert match is not None, "AC7: Eval should pass state=lstm_states to predict"

    def test_ac8_both_eval_calls_forward_is_recurrent(self):
        """AC8: Both val and test eval calls forward is_recurrent=args.recurrent."""
        pattern = r"is_recurrent\s*=\s*args\.recurrent"
        matches = re.findall(pattern, self.main_body)
        assert len(matches) >= 2, (
            f"AC8: Both eval calls should forward is_recurrent=args.recurrent (found {len(matches)})"
        )

    def test_ac9_sb3_contrib_not_top_level(self):
        """AC9: sb3_contrib is NOT imported at the top level."""
        first_def = re.search(r"^def\s+", self.source, re.MULTILINE)
        assert first_def is not None
        top_level = self.source[:first_def.start()]
        assert "sb3_contrib" not in top_level, "AC9: sb3_contrib should not be top-level import"

    def test_ac10_no_other_files_modified(self):
        """AC10: No other files are modified (RecurrentPPO only in train.py)."""
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
                assert "RecurrentPPO" not in content, (
                    f"AC10: RecurrentPPO should not appear in lob_rl/{fname}"
                )
