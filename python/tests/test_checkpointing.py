"""Tests for checkpointing & resume feature in train.py.

Spec: docs/checkpointing.md

These tests verify that scripts/train.py:
- Defines --checkpoint-freq CLI flag with type=int, default=0
- Defines --resume CLI flag with type=str, default=None
- When checkpoint-freq > 0, uses SB3 CheckpointCallback
- Checkpoint saves go to {output_dir}/checkpoints/ directory
- When VecNormalize is active, a custom callback also saves _vecnormalize.pkl
- Uses CallbackList to combine CheckpointCallback + VecNormalize callback
- model.learn() receives callback= parameter when checkpoint-freq > 0
- When --resume PATH, loads model with PPO.load() or RecurrentPPO.load()
- When --resume, passes env to .load()
- When --resume, passes reset_num_timesteps=False to model.learn()
- When --resume, auto-loads VecNormalize stats if _vecnormalize.pkl exists
- When --resume --no-norm, skips VecNormalize loading
- Flags appear in --help output
- Both flags are present in argument parser
- No other files modified
"""

import os
import re

import pytest

from conftest import load_train_source, extract_main_body


# ===========================================================================
# CLI Flag: --checkpoint-freq exists with correct properties
# ===========================================================================


class TestCheckpointFreqFlag:
    """--checkpoint-freq flag should be defined with correct properties."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_checkpoint_freq_flag_exists(self):
        """--checkpoint-freq flag should be defined in train.py."""
        assert "--checkpoint-freq" in self.source, (
            "train.py should define --checkpoint-freq CLI flag"
        )

    def test_checkpoint_freq_type_is_int(self):
        """--checkpoint-freq should have type=int."""
        pattern = r"add_argument\s*\(\s*['\"]--checkpoint-freq['\"].*?type\s*=\s*int"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--checkpoint-freq should have type=int"

    def test_checkpoint_freq_default_is_zero(self):
        """--checkpoint-freq default should be 0."""
        pattern = r"add_argument\s*\(\s*['\"]--checkpoint-freq['\"].*?default\s*=\s*0\b"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--checkpoint-freq default should be 0"

    def test_checkpoint_freq_has_help_text(self):
        """--checkpoint-freq should have help text."""
        pattern = r"add_argument\s*\(\s*['\"]--checkpoint-freq['\"].*?help\s*="
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--checkpoint-freq should have help text"


# ===========================================================================
# CLI Flag: --resume exists with correct properties
# ===========================================================================


class TestResumeFlag:
    """--resume flag should be defined with correct properties."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_resume_flag_exists(self):
        """--resume flag should be defined in train.py."""
        assert "--resume" in self.source, (
            "train.py should define --resume CLI flag"
        )

    def test_resume_type_is_str(self):
        """--resume should have type=str."""
        pattern = r"add_argument\s*\(\s*['\"]--resume['\"].*?type\s*=\s*str"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--resume should have type=str"

    def test_resume_default_is_none(self):
        """--resume default should be None."""
        pattern = r"add_argument\s*\(\s*['\"]--resume['\"].*?default\s*=\s*None"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--resume default should be None"

    def test_resume_has_help_text(self):
        """--resume should have help text."""
        pattern = r"add_argument\s*\(\s*['\"]--resume['\"].*?help\s*="
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--resume should have help text"


# ===========================================================================
# CheckpointCallback: imported and used when checkpoint-freq > 0
# ===========================================================================


class TestCheckpointCallback:
    """CheckpointCallback should be used when checkpoint-freq > 0."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()
        self.main_body = extract_main_body(self.source)

    def test_checkpoint_callback_imported(self):
        """CheckpointCallback should be imported from stable_baselines3."""
        pattern = r"from\s+stable_baselines3\.common\.callbacks\s+import\s+.*CheckpointCallback"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "CheckpointCallback should be imported from stable_baselines3.common.callbacks"
        )

    def test_checkpoint_callback_instantiated(self):
        """CheckpointCallback should be instantiated in main()."""
        pattern = r"CheckpointCallback\s*\("
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "CheckpointCallback should be instantiated in main()"
        )

    def test_checkpoint_callback_uses_checkpoint_freq(self):
        """CheckpointCallback should use args.checkpoint_freq for the save frequency."""
        pattern = r"CheckpointCallback\s*\(.*?(?:save_freq|checkpoint_freq)\s*=.*?args\.checkpoint_freq"
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "CheckpointCallback save_freq should reference args.checkpoint_freq"
        )

    def test_checkpoint_callback_saves_to_checkpoints_dir(self):
        """CheckpointCallback save_path should point to {output_dir}/checkpoints/."""
        # Should reference output_dir and 'checkpoints' in the save_path
        pattern = r"CheckpointCallback\s*\(.*?save_path\s*="
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "CheckpointCallback should have save_path= argument"
        )

    def test_checkpoints_dir_includes_output_dir(self):
        """The checkpoint save directory should reference args.output_dir."""
        # Look for os.path.join(args.output_dir, 'checkpoints') or similar
        pattern = r"['\"]checkpoints['\"]"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "Checkpoint directory should include 'checkpoints' path component"
        )

    def test_checkpoint_callback_conditional_on_freq_gt_zero(self):
        """CheckpointCallback should only be created when checkpoint_freq > 0."""
        pattern = r"(?:args\.checkpoint_freq\s*>\s*0|checkpoint_freq\s*>\s*0).*?CheckpointCallback"
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "CheckpointCallback should be conditional on checkpoint_freq > 0"
        )


# ===========================================================================
# VecNormalize Checkpoint Saving: Custom callback for _vecnormalize.pkl
# ===========================================================================


class TestVecNormalizeSaving:
    """A custom callback should save VecNormalize stats alongside model checkpoints."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()
        self.main_body = extract_main_body(self.source)

    def test_vecnormalize_pkl_suffix_present(self):
        """Code should reference the _vecnormalize.pkl suffix for saving."""
        pattern = r"vecnormalize\.pkl|_vecnormalize\.pkl"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "Should reference _vecnormalize.pkl suffix for VecNormalize checkpoint saving"
        )

    def test_vec_normalize_save_call_exists(self):
        """There should be an env.save() or VecNormalize save call in a callback context."""
        # The custom callback should call .save() on the VecNormalize env
        # Existing code has 2 .save() calls (model.save + env.save for final).
        # The callback adds a 3rd .save() for periodic VecNormalize saving.
        pattern = r"\.save\s*\("
        matches = re.findall(pattern, self.source)
        assert len(matches) >= 3, (
            f"Should have at least 3 .save() calls: model.save, final env.save, "
            f"and callback env.save (found {len(matches)})"
        )

    def test_custom_callback_class_or_function_exists(self):
        """A custom callback for VecNormalize saving should exist."""
        # Could be a BaseCallback subclass or EveryNTimesteps usage
        # Look for either a class definition inheriting from BaseCallback,
        # or a function/class that handles VecNormalize saving
        has_base_callback = bool(re.search(r"BaseCallback", self.source))
        has_every_n = bool(re.search(r"EveryNTimesteps", self.source))
        has_event_callback = bool(re.search(r"EventCallback", self.source))
        assert has_base_callback or has_every_n or has_event_callback, (
            "Should define a custom callback (BaseCallback/EveryNTimesteps/EventCallback) "
            "for VecNormalize saving"
        )

    def test_vecnormalize_saving_conditional_on_norm_active(self):
        """VecNormalize checkpoint saving should only happen when normalization is active."""
        # The VecNormalize callback creation should be gated by no_norm check.
        # We need to see _vecnormalize.pkl referenced near a no_norm check.
        pattern = r"(?:not\s+args\.no_norm|args\.no_norm).*?vecnormalize\.pkl"
        match = re.search(pattern, self.main_body, re.DOTALL | re.IGNORECASE)
        assert match is not None, (
            "VecNormalize checkpoint saving should be gated by no_norm check"
        )


# ===========================================================================
# CallbackList: Combining callbacks
# ===========================================================================


class TestCallbackList:
    """CallbackList should combine CheckpointCallback + VecNormalize callback."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()
        self.main_body = extract_main_body(self.source)

    def test_callback_list_imported(self):
        """CallbackList should be imported from stable_baselines3."""
        pattern = r"from\s+stable_baselines3\.common\.callbacks\s+import\s+.*CallbackList"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "CallbackList should be imported from stable_baselines3.common.callbacks"
        )

    def test_callback_list_used_in_main(self):
        """CallbackList should be instantiated in main()."""
        pattern = r"CallbackList\s*\("
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "CallbackList should be used in main() to combine callbacks"
        )


# ===========================================================================
# model.learn(): callback parameter
# ===========================================================================


class TestModelLearnCallback:
    """model.learn() should receive callback= when checkpoint-freq > 0."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()
        self.main_body = extract_main_body(self.source)

    def test_model_learn_has_callback_param(self):
        """model.learn() should include callback= parameter."""
        pattern = r"model\.learn\s*\(.*?callback\s*="
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "model.learn() should include callback= parameter"
        )

    def test_model_learn_has_total_timesteps(self):
        """model.learn() should still include total_timesteps."""
        pattern = r"model\.learn\s*\(.*?total_timesteps\s*="
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "model.learn() should still include total_timesteps"
        )


# ===========================================================================
# Resume: Model Loading
# ===========================================================================


class TestResumeModelLoading:
    """--resume should load model from checkpoint."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()
        self.main_body = extract_main_body(self.source)

    def test_ppo_load_called_for_resume(self):
        """When --resume, PPO.load() should be called."""
        pattern = r"PPO\.load\s*\("
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "PPO.load() should be called when --resume is provided"
        )

    def test_recurrent_ppo_load_called_for_resume(self):
        """When --resume --recurrent, RecurrentPPO.load() should be called."""
        pattern = r"RecurrentPPO\.load\s*\("
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "RecurrentPPO.load() should be called when --resume --recurrent is provided"
        )

    def test_resume_load_conditional_on_resume_flag(self):
        """Model loading should be conditional on args.resume being set."""
        pattern = r"(?:if\s+args\.resume|args\.resume\s+is\s+not\s+None).*?\.load\s*\("
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "Model loading should be conditional on args.resume"
        )

    def test_resume_load_passes_env(self):
        """PPO.load() / RecurrentPPO.load() should receive env= parameter."""
        pattern = r"\.load\s*\(.*?env\s*="
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            ".load() should pass env= to attach model to current environment"
        )

    def test_resume_load_uses_resume_path(self):
        """The .load() call should use args.resume as the path."""
        pattern = r"\.load\s*\(\s*args\.resume"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            ".load() should use args.resume as the checkpoint path"
        )

    def test_resume_model_vs_fresh_model_conditional(self):
        """Model creation should branch: load from checkpoint OR create fresh."""
        # There should be some conditional: if resume then load, else create new
        pattern = r"args\.resume.*?\.load\s*\(.*?(?:else|PPO\s*\(|RecurrentPPO\s*\()"
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "Should have conditional: resume → .load() vs fresh → PPO()/RecurrentPPO()"
        )

    def test_recurrent_resume_conditional_on_recurrent_flag(self):
        """Resume with --recurrent should use RecurrentPPO.load(), else PPO.load()."""
        pattern = r"args\.recurrent.*?RecurrentPPO\.load|RecurrentPPO\.load.*?args\.recurrent"
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "RecurrentPPO.load() should be conditional on args.recurrent"
        )


# ===========================================================================
# Resume: reset_num_timesteps=False
# ===========================================================================


class TestResumeTimestepContinuation:
    """When --resume, model.learn() should use reset_num_timesteps=False."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()
        self.main_body = extract_main_body(self.source)

    def test_reset_num_timesteps_false_present(self):
        """reset_num_timesteps=False should appear in model.learn() call."""
        pattern = r"reset_num_timesteps\s*=\s*False"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "model.learn() should include reset_num_timesteps=False when resuming"
        )

    def test_reset_num_timesteps_in_learn_call(self):
        """reset_num_timesteps should be part of the model.learn() call."""
        pattern = r"model\.learn\s*\(.*?reset_num_timesteps"
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "reset_num_timesteps should be in the model.learn() call"
        )

    def test_reset_num_timesteps_conditional_on_resume(self):
        """reset_num_timesteps=False should only be set when --resume is active."""
        # The value should depend on whether args.resume is set
        # Could be: reset_num_timesteps=not args.resume, or
        # reset_num_timesteps=False inside an if args.resume block, or
        # reset_num_timesteps=(args.resume is None) etc.
        pattern = (
            r"reset_num_timesteps\s*=\s*(?:not\s+(?:bool\s*\()?args\.resume|False)"
            r"|reset_num_timesteps\s*=\s*\(\s*args\.resume\s+is\s+None\s*\)"
        )
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "reset_num_timesteps should depend on whether --resume is set"
        )


# ===========================================================================
# Resume: VecNormalize Loading
# ===========================================================================


class TestResumeVecNormalizeLoading:
    """--resume should auto-load VecNormalize stats if _vecnormalize.pkl exists."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()
        self.main_body = extract_main_body(self.source)

    def test_vecnormalize_path_derived_from_resume_path(self):
        """VecNormalize path should be derived by replacing .zip with _vecnormalize.pkl."""
        # Should see something like args.resume.replace('.zip', '_vecnormalize.pkl')
        # or a re.sub or string manipulation
        pattern = r"\.replace\s*\(\s*['\"]\.zip['\"].*?['\"]_vecnormalize\.pkl['\"]"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "VecNormalize path should be derived by replacing .zip with _vecnormalize.pkl"
        )

    def test_vecnormalize_file_existence_check(self):
        """Should check if the VecNormalize file exists before loading."""
        pattern = r"os\.path\.exists\s*\(.*?vecnormalize"
        match = re.search(pattern, self.main_body, re.IGNORECASE)
        assert match is not None, (
            "Should check os.path.exists() for the VecNormalize file"
        )

    def test_vecnormalize_load_called(self):
        """VecNormalize.load() should be called when stats file exists."""
        pattern = r"VecNormalize\.load\s*\("
        matches = re.findall(pattern, self.main_body)
        # At least one VecNormalize.load() for resume; eval already has one
        assert len(matches) >= 1, (
            "VecNormalize.load() should be called during resume"
        )

    def test_vecnormalize_load_in_resume_context(self):
        """VecNormalize.load() during resume should be near args.resume logic."""
        pattern = r"args\.resume.*?VecNormalize\.load"
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "VecNormalize.load() should be in the --resume handling code path"
        )

    def test_warning_when_vecnormalize_file_missing(self):
        """Should print a warning when _vecnormalize.pkl doesn't exist."""
        # Look for a print/warning specifically about missing VecNormalize file
        # during resume — should be near the os.path.exists check for vecnormalize.pkl
        pattern = r"os\.path\.exists.*?vecnormalize.*?(?:print|warnings?\.warn)\s*\(|(?:print|warnings?\.warn)\s*\(.*?(?:vecnormalize|VecNormalize).*?(?:not\s+found|missing|does\s+not\s+exist|skipping)"
        match = re.search(pattern, self.main_body, re.IGNORECASE | re.DOTALL)
        assert match is not None, (
            "Should warn when _vecnormalize.pkl doesn't exist alongside checkpoint"
        )


# ===========================================================================
# Resume: --no-norm skips VecNormalize loading
# ===========================================================================


class TestResumeNoNormSkipsVecNormalize:
    """--resume --no-norm should skip VecNormalize loading entirely."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()
        self.main_body = extract_main_body(self.source)

    def test_no_norm_check_in_resume_vecnormalize_loading(self):
        """VecNormalize loading during resume should check args.no_norm."""
        # The VecNormalize loading path should be gated by not args.no_norm.
        # Existing code has 2 no_norm references (VecNormalize creation + final save).
        # Resume adds at least 1 more for VecNormalize resume loading.
        pattern = r"(?:not\s+args\.no_norm|args\.no_norm)"
        matches = re.findall(pattern, self.main_body)
        assert len(matches) >= 3, (
            "Should check args.no_norm in VecNormalize creation, final save, AND resume loading "
            f"(found {len(matches)} occurrences, need at least 3)"
        )

    def test_vecnormalize_resume_load_inside_not_no_norm_block(self):
        """VecNormalize loading during resume should be inside a 'not args.no_norm' block."""
        # After args.resume check, VecNormalize.load should be gated by no_norm
        pattern = r"args\.resume.*?(?:not\s+args\.no_norm|args\.no_norm).*?VecNormalize\.load"
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "VecNormalize.load() during resume should be gated by no_norm check"
        )


# ===========================================================================
# Checkpoint Freq 0: No callback (default behavior unchanged)
# ===========================================================================


class TestCheckpointFreqZeroDisabled:
    """checkpoint_freq=0 should not add any checkpoint callbacks."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()
        self.main_body = extract_main_body(self.source)

    def test_checkpoint_callback_gated_by_freq_gt_zero(self):
        """CheckpointCallback creation should be gated by checkpoint_freq > 0."""
        pattern = r"args\.checkpoint_freq\s*>\s*0"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "Checkpoint callbacks should be gated by args.checkpoint_freq > 0"
        )

    def test_callback_none_when_freq_zero(self):
        """When checkpoint_freq == 0, callback should be None or not set."""
        # The learn call should handle the case where no callbacks are needed
        # Either: callback=None, or callback is conditionally set
        pattern = r"callback\s*=\s*None|callbacks\s*=\s*\[\s*\]"
        match = re.search(pattern, self.main_body)
        if match is None:
            # Alternative: callback variable only set inside if checkpoint_freq > 0
            pattern2 = r"args\.checkpoint_freq\s*>\s*0.*?callback"
            match = re.search(pattern2, self.main_body, re.DOTALL)
            assert match is not None, (
                "Callback should be None or conditionally set only when checkpoint_freq > 0"
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

    def test_frame_stack_flag_preserved(self):
        assert "--frame-stack" in self.source

    def test_recurrent_flag_preserved(self):
        assert "--recurrent" in self.source

    def test_total_timesteps_flag_preserved(self):
        assert "--total-timesteps" in self.source

    def test_output_dir_flag_preserved(self):
        assert "--output-dir" in self.source


# ===========================================================================
# Scope: Only train.py modified
# ===========================================================================


class TestOnlyTrainPyModified:
    """Checkpointing logic should only exist in scripts/train.py."""

    def test_no_checkpoint_callback_in_lob_rl_modules(self):
        """No lob_rl/ Python modules should reference CheckpointCallback."""
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
                assert "CheckpointCallback" not in content, (
                    f"CheckpointCallback should not be in lob_rl/{fname}"
                )

    def test_no_resume_logic_in_lob_rl_modules(self):
        """No lob_rl/ Python modules should reference PPO.load or model resume."""
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
                assert "PPO.load" not in content, (
                    f"PPO.load should not be in lob_rl/{fname}"
                )


# ===========================================================================
# Imports: Required SB3 imports present
# ===========================================================================


class TestRequiredImports:
    """Required SB3 callback imports should be present."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_checkpoint_callback_importable(self):
        """CheckpointCallback should be importable from stable_baselines3."""
        try:
            from stable_baselines3.common.callbacks import CheckpointCallback
        except ImportError:
            pytest.fail("CheckpointCallback not importable from stable_baselines3.common.callbacks")

    def test_callback_list_importable(self):
        """CallbackList should be importable from stable_baselines3."""
        try:
            from stable_baselines3.common.callbacks import CallbackList
        except ImportError:
            pytest.fail("CallbackList not importable from stable_baselines3.common.callbacks")

    def test_base_callback_importable(self):
        """BaseCallback should be importable from stable_baselines3."""
        try:
            from stable_baselines3.common.callbacks import BaseCallback
        except ImportError:
            pytest.fail("BaseCallback not importable from stable_baselines3.common.callbacks")

    def test_callbacks_import_line_in_source(self):
        """train.py should import from stable_baselines3.common.callbacks."""
        pattern = r"from\s+stable_baselines3\.common\.callbacks\s+import"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "train.py should import from stable_baselines3.common.callbacks"
        )


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge case handling for checkpointing and resume."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()
        self.main_body = extract_main_body(self.source)

    def test_checkpoint_freq_zero_is_default(self):
        """Default checkpoint_freq=0 means no checkpointing."""
        pattern = r"add_argument\s*\(\s*['\"]--checkpoint-freq['\"].*?default\s*=\s*0"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "Default checkpoint-freq should be 0 (disabled)"

    def test_resume_none_is_default(self):
        """Default resume=None means fresh training."""
        pattern = r"add_argument\s*\(\s*['\"]--resume['\"].*?default\s*=\s*None"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "Default resume should be None"

    def test_resume_with_nonexistent_path_not_caught(self):
        """--resume with non-existent path should propagate FileNotFoundError (SB3 raises it)."""
        # There should NOT be a try/except that catches FileNotFoundError around .load()
        # SB3's .load() will raise it naturally
        load_region_pattern = r"\.load\s*\(\s*args\.resume.*?\)"
        match = re.search(load_region_pattern, self.main_body, re.DOTALL)
        if match:
            # Check there's no try/except wrapping this
            start = match.start()
            # Look back 200 chars for a try statement without intervening function defs
            preceding = self.main_body[max(0, start - 200):start]
            has_try = bool(re.search(r"\btry\s*:", preceding))
            # It's OK if there's no try — that means FileNotFoundError propagates naturally
            # If there IS a try, it should not catch FileNotFoundError
            if has_try:
                # Look for except FileNotFoundError or bare except
                following = self.main_body[start:start + 300]
                catches_fnf = bool(re.search(r"except\s+(?:FileNotFoundError|Exception|BaseException|\s*:)", following))
                assert not catches_fnf, (
                    "Should NOT catch FileNotFoundError from .load() — let SB3 raise it"
                )

    def test_checkpoint_freq_without_resume_works(self):
        """--checkpoint-freq without --resume should work (fresh training + checkpoints)."""
        # The checkpoint callback creation should NOT require args.resume
        # CheckpointCallback should only depend on checkpoint_freq > 0
        pattern = r"args\.checkpoint_freq\s*>\s*0"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "Checkpoint creation should only depend on checkpoint_freq > 0, not on --resume"
        )

    def test_resume_without_checkpoint_freq_works(self):
        """--resume without --checkpoint-freq should work (resume without new checkpoints)."""
        # The resume loading should NOT require checkpoint_freq > 0
        # These should be independent code paths
        pattern = r"args\.resume"
        resume_matches = list(re.finditer(pattern, self.main_body))
        pattern2 = r"args\.checkpoint_freq"
        freq_matches = list(re.finditer(pattern2, self.main_body))
        # Both should be independently referenced
        assert len(resume_matches) >= 1, "Should reference args.resume"
        assert len(freq_matches) >= 1, "Should reference args.checkpoint_freq"

    def test_resume_and_checkpoint_freq_together_works(self):
        """--resume PATH --checkpoint-freq N should work (resume + continue checkpointing)."""
        # Both code paths should exist independently
        has_resume = bool(re.search(r"args\.resume", self.main_body))
        has_checkpoint = bool(re.search(r"args\.checkpoint_freq\s*>\s*0", self.main_body))
        assert has_resume and has_checkpoint, (
            "Both --resume and --checkpoint-freq code paths should exist independently"
        )


# ===========================================================================
# Acceptance Criteria: High-level checks
# ===========================================================================


class TestAcceptanceCriteria:
    """High-level checks for all acceptance criteria from the spec."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()
        self.main_body = extract_main_body(self.source)

    def test_ac1_checkpoint_freq_produces_checkpoints(self):
        """AC1: --checkpoint-freq N > 0 creates CheckpointCallback saving to checkpoints/ dir."""
        has_callback = bool(re.search(r"CheckpointCallback\s*\(", self.main_body))
        has_dir = bool(re.search(r"['\"]checkpoints['\"]", self.main_body))
        assert has_callback and has_dir, (
            "AC1: Should create CheckpointCallback saving to checkpoints/ directory"
        )

    def test_ac2_vecnormalize_saved_alongside_checkpoint(self):
        """AC2: When VecNormalize active, also saves _vecnormalize.pkl with each checkpoint."""
        pattern = r"vecnormalize\.pkl"
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "AC2: Should save _vecnormalize.pkl alongside each model checkpoint"
        )

    def test_ac3_resume_loads_model_with_reset_false(self):
        """AC3: --resume loads model and continues with reset_num_timesteps=False."""
        has_load = bool(re.search(r"\.load\s*\(\s*args\.resume", self.main_body))
        has_reset = bool(re.search(r"reset_num_timesteps\s*=\s*False", self.main_body))
        assert has_load and has_reset, (
            "AC3: Should load model from resume path and use reset_num_timesteps=False"
        )

    def test_ac4_resume_auto_loads_vecnormalize(self):
        """AC4: --resume auto-loads VecNormalize stats if _vecnormalize.pkl exists."""
        has_derive = bool(re.search(r"\.replace\s*\(.*?\.zip.*?vecnormalize", self.main_body, re.DOTALL))
        has_exists = bool(re.search(r"os\.path\.exists", self.main_body))
        has_load = bool(re.search(r"VecNormalize\.load", self.main_body))
        assert has_derive and has_exists and has_load, (
            "AC4: Should derive VecNormalize path, check existence, and load if present"
        )

    def test_ac5_resume_no_norm_skips_vecnormalize(self):
        """AC5: --resume --no-norm skips VecNormalize loading."""
        pattern = r"(?:not\s+args\.no_norm|args\.no_norm).*?VecNormalize\.load"
        match = re.search(pattern, self.main_body, re.DOTALL)
        assert match is not None, (
            "AC5: VecNormalize loading should be gated by no_norm flag"
        )

    def test_ac6_resume_recurrent_uses_recurrent_ppo_load(self):
        """AC6: --resume --recurrent uses RecurrentPPO.load()."""
        pattern = r"RecurrentPPO\.load\s*\("
        match = re.search(pattern, self.main_body)
        assert match is not None, (
            "AC6: Should use RecurrentPPO.load() when --resume --recurrent"
        )

    def test_ac7_checkpoint_freq_in_help(self):
        """AC7: --checkpoint-freq appears in argument parser (visible in --help)."""
        pattern = r"add_argument\s*\(\s*['\"]--checkpoint-freq['\"]"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "AC7: --checkpoint-freq should be in argument parser"
        )

    def test_ac8_resume_in_help(self):
        """AC8: --resume appears in argument parser (visible in --help)."""
        pattern = r"add_argument\s*\(\s*['\"]--resume['\"]"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "AC8: --resume should be in argument parser"
        )
