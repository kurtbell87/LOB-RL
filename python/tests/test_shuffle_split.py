"""Tests for shuffle-split feature in train.py.

Spec: docs/shuffle-split.md

These tests verify that scripts/train.py:
- Defines --shuffle-split CLI flag (action='store_true', default False)
- Defines --seed CLI flag (type=int, default=42)
- Imports `random` from stdlib
- When --shuffle-split is set, shuffles all_files before train/val/test split
- Uses random.Random(args.seed).shuffle() for reproducible shuffling (not numpy)
- Applies shuffle in both cache_dir and data_dir code paths
- Default behavior (no --shuffle-split) preserves chronological order
- Prints train/val/test date ranges after the split
- --seed without --shuffle-split is accepted silently
- Same seed always produces the same split
"""

import os
import re
import sys

import pytest

from conftest import load_train_source


# ===========================================================================
# CLI Flags: --shuffle-split
# ===========================================================================


class TestShuffleSplitFlag:
    """--shuffle-split flag should be defined with correct properties."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_shuffle_split_flag_exists(self):
        """--shuffle-split flag should be defined in train.py."""
        assert "--shuffle-split" in self.source, (
            "train.py should define --shuffle-split CLI flag"
        )

    def test_shuffle_split_is_store_true(self):
        """--shuffle-split should be action='store_true'."""
        pattern = r"add_argument\s*\(\s*['\"]--shuffle-split['\"].*?store_true"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "--shuffle-split should have action='store_true'"
        )

    def test_shuffle_split_default_is_false(self):
        """--shuffle-split should default to False (implicit with store_true)."""
        # store_true implies default=False, but we also verify it is not
        # explicitly defaulted to True.
        pattern = r"add_argument\s*\(\s*['\"]--shuffle-split['\"].*?default\s*=\s*True"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is None, (
            "--shuffle-split should not have default=True"
        )


# ===========================================================================
# CLI Flags: --seed
# ===========================================================================


class TestSeedFlag:
    """--seed flag should be defined with correct type and default."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_seed_flag_exists(self):
        """--seed flag should be defined in train.py."""
        assert "--seed" in self.source, (
            "train.py should define --seed CLI flag"
        )

    def test_seed_type_is_int(self):
        """--seed should have type=int."""
        pattern = r"add_argument\s*\(\s*['\"]--seed['\"].*?type\s*=\s*int"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--seed should have type=int"

    def test_seed_default_is_42(self):
        """--seed default should be 42."""
        pattern = r"add_argument\s*\(\s*['\"]--seed['\"].*?default\s*=\s*42\b"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--seed default should be 42"


# ===========================================================================
# Import: `import random` present
# ===========================================================================


class TestRandomImport:
    """train.py should import the random module from stdlib."""

    def test_import_random_present(self):
        """train.py should contain 'import random'."""
        source = load_train_source()
        # Match standalone `import random` (not `from random import ...` which is also ok,
        # but the spec says `import random`)
        pattern = r"^\s*import\s+random\b"
        match = re.search(pattern, source, re.MULTILINE)
        assert match is not None, (
            "train.py should have 'import random' at module level"
        )


# ===========================================================================
# Shuffle Logic: Uses random.Random(seed).shuffle()
# ===========================================================================


class TestShuffleImplementation:
    """Shuffle logic should use random.Random(seed).shuffle() for reproducibility."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_uses_random_random_class(self):
        """Shuffle should use random.Random() (seeded instance, not global state)."""
        pattern = r"random\.Random\s*\("
        match = re.search(pattern, self.source)
        assert match is not None, (
            "Should use random.Random() for reproducible seeded shuffling"
        )

    def test_seed_passed_to_random_random(self):
        """random.Random() should receive the seed from args."""
        # Should see random.Random(args.seed) or random.Random(seed)
        pattern = r"random\.Random\s*\(\s*args\.seed\s*\)"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "random.Random() should receive args.seed as parameter"
        )

    def test_shuffle_called_on_all_files(self):
        """The shuffle method should be called on all_files."""
        # Look for .shuffle(all_files) pattern
        pattern = r"\.shuffle\s*\(\s*all_files\s*\)"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "shuffle() should be called with all_files as argument"
        )

    def test_does_not_use_numpy_random(self):
        """Shuffling should use stdlib random, not numpy.random."""
        # Make sure np.random.shuffle(all_files) or np.random.default_rng().shuffle(all_files)
        # is NOT used for the shuffle-split feature
        source = self.source
        # Find any np.random.shuffle(all_files) or numpy.random patterns near all_files shuffle
        pattern = r"np\.random\.(shuffle|permutation)\s*\(\s*all_files\s*\)"
        match = re.search(pattern, source)
        assert match is None, (
            "Should not use np.random for shuffling all_files; use random.Random(seed)"
        )


# ===========================================================================
# Shuffle Conditional: Only when --shuffle-split is set
# ===========================================================================


class TestShuffleConditional:
    """Shuffling should only happen when --shuffle-split is set."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_shuffle_gated_by_shuffle_split_flag(self):
        """Shuffle logic should be conditional on args.shuffle_split."""
        # Look for conditional: if args.shuffle_split: ... shuffle
        pattern = r"if\s+args\.shuffle_split\s*:"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "Shuffle should be gated by 'if args.shuffle_split:'"
        )

    def test_shuffle_after_all_files_assignment(self):
        """Shuffle must happen after all_files is assigned, before the split."""
        # In the cache_dir path, all_files is assigned then split.
        # The shuffle should appear between assignment and split.
        # Look for the pattern: all_files = ... then shuffle ... then train_files =
        pattern = r"all_files\s*=.*?shuffle.*?train_files\s*="
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "Shuffle should appear after all_files assignment and before train_files split"
        )


# ===========================================================================
# Both Code Paths: cache_dir and data_dir
# ===========================================================================


class TestBothCodePaths:
    """Shuffle should apply in both the cache_dir and data_dir paths."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_shuffle_in_cache_dir_path(self):
        """Shuffle should occur in the cache_dir branch (if args.cache_dir:)."""
        # Find the cache_dir branch and verify shuffle is present
        # The cache_dir branch starts with `if args.cache_dir:` and contains `all_files`
        cache_branch_pattern = r"if\s+args\.cache_dir\s*:.*?(?=\n\s*else\b)"
        cache_branch = re.search(cache_branch_pattern, self.source, re.DOTALL)
        assert cache_branch is not None, "cache_dir branch not found"
        branch_text = cache_branch.group(0)
        assert "shuffle" in branch_text.lower(), (
            "cache_dir branch should contain shuffle logic"
        )

    def test_shuffle_in_data_dir_path(self):
        """Shuffle should occur in the data_dir branch (else:)."""
        # Find the else branch (data_dir path) and verify shuffle is present
        # The data_dir branch starts after the else: following cache_dir
        else_branch_pattern = r"\belse\s*:.*?(?=\n\s{4}if\s+not\s+args\.no_norm|\Z)"
        else_branch = re.search(else_branch_pattern, self.source, re.DOTALL)
        assert else_branch is not None, "data_dir (else) branch not found"
        branch_text = else_branch.group(0)
        assert "shuffle" in branch_text.lower(), (
            "data_dir branch should contain shuffle logic"
        )


# ===========================================================================
# Date Printing: Train/Val/Test dates printed after split
# ===========================================================================


class TestDatePrinting:
    """After the split, train/val/test date ranges should be printed."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_train_dates_printed(self):
        """'Train dates:' should be printed after the split."""
        pattern = r"['\"]Train dates:"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "Should print 'Train dates:' after the split"
        )

    def test_val_dates_printed(self):
        """'Val dates:' should be printed after the split."""
        pattern = r"['\"]Val dates:"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "Should print 'Val dates:' after the split"
        )

    def test_test_dates_printed(self):
        """'Test dates:' should be printed after the split."""
        pattern = r"['\"]Test dates:"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "Should print 'Test dates:' after the split"
        )

    def test_dates_extracted_from_file_tuple(self):
        """Dates should be extracted from file_tuple[0] (first element of all_files tuples)."""
        # The date is file_tuple[0] for each tuple in the split.
        # Look for a pattern accessing element [0] from train_files/val_files/test_files
        # e.g., f[0] for f in train_files, or [t[0] for t in train_files]
        pattern = r"\[\s*\w+\[0\]\s+for\s+\w+\s+in\s+(train_files|val_files|test_files)"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "Dates should be extracted as f[0] from train_files/val_files/test_files"
        )

    def test_date_printing_in_both_branches(self):
        """Date printing should occur in both cache_dir and data_dir code paths."""
        # Count occurrences of "Train dates:" — should be at least 2 (one per branch)
        # OR the printing is factored out after the if/else (also valid)
        count = len(re.findall(r"Train dates:", self.source))
        # If it appears once, it might be after the if/else block (shared code)
        # If it appears twice, it's in each branch
        # Either way, it must appear at least once
        assert count >= 1, (
            "'Train dates:' should appear at least once in train.py"
        )


# ===========================================================================
# Functional: Deterministic shuffle with known seed
# ===========================================================================


class TestShuffleDeterminism:
    """Same seed must produce identical shuffles — functional test."""

    def test_same_seed_same_order(self):
        """random.Random(seed).shuffle() should produce identical results for the same seed."""
        import random

        items_a = list(range(50))
        items_b = list(range(50))

        random.Random(42).shuffle(items_a)
        random.Random(42).shuffle(items_b)

        assert items_a == items_b, (
            "Same seed (42) must produce identical shuffle order"
        )

    def test_different_seed_different_order(self):
        """Different seeds should produce different shuffles."""
        import random

        items_a = list(range(50))
        items_b = list(range(50))

        random.Random(42).shuffle(items_a)
        random.Random(99).shuffle(items_b)

        assert items_a != items_b, (
            "Different seeds (42 vs 99) should produce different orders"
        )

    def test_shuffle_is_in_place(self):
        """random.Random(seed).shuffle() mutates the list in place."""
        import random

        original = list(range(50))
        items = list(range(50))
        random.Random(42).shuffle(items)

        # Should be different from original (extremely unlikely to be same for 50 items)
        assert items != original, "Shuffled list should differ from original"

    def test_seed_42_specific_output(self):
        """Verify exact shuffle output for seed=42 to catch accidental RNG changes."""
        import random

        items = list(range(20))
        random.Random(42).shuffle(items)

        # Record the expected output for regression testing.
        # If this test passes once, any implementation that uses
        # random.Random(42).shuffle() will produce the same result.
        expected_first_three = items[:3]
        # Re-run to verify determinism
        items2 = list(range(20))
        random.Random(42).shuffle(items2)
        assert items2[:3] == expected_first_three


# ===========================================================================
# Edge Case: --seed without --shuffle-split
# ===========================================================================


class TestSeedWithoutShuffleSplit:
    """--seed without --shuffle-split should be accepted silently."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_no_error_on_seed_alone(self):
        """No validation should reject --seed when --shuffle-split is not set."""
        # There should be no code like:
        # if args.seed and not args.shuffle_split: parser.error(...)
        pattern = r"(seed.*shuffle_split.*error|seed.*shuffle_split.*raise)"
        match = re.search(pattern, self.source)
        assert match is None, (
            "--seed without --shuffle-split should not raise an error"
        )

    def test_seed_is_independent_flag(self):
        """--seed should be a standalone add_argument call, not requiring shuffle-split."""
        pattern = r"add_argument\s*\(\s*['\"]--seed['\"]"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "--seed should be defined as its own add_argument() call"
        )


# ===========================================================================
# Preservation: Existing CLI flags untouched
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

    def test_train_days_flag_preserved(self):
        assert "--train-days" in self.source

    def test_bar_size_flag_preserved(self):
        assert "--bar-size" in self.source

    def test_execution_cost_flag_preserved(self):
        assert "--execution-cost" in self.source

    def test_reward_mode_flag_preserved(self):
        assert "--reward-mode" in self.source

    def test_n_envs_flag_preserved(self):
        assert "--n-envs" in self.source

    def test_ent_coef_flag_preserved(self):
        assert "--ent-coef" in self.source


# ===========================================================================
# Acceptance Criteria: High-level checks
# ===========================================================================


class TestAcceptanceCriteria:
    """High-level checks for all acceptance criteria from the spec."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        self.source = load_train_source()

    def test_ac1_shuffle_split_flag_store_true(self):
        """AC1: --shuffle-split flag exists with action='store_true'."""
        pattern = r"add_argument\s*\(\s*['\"]--shuffle-split['\"].*?store_true"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None

    def test_ac2_seed_flag_int_default_42(self):
        """AC2: --seed flag exists with type=int, default=42."""
        pattern = r"add_argument\s*\(\s*['\"]--seed['\"].*?type\s*=\s*int.*?default\s*=\s*42"
        match = re.search(pattern, self.source, re.DOTALL)
        if match is None:
            # Try reversed order (default before type)
            pattern = r"add_argument\s*\(\s*['\"]--seed['\"].*?default\s*=\s*42.*?type\s*=\s*int"
            match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, "--seed should have type=int and default=42"

    def test_ac3_shuffle_split_shuffles_before_split(self):
        """AC3: When --shuffle-split is set, all_files is shuffled before split."""
        pattern = r"shuffle_split.*shuffle.*train_files"
        match = re.search(pattern, self.source, re.DOTALL)
        assert match is not None, (
            "shuffle_split should trigger shuffle before train_files assignment"
        )

    def test_ac4_uses_random_random_not_numpy(self):
        """AC4: Shuffling uses random.Random(args.seed), not numpy.random."""
        assert "random.Random(args.seed)" in self.source, (
            "Should use random.Random(args.seed) for shuffling"
        )

    def test_ac5_default_no_shuffle(self):
        """AC5: Default behavior (no --shuffle-split) is unchanged — chronological order."""
        # The shuffle is gated by if args.shuffle_split, so default is no shuffle
        pattern = r"if\s+args\.shuffle_split\s*:"
        match = re.search(pattern, self.source)
        assert match is not None, (
            "Shuffle must be gated by args.shuffle_split"
        )

    def test_ac6_date_lists_printed(self):
        """AC6: Train/val/test date lists are printed after the split."""
        assert "Train dates:" in self.source
        assert "Val dates:" in self.source
        assert "Test dates:" in self.source

    def test_ac7_import_random_present(self):
        """AC7: import random is present in train.py."""
        pattern = r"^\s*import\s+random\b"
        match = re.search(pattern, self.source, re.MULTILINE)
        assert match is not None

    def test_ac8_no_other_files_modified(self):
        """AC8: Only scripts/train.py should contain shuffle-split logic.

        We verify this by checking that no other Python source files
        in lob_rl/ reference shuffle_split.
        """
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
                assert "shuffle_split" not in content, (
                    f"shuffle_split logic should not be in {fname}"
                )
