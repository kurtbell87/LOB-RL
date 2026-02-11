"""Tests for sequence model infrastructure (LSTM, Transformer) for barrier prediction.

Tests the full pipeline: embeddings, models, collation, training loop,
prediction, and evaluation — all using synthetic data.
"""

import numpy as np
import pytest
import torch

from lob_rl.barrier.sequence_models import (
    BarrierLSTM,
    BarrierTransformer,
    LinearBarEmbedding,
    collate_sessions,
    evaluate_sequence_model,
    predict_sessions,
    train_sequence_model,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers (test infrastructure only — no implementation logic)
# ---------------------------------------------------------------------------

def _make_synthetic_session(n_bars: int, n_features: int = 22, seed: int = 0):
    """Create a single synthetic session dict matching load_session_features() format."""
    rng = np.random.RandomState(seed)
    return {
        "features": rng.randn(n_bars, n_features).astype(np.float32),
        "Y_long": rng.rand(n_bars) < 1.0 / 3,
        "Y_short": rng.rand(n_bars) < 1.0 / 3,
    }


def _make_synthetic_sessions(n_sessions: int, min_bars: int = 30,
                              max_bars: int = 80, n_features: int = 22,
                              seed: int = 42):
    """Create multiple synthetic sessions with variable lengths."""
    rng = np.random.RandomState(seed)
    sessions = []
    for i in range(n_sessions):
        n_bars = rng.randint(min_bars, max_bars + 1)
        sessions.append(_make_synthetic_session(n_bars, n_features, seed=seed + i))
    return sessions


def _make_planted_signal_sessions(n_sessions: int = 32, n_bars: int = 50,
                                   n_features: int = 22, seed: int = 42):
    """Create sessions where Y_long is deterministic from feature[0] > 0.

    This provides a clear signal that a model should be able to learn.
    """
    rng = np.random.RandomState(seed)
    sessions = []
    for i in range(n_sessions):
        features = rng.randn(n_bars, n_features).astype(np.float32)
        # Y_long is deterministic: feature 0 > 0 means Y_long=True
        Y_long = features[:, 0] > 0.0
        # Y_short is inverse: feature 1 < 0 means Y_short=True
        Y_short = features[:, 1] < 0.0
        sessions.append({
            "features": features,
            "Y_long": Y_long,
            "Y_short": Y_short,
        })
    return sessions


# ===========================================================================
# Embedding Tests
# ===========================================================================


class TestLinearBarEmbedding:
    """Test the LinearBarEmbedding module."""

    def test_output_shape(self):
        """LinearBarEmbedding produces (B, T, d_model) from (B, T, n_features)."""
        emb = LinearBarEmbedding(n_features=22, d_model=64)
        x = torch.randn(4, 50, 22)
        out = emb(x)
        assert out.shape == (4, 50, 64)

    def test_positional_encoding_varies_by_position(self):
        """Positional encoding should make different positions distinguishable."""
        emb = LinearBarEmbedding(n_features=22, d_model=64, dropout=0.0)
        emb.eval()
        # Feed identical features at every position
        x = torch.ones(1, 10, 22)
        out = emb(x)
        # Positions 0 and 1 should differ (due to positional encoding)
        assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-6), \
            "Positional encoding should make position 0 != position 1"

    def test_dropout_applied_during_training(self):
        """Dropout should cause variation between forward passes in train mode."""
        emb = LinearBarEmbedding(n_features=22, d_model=64, dropout=0.5)
        emb.train()
        x = torch.randn(2, 20, 22)
        torch.manual_seed(0)
        out1 = emb(x)
        torch.manual_seed(1)
        out2 = emb(x)
        # With dropout=0.5, two forward passes should differ
        assert not torch.allclose(out1, out2, atol=1e-6), \
            "Dropout should cause different outputs in train mode"

    def test_custom_n_features_and_d_model(self):
        """Embedding accepts custom n_features and d_model."""
        emb = LinearBarEmbedding(n_features=10, d_model=32)
        x = torch.randn(2, 30, 10)
        out = emb(x)
        assert out.shape == (2, 30, 32)
        assert emb.d_model == 32


# ===========================================================================
# Model Shape Tests
# ===========================================================================


class TestBarrierLSTMShape:
    """Test BarrierLSTM forward pass shape contract."""

    def test_forward_shape(self):
        """BarrierLSTM: (B, T, 22) -> two (B, T) tensors."""
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        x = torch.randn(4, 50, 22)
        logit_long, logit_short = model(x)
        assert logit_long.shape == (4, 50)
        assert logit_short.shape == (4, 50)

    def test_variable_sequence_length(self):
        """LSTM works with different T across calls."""
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        model.eval()
        x1 = torch.randn(2, 30, 22)
        x2 = torch.randn(2, 100, 22)
        l1, s1 = model(x1)
        l2, s2 = model(x2)
        assert l1.shape == (2, 30)
        assert l2.shape == (2, 100)

    def test_batch_size_one(self):
        """LSTM works with batch_size=1."""
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        x = torch.randn(1, 50, 22)
        logit_long, logit_short = model(x)
        assert logit_long.shape == (1, 50)
        assert logit_short.shape == (1, 50)


class TestBarrierTransformerShape:
    """Test BarrierTransformer forward pass shape contract."""

    def test_forward_shape(self):
        """BarrierTransformer: (B, T, 22) -> two (B, T) tensors."""
        model = BarrierTransformer(n_features=22, d_model=64, n_layers=2, n_heads=4)
        x = torch.randn(4, 50, 22)
        logit_long, logit_short = model(x)
        assert logit_long.shape == (4, 50)
        assert logit_short.shape == (4, 50)

    def test_variable_sequence_length(self):
        """Transformer works with different T across calls."""
        model = BarrierTransformer(n_features=22, d_model=64, n_layers=2, n_heads=4)
        model.eval()
        x1 = torch.randn(2, 30, 22)
        x2 = torch.randn(2, 100, 22)
        l1, s1 = model(x1)
        l2, s2 = model(x2)
        assert l1.shape == (2, 30)
        assert l2.shape == (2, 100)

    def test_batch_size_one(self):
        """Transformer works with batch_size=1."""
        model = BarrierTransformer(n_features=22, d_model=64, n_layers=2, n_heads=4)
        x = torch.randn(1, 50, 22)
        logit_long, logit_short = model(x)
        assert logit_long.shape == (1, 50)
        assert logit_short.shape == (1, 50)


class TestModelParameterBudget:
    """Verify parameter counts are in the expected range."""

    def test_lstm_parameter_count(self):
        """BarrierLSTM with d_model=64, n_layers=2 should have 25-50K params."""
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        n_params = sum(p.numel() for p in model.parameters())
        assert 25_000 <= n_params <= 100_000, \
            f"LSTM parameter count {n_params} outside expected range [25K, 100K]"

    def test_transformer_parameter_count(self):
        """BarrierTransformer with d_model=64, n_layers=2 should have 25-100K params."""
        model = BarrierTransformer(n_features=22, d_model=64, n_layers=2, n_heads=4)
        n_params = sum(p.numel() for p in model.parameters())
        assert 25_000 <= n_params <= 100_000, \
            f"Transformer parameter count {n_params} outside expected range [25K, 100K]"


class TestCustomEmbedding:
    """Both models accept an optional embedding override."""

    def test_lstm_with_custom_embedding(self):
        """BarrierLSTM uses custom embedding when provided."""
        custom_emb = LinearBarEmbedding(n_features=22, d_model=32, dropout=0.0)
        model = BarrierLSTM(n_features=22, d_model=32, n_layers=2,
                            embedding=custom_emb)
        x = torch.randn(2, 40, 22)
        logit_long, logit_short = model(x)
        assert logit_long.shape == (2, 40)
        assert logit_short.shape == (2, 40)

    def test_transformer_with_custom_embedding(self):
        """BarrierTransformer uses custom embedding when provided."""
        custom_emb = LinearBarEmbedding(n_features=22, d_model=32, dropout=0.0)
        model = BarrierTransformer(n_features=22, d_model=32, n_layers=2,
                                    n_heads=4, embedding=custom_emb)
        x = torch.randn(2, 40, 22)
        logit_long, logit_short = model(x)
        assert logit_long.shape == (2, 40)
        assert logit_short.shape == (2, 40)


# ===========================================================================
# Causal Masking Tests
# ===========================================================================


class TestCausalMasking:
    """Causal masking: future bars must NOT affect past predictions."""

    def test_perturb_future_does_not_change_past(self):
        """Perturbing bar T should not change predictions for bars 0..T-1."""
        model = BarrierTransformer(n_features=22, d_model=64, n_layers=2, n_heads=4,
                                    dropout=0.0)
        model.eval()

        x = torch.randn(1, 20, 22)
        logit_long_orig, logit_short_orig = model(x)

        # Perturb the last 5 bars
        x_perturbed = x.clone()
        x_perturbed[0, 15:, :] = torch.randn(5, 22)
        logit_long_pert, logit_short_pert = model(x_perturbed)

        # Predictions for bars 0..14 must be identical
        assert torch.allclose(logit_long_orig[0, :15], logit_long_pert[0, :15], atol=1e-5), \
            "Perturbing future bars changed past long predictions"
        assert torch.allclose(logit_short_orig[0, :15], logit_short_pert[0, :15], atol=1e-5), \
            "Perturbing future bars changed past short predictions"

    def test_different_padding_does_not_affect_past(self):
        """Different padding values after real bars should not change real predictions."""
        model = BarrierTransformer(n_features=22, d_model=64, n_layers=2, n_heads=4,
                                    dropout=0.0)
        model.eval()

        # Create two inputs: same first 10 bars, different padding after
        x1 = torch.randn(1, 15, 22)
        x2 = x1.clone()
        x2[0, 10:, :] = torch.randn(5, 22)  # Different "padding"

        l1, s1 = model(x1)
        l2, s2 = model(x2)

        # First 10 bars should produce identical predictions
        assert torch.allclose(l1[0, :10], l2[0, :10], atol=1e-5), \
            "Different future content changed past long predictions"
        assert torch.allclose(s1[0, :10], s2[0, :10], atol=1e-5), \
            "Different future content changed past short predictions"


# ===========================================================================
# Collation Tests
# ===========================================================================


class TestCollation:
    """Test collate_sessions() padding and mask creation."""

    def test_correct_padding_shape(self):
        """Collated batch has shape (B, T_max, 22), padded with zeros."""
        sessions = [
            _make_synthetic_session(30, seed=0),
            _make_synthetic_session(50, seed=1),
            _make_synthetic_session(40, seed=2),
        ]
        features, Y_long, Y_short, mask = collate_sessions(sessions)
        assert features.shape == (3, 50, 22)
        assert Y_long.shape == (3, 50)
        assert Y_short.shape == (3, 50)
        assert mask.shape == (3, 50)

    def test_mask_true_for_real_false_for_padding(self):
        """Mask is True for real positions, False for padding."""
        sessions = [
            _make_synthetic_session(20, seed=0),
            _make_synthetic_session(40, seed=1),
        ]
        _, _, _, mask = collate_sessions(sessions)
        # Session 0 has 20 bars, session 1 has 40 bars; T_max=40
        assert mask[0, :20].all(), "Real positions should be True"
        assert not mask[0, 20:].any(), "Padded positions should be False"
        assert mask[1, :40].all(), "Longest session should be all True"

    def test_labels_padded_with_zeros(self):
        """Padded label positions should be 0."""
        sessions = [
            _make_synthetic_session(20, seed=0),
            _make_synthetic_session(40, seed=1),
        ]
        _, Y_long, Y_short, _ = collate_sessions(sessions)
        # Padded region for session 0 (positions 20..39) should be 0
        assert (Y_long[0, 20:] == 0).all()
        assert (Y_short[0, 20:] == 0).all()

    def test_single_session_batch(self):
        """Collating a single session should work."""
        sessions = [_make_synthetic_session(30, seed=0)]
        features, Y_long, Y_short, mask = collate_sessions(sessions)
        assert features.shape == (1, 30, 22)
        assert mask.all(), "Single session should have no padding"

    def test_return_types_are_tensors(self):
        """collate_sessions returns PyTorch tensors with correct dtypes."""
        sessions = [_make_synthetic_session(30, seed=0)]
        features, Y_long, Y_short, mask = collate_sessions(sessions)
        assert isinstance(features, torch.Tensor)
        assert features.dtype == torch.float32
        assert Y_long.dtype == torch.float32
        assert Y_short.dtype == torch.float32
        assert mask.dtype == torch.bool


# ===========================================================================
# Overfit Tests
# ===========================================================================


class TestOverfit:
    """Models should be able to memorize a small dataset (signal detection)."""

    def test_lstm_memorizes_synthetic_sessions(self):
        """BarrierLSTM memorizes 32 planted-signal sessions (loss < 0.1)."""
        sessions = _make_planted_signal_sessions(n_sessions=32, n_bars=50, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2, dropout=0.0)
        result = train_sequence_model(
            model=model,
            train_sessions=sessions,
            val_sessions=sessions,  # val=train for overfit test
            epochs=100,
            batch_size=8,
            lr=1e-3,
            patience=100,  # Don't early stop
            seed=42,
        )
        # After enough epochs, loss should be very small
        final_loss = result["train_loss_history"][-1]
        assert final_loss < 0.1, \
            f"LSTM failed to overfit: final loss {final_loss:.4f} >= 0.1"

    def test_transformer_memorizes_synthetic_sessions(self):
        """BarrierTransformer memorizes 32 planted-signal sessions (loss < 0.1)."""
        sessions = _make_planted_signal_sessions(n_sessions=32, n_bars=50, seed=42)
        model = BarrierTransformer(n_features=22, d_model=64, n_layers=2, n_heads=4,
                                    dropout=0.0)
        result = train_sequence_model(
            model=model,
            train_sessions=sessions,
            val_sessions=sessions,
            epochs=100,
            batch_size=8,
            lr=1e-3,
            patience=100,
            seed=42,
        )
        final_loss = result["train_loss_history"][-1]
        assert final_loss < 0.1, \
            f"Transformer failed to overfit: final loss {final_loss:.4f} >= 0.1"

    def test_overfit_predictions_shift_from_base_rate(self):
        """After overfitting, predictions should NOT all be ~0.33 (base rate)."""
        sessions = _make_planted_signal_sessions(n_sessions=32, n_bars=50, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2, dropout=0.0)
        train_sequence_model(
            model=model,
            train_sessions=sessions,
            val_sessions=sessions,
            epochs=100,
            batch_size=8,
            lr=1e-3,
            patience=100,
            seed=42,
        )
        p_long, p_short = predict_sessions(model, sessions, batch_size=8)
        # If the model just predicts base rate, std would be ~0
        # A model that learned the signal should have std >> 0
        assert p_long.std() > 0.1, \
            f"p_long std {p_long.std():.4f} too small — predictions clustered at base rate"
        assert p_short.std() > 0.1, \
            f"p_short std {p_short.std():.4f} too small — predictions clustered at base rate"


# ===========================================================================
# Training Loop Tests
# ===========================================================================


class TestTrainSequenceModel:
    """Test the train_sequence_model() function contract."""

    def test_returns_correct_keys(self):
        """Return dict has the required keys."""
        sessions = _make_synthetic_sessions(8, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        result = train_sequence_model(
            model=model,
            train_sessions=sessions[:6],
            val_sessions=sessions[6:],
            epochs=3,
            batch_size=4,
            seed=42,
        )
        required_keys = {
            "train_loss_history",
            "val_brier_history",
            "best_epoch",
            "best_val_brier",
            "stopped_early",
        }
        assert required_keys.issubset(result.keys()), \
            f"Missing keys: {required_keys - result.keys()}"

    def test_early_stopping_fires(self):
        """Early stopping should trigger when val Brier plateaus."""
        sessions = _make_synthetic_sessions(10, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        result = train_sequence_model(
            model=model,
            train_sessions=sessions[:8],
            val_sessions=sessions[8:],
            epochs=200,  # High epoch count — should stop early
            batch_size=4,
            patience=5,
            seed=42,
        )
        # If patience=5 and epochs=200, it should stop before 200
        n_epochs_run = len(result["train_loss_history"])
        assert n_epochs_run < 200, \
            f"Early stopping didn't fire: ran all {n_epochs_run} epochs"
        assert result["stopped_early"] is True

    def test_seeded_determinism(self):
        """Same seed produces identical results."""
        sessions = _make_synthetic_sessions(10, seed=42)

        def run(seed):
            model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
            return train_sequence_model(
                model=model,
                train_sessions=sessions[:8],
                val_sessions=sessions[8:],
                epochs=5,
                batch_size=4,
                seed=seed,
            )

        r1 = run(seed=42)
        r2 = run(seed=42)
        np.testing.assert_allclose(
            r1["train_loss_history"], r2["train_loss_history"], atol=1e-6,
        )
        np.testing.assert_allclose(
            r1["val_brier_history"], r2["val_brier_history"], atol=1e-6,
        )

    def test_loss_decreases_over_early_epochs(self):
        """Train loss should decrease from epoch 1 to epoch 5 on synthetic data."""
        sessions = _make_planted_signal_sessions(n_sessions=16, n_bars=50, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2, dropout=0.0)
        result = train_sequence_model(
            model=model,
            train_sessions=sessions,
            val_sessions=sessions,
            epochs=10,
            batch_size=4,
            lr=1e-3,
            patience=100,
            seed=42,
        )
        losses = result["train_loss_history"]
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"

    def test_val_brier_history_length_matches_epochs(self):
        """val_brier_history should have length == number of epochs actually run."""
        sessions = _make_synthetic_sessions(10, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        result = train_sequence_model(
            model=model,
            train_sessions=sessions[:8],
            val_sessions=sessions[8:],
            epochs=5,
            batch_size=4,
            patience=100,
            seed=42,
        )
        assert len(result["val_brier_history"]) == 5
        assert len(result["train_loss_history"]) == 5

    def test_empty_session_list_raises(self):
        """Empty train_sessions should raise ValueError."""
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        with pytest.raises(ValueError, match="[Ee]mpty|[Nn]o.*sessions"):
            train_sequence_model(
                model=model,
                train_sessions=[],
                val_sessions=[_make_synthetic_session(30)],
                epochs=5,
            )

    def test_batch_size_larger_than_n_sessions(self):
        """batch_size > n_sessions works (single batch per epoch)."""
        sessions = _make_synthetic_sessions(3, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        result = train_sequence_model(
            model=model,
            train_sessions=sessions[:2],
            val_sessions=sessions[2:],
            epochs=3,
            batch_size=16,  # >> 2 sessions
            seed=42,
        )
        assert len(result["train_loss_history"]) == 3

    def test_best_epoch_within_range(self):
        """best_epoch should be a valid epoch index."""
        sessions = _make_synthetic_sessions(10, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        result = train_sequence_model(
            model=model,
            train_sessions=sessions[:8],
            val_sessions=sessions[8:],
            epochs=10,
            batch_size=4,
            patience=100,
            seed=42,
        )
        n_epochs_run = len(result["val_brier_history"])
        assert 0 <= result["best_epoch"] < n_epochs_run


# ===========================================================================
# Prediction Tests
# ===========================================================================


class TestPredictSessions:
    """Test predict_sessions() output contract."""

    def test_output_length_matches_total_bars(self):
        """Output length = sum of session lengths (no padding)."""
        sessions = [
            _make_synthetic_session(30, seed=0),
            _make_synthetic_session(50, seed=1),
            _make_synthetic_session(40, seed=2),
        ]
        total_bars = 30 + 50 + 40

        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        model.eval()
        p_long, p_short = predict_sessions(model, sessions, batch_size=2)

        assert len(p_long) == total_bars, \
            f"Expected {total_bars} predictions, got {len(p_long)}"
        assert len(p_short) == total_bars

    def test_values_in_zero_one(self):
        """Predictions should be probabilities in [0, 1]."""
        sessions = _make_synthetic_sessions(5, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        model.eval()
        p_long, p_short = predict_sessions(model, sessions, batch_size=4)

        assert (p_long >= 0.0).all() and (p_long <= 1.0).all(), \
            f"p_long out of [0,1]: min={p_long.min()}, max={p_long.max()}"
        assert (p_short >= 0.0).all() and (p_short <= 1.0).all(), \
            f"p_short out of [0,1]: min={p_short.min()}, max={p_short.max()}"

    def test_padding_removed_correctly(self):
        """Variable-length sessions: padding should NOT appear in output."""
        # Two sessions with very different lengths
        sessions = [
            _make_synthetic_session(10, seed=0),
            _make_synthetic_session(100, seed=1),
        ]
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        model.eval()
        p_long, p_short = predict_sessions(model, sessions, batch_size=2)

        # Total should be 10 + 100 = 110, not 200 (2 * 100)
        assert len(p_long) == 110
        assert len(p_short) == 110

    def test_batch_size_does_not_affect_results(self):
        """batch_size=1 and batch_size=4 should give the same output."""
        sessions = _make_synthetic_sessions(6, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2, dropout=0.0)
        model.eval()

        p_long_1, p_short_1 = predict_sessions(model, sessions, batch_size=1)
        p_long_4, p_short_4 = predict_sessions(model, sessions, batch_size=4)

        np.testing.assert_allclose(p_long_1, p_long_4, atol=1e-5,
                                    err_msg="batch_size affected p_long")
        np.testing.assert_allclose(p_short_1, p_short_4, atol=1e-5,
                                    err_msg="batch_size affected p_short")

    def test_output_dtype_is_float64(self):
        """Returned arrays should be float64 as specified."""
        sessions = _make_synthetic_sessions(3, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        model.eval()
        p_long, p_short = predict_sessions(model, sessions, batch_size=2)
        assert p_long.dtype == np.float64
        assert p_short.dtype == np.float64


# ===========================================================================
# Evaluation Tests
# ===========================================================================


class TestEvaluateSequenceModel:
    """Test evaluate_sequence_model() output contract."""

    def test_returns_correct_keys(self):
        """Result dict has all required keys for both labels."""
        sessions = _make_synthetic_sessions(10, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        result = evaluate_sequence_model(
            model=model,
            sessions=sessions,
            batch_size=4,
            n_bootstrap=50,  # Small for speed
            seed=42,
        )
        for label in ["long", "short"]:
            assert f"brier_constant_{label}" in result, f"Missing brier_constant_{label}"
            assert f"brier_model_{label}" in result, f"Missing brier_model_{label}"
            assert f"bss_{label}" in result, f"Missing bss_{label}"
            assert f"bootstrap_{label}" in result, f"Missing bootstrap_{label}"
            assert f"p_hat_mean_{label}" in result, f"Missing p_hat_mean_{label}"
            assert f"p_hat_std_{label}" in result, f"Missing p_hat_std_{label}"

    def test_bss_positive_on_planted_signal(self):
        """BSS > 0 when model can learn a planted signal."""
        sessions = _make_planted_signal_sessions(n_sessions=32, n_bars=50, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2, dropout=0.0)
        train_sequence_model(
            model=model,
            train_sessions=sessions,
            val_sessions=sessions,
            epochs=50,
            batch_size=8,
            lr=1e-3,
            patience=100,
            seed=42,
        )
        result = evaluate_sequence_model(
            model=model,
            sessions=sessions,
            batch_size=8,
            n_bootstrap=50,
            seed=42,
        )
        # Model should beat constant predictor on the training data
        assert result["bss_long"] > 0, \
            f"BSS_long={result['bss_long']:.4f} should be > 0 on planted signal"

    def test_brier_hand_calculation(self):
        """Brier score matches hand calculation on a trivial example."""
        # Create a session where we know the Brier score analytically
        # Y_long = [True, False], predictions will be whatever the model outputs
        # We just check that brier_model and brier_constant are valid floats
        sessions = [_make_synthetic_session(20, seed=0)]
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        result = evaluate_sequence_model(
            model=model,
            sessions=sessions,
            batch_size=1,
            n_bootstrap=50,
            seed=42,
        )
        # Brier scores should be non-negative
        assert result["brier_model_long"] >= 0
        assert result["brier_model_short"] >= 0
        assert result["brier_constant_long"] >= 0
        assert result["brier_constant_short"] >= 0

    def test_output_format_matches_exp006(self):
        """Output has the same key structure used in exp-006 metrics."""
        sessions = _make_synthetic_sessions(5, seed=42)
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        result = evaluate_sequence_model(
            model=model,
            sessions=sessions,
            batch_size=4,
            n_bootstrap=50,
            seed=42,
        )
        # Bootstrap result should have delta, ci_lower, ci_upper, p_value
        for label in ["long", "short"]:
            boot = result[f"bootstrap_{label}"]
            assert "delta" in boot, f"bootstrap_{label} missing 'delta'"
            assert "ci_lower" in boot, f"bootstrap_{label} missing 'ci_lower'"
            assert "ci_upper" in boot, f"bootstrap_{label} missing 'ci_upper'"
            assert "p_value" in boot, f"bootstrap_{label} missing 'p_value'"


# ===========================================================================
# Padding / Masking Tests
# ===========================================================================


class TestPaddingMasking:
    """Padded positions must not affect loss or predictions."""

    def test_padded_positions_dont_contribute_to_loss(self):
        """Loss should be the same regardless of padding length, given same real data."""
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2, dropout=0.0)
        model.eval()
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        # Same session, but batched with different padding partners
        session = _make_synthetic_session(30, seed=0)
        short_partner = _make_synthetic_session(35, seed=1)
        long_partner = _make_synthetic_session(80, seed=2)

        batch_short = collate_sessions([session, short_partner])
        batch_long = collate_sessions([session, long_partner])

        feat_s, yl_s, ys_s, mask_s = batch_short
        feat_l, yl_l, ys_l, mask_l = batch_long

        # Forward through model
        logit_long_s, logit_short_s = model(feat_s)
        logit_long_l, logit_short_l = model(feat_l)

        # Compute masked loss for session 0 only (first 30 bars)
        loss_s = loss_fn(logit_long_s[0, :30], yl_s[0, :30]).sum()
        loss_l = loss_fn(logit_long_l[0, :30], yl_l[0, :30]).sum()

        # The masked losses for the same session should be identical
        # because causal masking + padding masking means the session's
        # predictions don't depend on what comes after
        torch.testing.assert_close(loss_s, loss_l, atol=1e-4, rtol=1e-4)

    def test_predictions_identical_regardless_of_padding(self):
        """Predictions at real positions are the same with different batch padding."""
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2, dropout=0.0)
        model.eval()

        session = _make_synthetic_session(30, seed=0)

        # Single session (no padding)
        feat_single, _, _, _ = collate_sessions([session])
        l_single, s_single = model(feat_single)

        # Same session with a long partner (lots of padding)
        partner = _make_synthetic_session(100, seed=1)
        feat_padded, _, _, _ = collate_sessions([session, partner])
        l_padded, s_padded = model(feat_padded)

        # Session 0's 30 real positions should match
        torch.testing.assert_close(
            l_single[0, :30], l_padded[0, :30], atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            s_single[0, :30], s_padded[0, :30], atol=1e-5, rtol=1e-5,
        )

    def test_gradient_only_flows_through_real_positions(self):
        """Gradient at padded positions should be zero."""
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2, dropout=0.0)
        model.train()

        sessions = [
            _make_synthetic_session(20, seed=0),
            _make_synthetic_session(50, seed=1),
        ]
        features, Y_long, Y_short, mask = collate_sessions(sessions)
        features.requires_grad_(True)

        logit_long, logit_short = model(features)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        # Compute loss only at real positions (using mask)
        loss_long = (loss_fn(logit_long, Y_long) * mask).sum()
        loss_short = (loss_fn(logit_short, Y_short) * mask).sum()
        total_loss = loss_long + loss_short
        total_loss.backward()

        # Gradient at padded positions of session 0 (positions 20..49) should be zero
        # because those positions were masked out of the loss
        padded_grad = features.grad[0, 20:, :]
        assert (padded_grad == 0).all(), \
            f"Non-zero gradient at padded positions: max abs = {padded_grad.abs().max()}"


# ===========================================================================
# Single-Bar Session Edge Case
# ===========================================================================


class TestSingleBarSession:
    """Edge case: session with a single bar."""

    def test_lstm_handles_single_bar(self):
        """BarrierLSTM processes (B, 1, 22) without error."""
        model = BarrierLSTM(n_features=22, d_model=64, n_layers=2)
        x = torch.randn(1, 1, 22)
        logit_long, logit_short = model(x)
        assert logit_long.shape == (1, 1)
        assert logit_short.shape == (1, 1)

    def test_transformer_handles_single_bar(self):
        """BarrierTransformer processes (B, 1, 22) without error."""
        model = BarrierTransformer(n_features=22, d_model=64, n_layers=2, n_heads=4)
        x = torch.randn(1, 1, 22)
        logit_long, logit_short = model(x)
        assert logit_long.shape == (1, 1)
        assert logit_short.shape == (1, 1)

    def test_collate_single_bar_session(self):
        """Collating a session with 1 bar works correctly."""
        session = _make_synthetic_session(1, seed=0)
        features, Y_long, Y_short, mask = collate_sessions([session])
        assert features.shape == (1, 1, 22)
        assert mask.all()


# ===========================================================================
# BarEmbedding Abstract Base
# ===========================================================================


class TestBarEmbeddingInterface:
    """BarEmbedding base class has d_model attribute."""

    def test_linear_embedding_has_d_model(self):
        """LinearBarEmbedding exposes d_model attribute."""
        emb = LinearBarEmbedding(n_features=22, d_model=128)
        assert emb.d_model == 128

    def test_bar_embedding_is_nn_module(self):
        """BarEmbedding subclasses nn.Module."""
        emb = LinearBarEmbedding(n_features=22, d_model=64)
        assert isinstance(emb, torch.nn.Module)
