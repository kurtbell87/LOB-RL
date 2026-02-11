"""Sequence model infrastructure for barrier prediction (LSTM, Transformer).

Processes full trading sessions as sequences, predicting per-bar barrier outcomes
with causal context. Includes swappable embedding interface, training loop with
Brier evaluation, and integration with first_passage_analysis scoring functions.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from lob_rl.barrier.first_passage_analysis import (
    brier_score,
    brier_skill_score,
    constant_brier,
    paired_bootstrap_brier,
)


# ===========================================================================
# Embedding
# ===========================================================================


class BarEmbedding(nn.Module):
    """Abstract base for bar embeddings.

    Subclasses must set ``self.d_model`` and implement
    ``forward(x: [B, T, n_features]) -> [B, T, d_model]``.
    """

    d_model: int


class LinearBarEmbedding(BarEmbedding):
    """Linear projection + sinusoidal positional encoding.

    One token per bar (T_out == T_in).
    """

    def __init__(
        self,
        n_features: int = 22,
        d_model: int = 64,
        max_len: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(n_features, d_model)
        self.dropout = nn.Dropout(dropout)

        # Sinusoidal positional encoding (registered as buffer, not parameter)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, n_features) -> (B, T, d_model)."""
        h = self.proj(x)  # (B, T, d_model)
        h = h + self.pe[:, : h.size(1), :]
        return self.dropout(h)


# ===========================================================================
# Models
# ===========================================================================


class _DualHeadModel(nn.Module):
    """Base class for barrier models with shared long/short output heads."""

    def _init_heads(self, d: int) -> None:
        self.head_long = nn.Linear(d, 1)
        self.head_short = nn.Linear(d, 1)

    def _apply_heads(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, d) -> (logit_long (B,T), logit_short (B,T))."""
        return self.head_long(h).squeeze(-1), self.head_short(h).squeeze(-1)


class BarrierLSTM(_DualHeadModel):
    """Bidirectional=False LSTM with per-step linear heads for Y_long / Y_short."""

    def __init__(
        self,
        n_features: int = 22,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        embedding: BarEmbedding | None = None,
    ):
        super().__init__()
        self.embedding = embedding or LinearBarEmbedding(n_features, d_model, dropout=dropout)
        d = self.embedding.d_model
        self.lstm = nn.LSTM(
            input_size=d,
            hidden_size=d,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self._init_heads(d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, n_features) -> (logit_long (B,T), logit_short (B,T))."""
        h = self.embedding(x)  # (B, T, d)
        h, _ = self.lstm(h)    # (B, T, d)
        return self._apply_heads(h)


class BarrierTransformer(_DualHeadModel):
    """Causal Transformer encoder with per-step output heads."""

    def __init__(
        self,
        n_features: int = 22,
        d_model: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        embedding: BarEmbedding | None = None,
    ):
        super().__init__()
        self.embedding = embedding or LinearBarEmbedding(n_features, d_model, dropout=dropout)
        d = self.embedding.d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=2 * d,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self._init_heads(d)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, T, n_features) -> (logit_long (B,T), logit_short (B,T))."""
        h = self.embedding(x)  # (B, T, d)
        T = h.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1
        )
        h = self.encoder(h, mask=causal_mask)  # (B, T, d)
        return self._apply_heads(h)


# ===========================================================================
# Collation
# ===========================================================================


def collate_sessions(sessions: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sessions into a batch.

    Args:
        sessions: list of dicts, each with:
            features: (T_i, n_features) float32
            Y_long: (T_i,) bool
            Y_short: (T_i,) bool

    Returns:
        features: (B, T_max, n_features) float32, zero-padded
        Y_long: (B, T_max) float32
        Y_short: (B, T_max) float32
        mask: (B, T_max) bool — True for real, False for padding
    """
    B = len(sessions)
    lengths = [len(s["features"]) for s in sessions]
    T_max = max(lengths)
    n_feat = sessions[0]["features"].shape[1]

    features = torch.zeros(B, T_max, n_feat, dtype=torch.float32)
    Y_long = torch.zeros(B, T_max, dtype=torch.float32)
    Y_short = torch.zeros(B, T_max, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, s in enumerate(sessions):
        T_i = lengths[i]
        features[i, :T_i] = torch.from_numpy(np.asarray(s["features"], dtype=np.float32))
        Y_long[i, :T_i] = torch.from_numpy(np.asarray(s["Y_long"], dtype=np.float32))
        Y_short[i, :T_i] = torch.from_numpy(np.asarray(s["Y_short"], dtype=np.float32))
        mask[i, :T_i] = True

    return features, Y_long, Y_short, mask


# ===========================================================================
# Helpers
# ===========================================================================


def _extract_true_labels(
    sessions: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate ground-truth labels across sessions.

    Returns (y_long, y_short) each as float64 arrays.
    """
    y_long = np.concatenate([np.asarray(s["Y_long"], dtype=np.float64) for s in sessions])
    y_short = np.concatenate([np.asarray(s["Y_short"], dtype=np.float64) for s in sessions])
    return y_long, y_short


# ===========================================================================
# Training
# ===========================================================================


def train_sequence_model(
    model: nn.Module,
    train_sessions: list[dict],
    val_sessions: list[dict],
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Train a sequence model on barrier prediction.

    Loss: BCEWithLogitsLoss per bar, masked for padding. Sum of Y_long + Y_short losses.
    Optimizer: AdamW with cosine annealing (T_max=epochs).
    Early stopping: patience epochs on mean val Brier score (lower is better).
    """
    if not train_sessions:
        raise ValueError("Empty train_sessions: no sessions provided for training")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Re-initialize model parameters deterministically so that the same seed
    # always produces the same weights, regardless of prior RNG state.
    def _reset_params(m: nn.Module):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    model.apply(_reset_params)

    rng = np.random.RandomState(seed)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    train_loss_history: list[float] = []
    val_brier_history: list[float] = []
    best_val_brier = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    stopped_early = False

    n_train = len(train_sessions)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        indices = rng.permutation(n_train)
        epoch_loss = 0.0
        n_real_bars = 0

        for start in range(0, n_train, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_sessions = [train_sessions[i] for i in batch_idx]
            features, yl, ys, mask = collate_sessions(batch_sessions)
            features = features.to(device)
            yl = yl.to(device)
            ys = ys.to(device)
            mask = mask.to(device)

            logit_long, logit_short = model(features)

            # Masked loss
            loss_long = (loss_fn(logit_long, yl) * mask).sum()
            loss_short = (loss_fn(logit_short, ys) * mask).sum()
            n_bars = mask.sum().item()
            total_loss = (loss_long + loss_short) / max(n_bars, 1)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += (loss_long + loss_short).item()
            n_real_bars += n_bars

        avg_train_loss = epoch_loss / max(n_real_bars, 1)
        train_loss_history.append(avg_train_loss)
        scheduler.step()

        # --- Val ---
        model.eval()
        p_long, p_short = predict_sessions(model, val_sessions, batch_size=batch_size, device=device)
        y_long_true, y_short_true = _extract_true_labels(val_sessions)

        brier_long = brier_score(y_long_true, p_long)
        brier_short = brier_score(y_short_true, p_short)
        mean_brier = (brier_long + brier_short) / 2.0
        val_brier_history.append(mean_brier)

        if mean_brier < best_val_brier:
            best_val_brier = mean_brier
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            stopped_early = True
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return {
        "train_loss_history": train_loss_history,
        "val_brier_history": val_brier_history,
        "best_epoch": best_epoch,
        "best_val_brier": best_val_brier,
        "stopped_early": stopped_early,
    }


# ===========================================================================
# Prediction
# ===========================================================================


def predict_sessions(
    model: nn.Module,
    sessions: list[dict],
    batch_size: int = 8,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference on sessions, return per-bar probability predictions.

    Returns:
        p_long: (N_total,) float64 — predicted P(Y_long=1) for each bar
        p_short: (N_total,) float64 — predicted P(Y_short=1) for each bar

    Predictions are concatenated across sessions with padding removed.
    """
    model.eval()
    all_long = []
    all_short = []

    n_sessions = len(sessions)
    lengths = [len(s["features"]) for s in sessions]

    with torch.no_grad():
        for start in range(0, n_sessions, batch_size):
            batch_sessions = sessions[start : start + batch_size]
            batch_lengths = lengths[start : start + batch_size]
            features, _, _, mask = collate_sessions(batch_sessions)
            features = features.to(device)

            logit_long, logit_short = model(features)
            prob_long = torch.sigmoid(logit_long).cpu().numpy()
            prob_short = torch.sigmoid(logit_short).cpu().numpy()

            # Extract only real (non-padded) positions
            for i, T_i in enumerate(batch_lengths):
                all_long.append(prob_long[i, :T_i])
                all_short.append(prob_short[i, :T_i])

    p_long = np.concatenate(all_long).astype(np.float64)
    p_short = np.concatenate(all_short).astype(np.float64)
    return p_long, p_short


# ===========================================================================
# Evaluation
# ===========================================================================


def evaluate_sequence_model(
    model: nn.Module,
    sessions: list[dict],
    batch_size: int = 8,
    device: str = "cpu",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Evaluate model using Brier score framework, compatible with exp-006.

    Returns dict with keys (for each label in [long, short]):
        brier_constant_{label}: float
        brier_model_{label}: float
        bss_{label}: float
        bootstrap_{label}: dict with delta, ci_lower, ci_upper, p_value
        p_hat_mean_{label}: float
        p_hat_std_{label}: float
    """
    p_long, p_short = predict_sessions(model, sessions, batch_size=batch_size, device=device)
    y_long_true, y_short_true = _extract_true_labels(sessions)

    result = {}
    for label, y_true, p_hat in [("long", y_long_true, p_long), ("short", y_short_true, p_short)]:
        ybar = y_true.mean()
        pred_baseline = np.full_like(y_true, ybar)

        brier_const = constant_brier(y_true)
        brier_mod = brier_score(y_true, p_hat)
        bss = brier_skill_score(y_true, p_hat)

        boot = paired_bootstrap_brier(
            y_true, p_hat, pred_baseline, n_boot=n_bootstrap, seed=seed,
        )

        result[f"brier_constant_{label}"] = brier_const
        result[f"brier_model_{label}"] = brier_mod
        result[f"bss_{label}"] = bss
        result[f"bootstrap_{label}"] = boot
        result[f"p_hat_mean_{label}"] = float(p_hat.mean())
        result[f"p_hat_std_{label}"] = float(p_hat.std())

    return result
