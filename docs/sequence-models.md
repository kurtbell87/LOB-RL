# Sequence Model Infrastructure

## Overview

Build sequence-aware models (LSTM, Transformer) for barrier prediction. These models process full trading sessions as sequences, predicting per-bar barrier outcomes with causal context. Includes swappable embedding interface, training loop with Brier evaluation, and integration with existing `first_passage_analysis.py` scoring functions.

## What to Build

### New file: `python/lob_rl/barrier/sequence_models.py`

#### 1. Swappable Embedding Interface

```python
class BarEmbedding(nn.Module):
    """Abstract base. Subclasses: forward(x: [B, T, n_features]) -> [B, T', d_model]"""
    d_model: int  # must be set by subclass

class LinearBarEmbedding(BarEmbedding):
    """Strategy C: Linear(n_features, d_model) + sinusoidal positional encoding.
    Token_len = seq_len (one token per bar)."""
    def __init__(self, n_features: int = 22, d_model: int = 64,
                 max_len: int = 2048, dropout: float = 0.1)
    def forward(self, x: Tensor) -> Tensor:
        """(B, T, n_features) -> (B, T, d_model)"""
```

Sinusoidal positional encoding: standard sin/cos from "Attention is All You Need". Registered as a buffer (not a parameter).

#### 2. Models

All share the same interface:
- Constructor takes `n_features: int = 22, d_model: int = 64, n_layers: int = 2, dropout: float = 0.1`
- Plus model-specific args (n_heads for Transformer)
- Accept optional `embedding: BarEmbedding` (defaults to `LinearBarEmbedding`)
- `forward(x: Tensor[B, T, 22]) -> tuple[Tensor[B, T], Tensor[B, T]]` — (logit_long, logit_short)

```python
class BarrierLSTM(nn.Module):
    """2-layer LSTM, per-step linear heads for Y_long and Y_short."""
    def __init__(self, n_features=22, d_model=64, n_layers=2, dropout=0.1,
                 embedding: BarEmbedding | None = None)

class BarrierTransformer(nn.Module):
    """Causal Transformer encoder. Uses nn.TransformerEncoder with causal mask.
    4 attention heads, ff_dim = 2 * d_model."""
    def __init__(self, n_features=22, d_model=64, n_layers=2, n_heads=4,
                 dropout=0.1, embedding: BarEmbedding | None = None)
```

**Causal masking:** The Transformer uses `nn.Transformer.generate_square_subsequent_mask()` or equivalent `torch.triu` mask so bar k only attends to bars 0..k. LSTM is naturally causal.

**Output heads:** Two independent `nn.Linear(d_model, 1)` heads — one for Y_long logits, one for Y_short logits. Squeeze the last dim to get `(B, T)`.

**Parameter budget:** ~25-50K params with d_model=64, n_layers=2. Deliberately small — signal detection, not tuning.

#### 3. Collation and Batching

```python
def collate_sessions(sessions: list[dict]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pad variable-length sessions into a batch.

    Args:
        sessions: list of dicts from load_session_features(), each with:
            features: (T_i, 22) float32
            Y_long: (T_i,) bool
            Y_short: (T_i,) bool

    Returns:
        features: (B, T_max, 22) float32, zero-padded
        Y_long: (B, T_max) float32
        Y_short: (B, T_max) float32
        mask: (B, T_max) bool — True for real positions, False for padding
    """
```

#### 4. Training Loop

```python
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

    Returns dict with keys:
        train_loss_history: list[float] — per-epoch mean train loss
        val_brier_history: list[float] — per-epoch mean val Brier
        best_epoch: int
        best_val_brier: float
        stopped_early: bool
    """
```

**Seeding:** Set `torch.manual_seed(seed)`, `np.random.seed(seed)`, and use a `torch.Generator` for DataLoader shuffling.

**Batch iteration:** Shuffle train sessions each epoch. Chunk into batches of `batch_size`. Call `collate_sessions()` on each batch.

**Val evaluation:** After each epoch, run `predict_sessions()` on val set, compute mean Brier score across both labels, check early stopping.

#### 5. Prediction

```python
def predict_sessions(model: nn.Module, sessions: list[dict],
                     batch_size: int = 8, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """Run model inference on sessions, return per-bar probability predictions.

    Returns:
        p_long: (N_total,) float64 — predicted P(Y_long=1) for each bar
        p_short: (N_total,) float64 — predicted P(Y_short=1) for each bar

    Predictions are concatenated across sessions (variable length), with
    padding removed. N_total = sum of session lengths.
    """
```

#### 6. Evaluation

```python
def evaluate_sequence_model(
    model: nn.Module,
    sessions: list[dict],
    batch_size: int = 8,
    device: str = "cpu",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Evaluate model using Brier score framework, compatible with exp-006.

    Uses existing functions from first_passage_analysis:
        brier_score(), brier_skill_score(), paired_bootstrap_brier()

    Returns dict with keys (for each label in [long, short]):
        brier_constant_{label}: float
        brier_model_{label}: float
        bss_{label}: float
        bootstrap_{label}: dict (from paired_bootstrap_brier)
        p_hat_mean_{label}: float
        p_hat_std_{label}: float
    """
```

## Edge Cases

1. **Single-bar sessions:** Model receives (B, 1, 22). LSTM produces one output. Transformer has trivial self-attention. Should work but may have degenerate gradients — test this.
2. **All-padding batches:** Cannot happen since collate requires at least one real session, but mask logic should handle it gracefully.
3. **Very long sessions (~2000 bars):** Transformer attention is O(T^2). At T=2000, d_model=64, this is ~32MB — fine for CPU.
4. **NaN in features:** Should not happen (z-score normalization clips to [-5, 5]), but check and raise if detected.
5. **Empty session list:** `train_sequence_model` should raise `ValueError`.
6. **All labels identical in a session:** Valid — model should learn the base rate.

## Acceptance Criteria

1. `LinearBarEmbedding` produces (B, T, d_model) from (B, T, 22). Positional encoding varies by position.
2. `BarrierLSTM.forward()` returns (logit_long, logit_short) both shape (B, T).
3. `BarrierTransformer.forward()` returns (logit_long, logit_short) both shape (B, T).
4. Causal masking: perturbing future bars does NOT change past predictions in Transformer.
5. Each model can overfit 32 synthetic sessions to near-zero loss.
6. `collate_sessions()` correctly pads and creates masks.
7. `train_sequence_model()` returns correct keys, respects early stopping and seeding.
8. `evaluate_sequence_model()` output is compatible with exp-006 metrics format.
9. Padded positions do not affect loss or predictions.
10. `predict_sessions()` returns calibrated probabilities (sigmoid of logits) in [0, 1].

## Tests (~35-40)

### Embedding Tests (4)
- LinearBarEmbedding output shape: (B, T, d_model)
- Positional encoding varies by position (pos 0 != pos 1)
- Dropout applied during training
- Embedding accepts custom n_features and d_model

### Model Shape Tests (6)
- BarrierLSTM forward: (B, T, 22) -> two (B, T) tensors
- BarrierTransformer forward: (B, T, 22) -> two (B, T) tensors
- Both work with variable T across calls
- Both work with batch_size=1
- Parameter count is in expected range (25-50K for d_model=64)
- Custom embedding is used when provided

### Causal Masking Tests (2)
- Transformer: perturb bar T, predictions for bars 0..T-1 unchanged
- Transformer: different future padding doesn't affect past predictions

### Collation Tests (4)
- Correct padding shape (B, T_max, 22)
- Mask True for real, False for padding
- Labels padded to T_max with 0s
- Single-session batch works

### Overfit Tests (3)
- BarrierLSTM memorizes 32 synthetic sessions (loss < 0.1 after N epochs)
- BarrierTransformer memorizes 32 synthetic sessions (loss < 0.1 after N epochs)
- Overfitting produces predictions that shift away from base rate

### Training Loop Tests (8)
- Returns correct keys
- Early stopping fires when val loss plateaus
- Predictions in [0, 1]
- Seeded determinism: same seed -> same results
- Loss decreases over first few epochs on synthetic data
- val_brier_history has length == number of epochs run
- Empty session list raises ValueError
- batch_size > n_sessions works (single batch)

### Prediction Tests (4)
- Output length matches total bars across sessions
- Values in [0, 1]
- Padding removed correctly
- Batch size doesn't affect results (batch_size=1 vs 4 give same output)

### Evaluation Tests (4)
- Returns correct keys for long and short
- BSS > 0 on planted signal (synthetic data where one direction is predictable)
- Brier matches hand calculation on small example
- Output format matches exp-006 structure

### Padding/Masking Tests (3)
- Padded positions don't contribute to loss
- Predictions at real positions are identical regardless of padding length
- Gradient only flows through real positions

## Key Files

| File | Action |
|------|--------|
| `python/lob_rl/barrier/sequence_models.py` | **CREATE** |
| `python/tests/barrier/test_sequence_models.py` | **CREATE** (by RED phase) |
| `python/lob_rl/barrier/README.md` | **UPDATE** |

## Dependencies

| Module | What's Used |
|--------|------------|
| `first_passage_analysis.py` | `brier_score()`, `brier_skill_score()`, `paired_bootstrap_brier()`, `load_session_features()`, `temporal_split()` |
| `supervised_diagnostic.py` | Pattern reference: `_train_loop()`, `BarrierMLP`, `evaluate_classifier()` |
| `torch` | `nn.Module`, `nn.LSTM`, `nn.TransformerEncoder`, `nn.Linear`, `BCEWithLogitsLoss`, `AdamW` |
| `numpy` | Array operations for evaluation |

## Modification Hints

To add a new embedding strategy (e.g., PatchBarEmbedding):
1. Subclass `BarEmbedding`
2. Implement `forward(x: [B, T, n_features]) -> [B, T', d_model]`
3. Set `self.d_model = ...`
4. Pass to any model constructor: `BarrierTransformer(embedding=PatchBarEmbedding(...))`

To add a new model architecture:
1. Follow the same interface: `forward(x: [B, T, 22]) -> (logit_long: [B, T], logit_short: [B, T])`
2. Accept `embedding: BarEmbedding | None` in constructor
3. Use `self.embedding = embedding or LinearBarEmbedding(n_features, d_model)`
