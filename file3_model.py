"""
=============================================================
 MMFi Dataset — File 3: Transformer Encoder Model
 Pure Transformer Encoder — Dual Head (root + keypoints)
=============================================================
Input  : (batch, 114, 60)  — 114 subcarrier tokens, dim=60
Output :
  root_pred : (batch, 3)   — (x, y, z) root position
  kp_pred   : (batch, 51)  — 17 joints × 3 coords
=============================================================
Architecture Rationale
─────────────────────
Each subcarrier acts as a "token" (analogous to a word in NLP).
The Transformer Encoder learns relationships between subcarriers
(cross-subcarrier attention), which captures multipath effects
and spatial interference patterns across the frequency domain.
A learnable [CLS] token aggregates global context, which is then
split into two regression heads: one for root position, one for
all 17 body keypoints.
=============================================================
"""

import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────
# POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (Vaswani et al., 2017).
    Adds position-dependent sine/cosine signals to token embeddings.

    Allows the Transformer to distinguish subcarrier ordering
    (frequency position) — critical for CSI as subcarrier index
    corresponds to frequency, which carries path information.
    """

    def __init__(self, d_model: int, max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe  = torch.zeros(max_len, d_model)                     # (max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()     # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)                      # even dims
        pe[:, 1::2] = torch.cos(pos * div)                      # odd dims
        pe = pe.unsqueeze(0)                                     # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────
# TRANSFORMER ENCODER MODEL
# ─────────────────────────────────────────────────────────────
class CSITransformerEncoder(nn.Module):
    """
    Pure Transformer Encoder for WiFi CSI → 3D Position Estimation.

    Architecture
    ────────────
    1. Input Projection   : Linear(raw_token_dim=60 → d_model=128)
    2. CLS Token          : Learnable [CLS] prepended to token sequence
    3. Positional Encoding: Sinusoidal (subcarrier frequency ordering)
    4. Transformer Encoder: N layers of Multi-Head Self-Attention + FFN
    5. Root Head          : MLP → (3,)
    6. Keypoint Head      : MLP → (51,)

    Parameters
    ──────────
    raw_token_dim : input token dimension (3 antennas × 10 packets × 2 = 60)
    n_tokens      : number of tokens / subcarriers (114)
    d_model       : Transformer hidden dimension (128)
    n_heads       : number of attention heads (8)  — d_model must be divisible
    n_layers      : number of Transformer encoder layers (4)
    d_ff          : feed-forward inner dimension (512 = 4 × d_model)
    dropout       : dropout probability (0.1)
    """

    def __init__(self,
                 raw_token_dim: int   = 60,
                 n_tokens:      int   = 114,
                 d_model:       int   = 128,
                 n_heads:       int   = 8,
                 n_layers:      int   = 4,
                 d_ff:          int   = 512,
                 dropout:       float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model  = d_model
        self.n_tokens = n_tokens

        # ── 1. Input Projection ─────────────────────────────
        # Project raw token dim (60) → d_model (128)
        self.input_proj = nn.Sequential(
            nn.Linear(raw_token_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── 2. Learnable CLS Token ───────────────────────────
        # Shape: (1, 1, d_model) — broadcast over batch
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # ── 3. Positional Encoding ───────────────────────────
        # seq_len = n_tokens + 1 (for CLS)
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=n_tokens + 1, dropout=dropout
        )

        # ── 4. Transformer Encoder ───────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = d_model,
            nhead      = n_heads,
            dim_feedforward = d_ff,
            dropout    = dropout,
            activation = "gelu",          # GELU smoother than ReLU for Transformers
            batch_first = True,           # (B, seq, d_model) convention
            norm_first  = True,           # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_layers,
            norm       = nn.LayerNorm(d_model),
        )

        # ── 5. Root Position Head ────────────────────────────
        # CLS token → 3D root position
        self.root_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 3),             # (x, y, z)
        )

        # ── 6. Keypoint Head ─────────────────────────────────
        # CLS token → 17 joints × 3 = 51 values
        self.keypoint_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 51),           # 17 joints × 3 coords
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for linear layers, zero bias."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 114, 60)  — batch of subcarrier token sequences

        Returns
        -------
        root_pred : (B, 3)
        kp_pred   : (B, 51)
        cls_feat  : (B, d_model)  — CLS embedding (used in loss)
        """
        B = x.size(0)

        # Project raw tokens to d_model
        x = self.input_proj(x)                              # (B, 114, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)             # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                   # (B, 115, d_model)

        # Positional encoding
        x = self.pos_enc(x)                                 # (B, 115, d_model)

        # Transformer encoder
        x = self.transformer(x)                             # (B, 115, d_model)

        # Extract CLS token output (position 0)
        cls_feat = x[:, 0, :]                               # (B, d_model)

        # Dual regression heads
        root_pred = self.root_head(cls_feat)                # (B, 3)
        kp_pred   = self.keypoint_head(cls_feat)            # (B, 51)

        return root_pred, kp_pred, cls_feat


# ─────────────────────────────────────────────────────────────
# QUICK VALIDATION
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Device] {device}")

    model = CSITransformerEncoder(
        raw_token_dim = 60,
        n_tokens      = 114,
        d_model       = 128,
        n_heads       = 8,
        n_layers      = 4,
        d_ff          = 512,
        dropout       = 0.1,
    ).to(device)

    total  = sum(p.numel() for p in model.parameters())
    trainp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model]  Total parameters     : {total:,}")
    print(f"[Model]  Trainable parameters : {trainp:,}")

    dummy      = torch.randn(4, 114, 60).to(device)
    root, kp, feat = model(dummy)

    print(f"\n[Forward Test]")
    print(f"  Input    : {dummy.shape}   → (B, 114, 60)")
    print(f"  root_pred: {root.shape}    → (B, 3)")
    print(f"  kp_pred  : {kp.shape}      → (B, 51)")
    print(f"  cls_feat : {feat.shape}    → (B, 128)")
    print("\n✓ Model forward pass successful")
