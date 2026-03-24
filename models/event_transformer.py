"""
Stage 3: Lightweight Event Transformer
───────────────────────────────────────
2-layer Transformer encoder (128D, 4 heads) with:
  - Sinusoidal temporal positional encoding
  - Learnable event type embedding
  - Masked attention for variable-length sequences
  - [CLS] token for classification
  - Temperature-scaled sigmoid for calibrated P(ASD)

Output: Calibrated probability of ASD.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        """x: [B, T, D]"""
        return x + self.pe[:, : x.size(1), :]


class EventTransformer(nn.Module):
    """
    Lightweight 2-layer Transformer for ASD event classification.

    Features:
    - [CLS] token prepended to sequence
    - Sinusoidal positional encoding
    - Learnable event type embedding (2 event types)
    - Temperature-scaled calibrated output
    - Proper calibration on held-out data (improvement #7)
    """

    def __init__(
        self,
        d_model=config.TRANSFORMER_DIM,
        n_heads=config.TRANSFORMER_HEADS,
        n_layers=config.TRANSFORMER_LAYERS,
        d_ff=config.TRANSFORMER_FF_DIM,
        dropout=config.TRANSFORMER_DROPOUT,
        num_event_types=3,
    ):
        super().__init__()
        self.d_model = d_model

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model)

        # Event type embedding (e.g., face-dominant, motion-dominant, mixed)
        self.event_embed = nn.Embedding(num_event_types, d_model)

        # Input projection (in case input dim != d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Temperature parameter for calibration (improvement #7)
        # Initialized at 1.0, optimized separately during calibration
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x, event_types=None, src_key_padding_mask=None):
        """
        Args:
            x: [B, K, D] — selected salient clips from Stage 2
            event_types: [B, K] — optional event type indices
            src_key_padding_mask: [B, K+1] — True = padded (masked out)

        Returns:
            logit: [B, 1] — raw logit (before sigmoid)
            calibrated_prob: [B, 1] — temperature-scaled probability
        """
        B, K, D = x.shape

        # Project input
        x = self.input_norm(self.input_proj(x))

        # Add event type embedding if provided
        if event_types is not None:
            x = x + self.event_embed(event_types)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)  # [B, K+1, D]

        # Add positional encoding
        x = self.pos_enc(x)

        # Transformer forward
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Take [CLS] token output
        cls_output = x[:, 0, :]  # [B, D]

        # Classification
        logit = self.classifier(cls_output)  # [B, 1]

        # Temperature-scaled calibrated probability using fixed T=2.5 parameter
        calibrated_prob = torch.sigmoid(logit / config.INFERENCE_TEMPERATURE)

        return logit, calibrated_prob

    def calibrate_temperature(self, val_logits, val_labels, max_iter=50):
        """
        Post-hoc temperature scaling on validation data (improvement #7).

        Args:
            val_logits: [N, 1] — raw logits from validation set
            val_labels: [N] — ground truth labels
        """
        self.temperature.requires_grad_(True)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = torch.sigmoid(val_logits / self.temperature)
            loss = F.binary_cross_entropy(scaled.squeeze(), val_labels.float())
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature.requires_grad_(False)
        return self.temperature.item()
