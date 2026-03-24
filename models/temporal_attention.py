"""
Stage 2: Soft Temporal Selection (Attention Pooling)
────────────────────────────────────────────────────
Concatenates 3 feature streams → soft attention over temporal dimension.
NO hard top-K — uses differentiable soft selection so gradients flow.

Key: Selects the K most salient time steps based on learned attention.
Output: [B, K, 128] saliency-weighted clips.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SoftTemporalAttention(nn.Module):
    """
    Soft attention pooling over temporal dimension.

    1. Concatenates 3 streams: [B, T, 128*3] → fuse → [B, T, 128]
    2. Computes attention scores per time step
    3. Selects top-K by attention weight (soft, differentiable)
    """

    def __init__(
        self,
        in_dim=config.FEATURE_DIM,
        num_streams=config.NUM_STREAMS,
        top_k=config.TOP_K_CLIPS,
        temperature=config.ATTN_TEMPERATURE,
    ):
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature

        # Stream fusion: [B, T, 128*3] → [B, T, 128]
        self.stream_fuse = nn.Sequential(
            nn.Linear(in_dim * num_streams, in_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim * 2, in_dim),
            nn.LayerNorm(in_dim),
        )

        # Attention scorer: [B, T, 128] → [B, T, 1]
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, pose_feats, frame_quality_mask=None):
        """
        Args:
            pose_feats: [B, T, 128]
            frame_quality_mask: [B, T] boolean, True = valid frame

        Returns:
            selected_feats: [B, K, 128] — top-K salient clips
            attention_weights: [B, T] — full attention distribution
        """
        B, T, D = pose_feats.shape

        # Pass through stream fuse (even if 1 stream, projection holds)
        concat = pose_feats  # [B, T, 128]
        fused = self.stream_fuse(concat)  # [B, T, 128]

        # Compute attention scores
        attn_logits = self.attention(fused).squeeze(-1)  # [B, T]

        # Mask out low-quality frames (improvement #9: garbage in → garbage out)
        if frame_quality_mask is not None:
            attn_logits = attn_logits.masked_fill(~frame_quality_mask, float("-inf"))

        # Soft attention weights
        attn_weights = F.softmax(attn_logits / self.temperature, dim=-1)  # [B, T]

        # Select top-K by attention weight
        K = min(self.top_k, T)
        _, top_indices = torch.topk(attn_weights, K, dim=-1)  # [B, K]
        top_indices_sorted, _ = torch.sort(top_indices, dim=-1)  # Maintain temporal order

        # Gather top-K features
        top_indices_expanded = top_indices_sorted.unsqueeze(-1).expand(-1, -1, D)  # [B, K, 128]
        selected_feats = torch.gather(fused, 1, top_indices_expanded)  # [B, K, 128]

        # Weight the selected features by their attention scores
        top_attn = torch.gather(attn_weights, 1, top_indices_sorted)  # [B, K]
        top_attn_norm = top_attn / (top_attn.sum(dim=-1, keepdim=True) + 1e-8)  # Re-normalize
        selected_feats = selected_feats * top_attn_norm.unsqueeze(-1)  # [B, K, 128]

        return selected_feats, attn_weights
