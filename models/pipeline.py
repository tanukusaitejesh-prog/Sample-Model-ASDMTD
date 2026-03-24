"""
Full ASD Screening Pipeline — Stages 1-4 Combined
──────────────────────────────────────────────────
Combines all stages into one model and adds:
  - Stage 4: Calibrated risk output (Low/Recheck/High/ABSTAIN)
  - Top-percentile aggregation (improvement #3)
  - Consistency check across clips (improvement #4)
  - Quality filter with ABSTAIN (improvement #2)
"""
import torch
import torch.nn as nn
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.feature_extractor import PoseFeatureExtractor
from models.temporal_attention import SoftTemporalAttention
from models.event_transformer import EventTransformer


class ASDScreeningPipeline(nn.Module):
    """
    Complete pipeline: features → temporal attention → transformer → risk.

    Operates on pre-extracted backbone features (512-dim) for efficiency.
    The frozen ResNet backbone is run separately in extract_features.py.
    """

    def __init__(self):
        super().__init__()

        # Stage 1: Trainable projection heads (backbone features already extracted)
        self.feature_extractor = PoseFeatureExtractor()

        # Stage 2: Soft temporal attention
        self.temporal_attention = SoftTemporalAttention()

        # Stage 3: Event transformer
        self.event_transformer = EventTransformer()

    def forward(self, pose_feats, frame_quality_mask=None):
        """
        Forward pass through stages 1-3.

        Args:
            pose_feats: [B, T, 99] — 3D landmarks
            frame_quality_mask: [B, T] — True = valid frame

        Returns:
            logit: [B, 1] — raw logit
            prob: [B, 1] — calibrated probability
            attn_weights: [B, T] — attention weights
        """
        # Stage 1: Project features
        pose_proj = self.feature_extractor(pose_feats, frame_quality_mask)

        # Stage 2: Soft temporal selection
        selected_feats, attn_weights = self.temporal_attention(
            pose_proj,
            frame_quality_mask=frame_quality_mask
        )

        # Stage 3: Event transformer
        logit, prob = self.event_transformer(selected_feats)

        return logit, prob, attn_weights

    @staticmethod
    def classify_risk(prob, clip_probs=None):
        """
        Stage 4: Calibrated Risk Output.

        Args:
            prob: float — calibrated P(ASD) for a single video
            clip_probs: list of floats — per-clip probabilities for consistency check

        Returns:
            risk_level: str — "LOW_RISK", "RECHECK", "HIGH_RISK", or "ABSTAIN"
            confidence: float — prediction confidence
            details: dict — additional info
        """
        confidence = abs(prob - 0.5) * 2.0  # 0 = uncertain, 1 = certain
        details = {
            "probability": prob,
            "confidence": confidence,
            "risk_level": None,
            "abstain_reason": None,
        }

        # ─── Quality filter: ABSTAIN if low confidence ───
        if confidence < config.ABSTAIN_CONFIDENCE:
            details["risk_level"] = "ABSTAIN"
            details["abstain_reason"] = f"Low confidence ({confidence:.3f} < {config.ABSTAIN_CONFIDENCE})"
            return "ABSTAIN", confidence, details

        # ─── Consistency check across clips (std-based) ───
        if clip_probs is not None and len(clip_probs) > 1:
            clip_std = float(np.std(clip_probs))
            details["clip_std"] = clip_std
            if clip_std > config.CONSISTENCY_STD_THRESH:
                details["risk_level"] = "ABSTAIN"
                details["abstain_reason"] = f"High inconsistency (std={clip_std:.3f} > {config.CONSISTENCY_STD_THRESH})"
                return "ABSTAIN", confidence, details

        # ─── Risk classification (uses DECISION_THRESHOLD, not 0.5) ───
        if prob < config.TAU_LOW:
            details["risk_level"] = "LOW_RISK"
            return "LOW_RISK", confidence, details
        elif prob > config.TAU_HIGH:
            details["risk_level"] = "HIGH_RISK"
            return "HIGH_RISK", confidence, details
        else:
            details["risk_level"] = "RECHECK"
            return "RECHECK", confidence, details

    @staticmethod
    def aggregate_video_predictions(clip_probs):
        """
        Trimmed mean aggregation with consistency check.

        1. Sort clip probabilities
        2. Drop bottom 10% and top 10% (trimmed mean)
        3. Check clip consistency: if fewer than 30% of clips exceed
           the decision threshold, dampen the score

        Args:
            clip_probs: list or numpy array of per-clip probabilities

        Returns:
            aggregated_prob: float — final video-level probability
        """
        clip_probs = np.array(clip_probs)
        n = len(clip_probs)

        if n <= 2:
            # Too few clips for trimming — just use mean
            score = float(np.mean(clip_probs))
        else:
            sorted_probs = np.sort(clip_probs)
            lower = max(1, int(config.TRIMMED_MEAN_LOW * n))
            upper = min(n - 1, int(config.TRIMMED_MEAN_HIGH * n))
            if upper <= lower:
                upper = lower + 1
            trimmed = sorted_probs[lower:upper]
            score = float(np.mean(trimmed))

        # ─── Clip consistency check ───
        # If fewer than CONSISTENCY_RATIO_THRESH of clips agree on ASD,
        # dampen the score to reduce false positive spikes
        asd_ratio = float(np.mean(clip_probs > config.DECISION_THRESHOLD))
        if asd_ratio < config.CONSISTENCY_RATIO_THRESH:
            score *= config.CONSISTENCY_DAMPEN

        return score
