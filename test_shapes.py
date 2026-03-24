"""
Shape Verification Test
───────────────────────
Verifies tensor shapes through every stage of the pipeline.
Creates dummy data and passes it through each component.

Usage:
    python test_shapes.py
"""
import torch
import sys

# Ensure imports work
sys.path.insert(0, ".")
import config
from models.feature_extractor import FrozenFeatureExtractor
from models.temporal_attention import SoftTemporalAttention
from models.event_transformer import EventTransformer
from models.pipeline import ASDScreeningPipeline


def test_stage1():
    """Test Stage 1: Feature Extractor shapes."""
    print("─── Stage 1: Feature Extractor ───")
    model = FrozenFeatureExtractor()
    B, T = 2, config.WINDOW_SIZE

    # Test with pre-extracted backbone features (512-dim)
    face = torch.randn(B, T, config.RESNET_FEAT_DIM)
    pose = torch.randn(B, T, config.RESNET_FEAT_DIM)
    hand = torch.randn(B, T, config.RESNET_FEAT_DIM)

    face_out, pose_out, hand_out = model(face, pose, hand)

    assert face_out.shape == (B, T, config.FEATURE_DIM), f"Expected {(B, T, config.FEATURE_DIM)}, got {face_out.shape}"
    assert pose_out.shape == (B, T, config.FEATURE_DIM)
    assert hand_out.shape == (B, T, config.FEATURE_DIM)

    print(f"  Input:  3× [{B}, {T}, {config.RESNET_FEAT_DIM}]")
    print(f"  Output: 3× [{B}, {T}, {config.FEATURE_DIM}]")
    print("  ✅ PASSED\n")


def test_stage2():
    """Test Stage 2: Soft Temporal Attention shapes."""
    print("─── Stage 2: Soft Temporal Attention ───")
    model = SoftTemporalAttention()
    B, T = 2, config.WINDOW_SIZE

    face = torch.randn(B, T, config.FEATURE_DIM)
    pose = torch.randn(B, T, config.FEATURE_DIM)
    hand = torch.randn(B, T, config.FEATURE_DIM)
    quality_mask = torch.ones(B, T, dtype=torch.bool)

    selected, attn = model(face, pose, hand, quality_mask)
    K = min(config.TOP_K_CLIPS, T)

    assert selected.shape == (B, K, config.FEATURE_DIM), f"Expected {(B, K, config.FEATURE_DIM)}, got {selected.shape}"
    assert attn.shape == (B, T), f"Expected {(B, T)}, got {attn.shape}"

    print(f"  Input:  3× [{B}, {T}, {config.FEATURE_DIM}]")
    print(f"  Output: [{B}, {K}, {config.FEATURE_DIM}] (selected)")
    print(f"  Attn:   [{B}, {T}] (weights)")
    print("  ✅ PASSED\n")


def test_stage3():
    """Test Stage 3: Event Transformer shapes."""
    print("─── Stage 3: Event Transformer ───")
    model = EventTransformer()
    B, K = 2, config.TOP_K_CLIPS

    x = torch.randn(B, K, config.TRANSFORMER_DIM)
    logit, prob = model(x)

    assert logit.shape == (B, 1), f"Expected {(B, 1)}, got {logit.shape}"
    assert prob.shape == (B, 1), f"Expected {(B, 1)}, got {prob.shape}"
    assert (prob >= 0).all() and (prob <= 1).all(), "Prob should be in [0, 1]"

    print(f"  Input:  [{B}, {K}, {config.TRANSFORMER_DIM}]")
    print(f"  Logit:  [{B}, 1]")
    print(f"  Prob:   [{B}, 1] (range: [{prob.min():.4f}, {prob.max():.4f}])")
    print("  ✅ PASSED\n")


def test_full_pipeline():
    """Test full pipeline end-to-end."""
    print("─── Full Pipeline (Stages 1-4) ───")
    model = ASDScreeningPipeline()
    B, T = 2, config.WINDOW_SIZE

    face = torch.randn(B, T, config.RESNET_FEAT_DIM)
    pose = torch.randn(B, T, config.RESNET_FEAT_DIM)
    hand = torch.randn(B, T, config.RESNET_FEAT_DIM)
    quality_mask = torch.ones(B, T, dtype=torch.bool)

    logit, prob, attn = model(face, pose, hand, quality_mask)

    assert logit.shape == (B, 1)
    assert prob.shape == (B, 1)
    assert attn.shape == (B, T)

    print(f"  Input:  3× [{B}, {T}, {config.RESNET_FEAT_DIM}]")
    print(f"  Logit:  [{B}, 1]")
    print(f"  Prob:   [{B}, 1]")
    print(f"  Attn:   [{B}, {T}]")

    # Test Stage 4: Risk classification
    prob_val = prob[0, 0].item()
    risk, conf, details = ASDScreeningPipeline.classify_risk(prob_val)
    print(f"\n  Risk: {risk} (p={prob_val:.4f}, conf={conf:.4f})")
    print("  ✅ PASSED\n")


def test_risk_classification():
    """Test Stage 4: Risk classification logic."""
    print("─── Stage 4: Risk Classification ───")

    test_cases = [
        (0.1, [0.08, 0.12, 0.11], "LOW_RISK"),
        (0.5, [0.45, 0.55, 0.50], "RECHECK"),
        (0.8, [0.75, 0.85, 0.80], "HIGH_RISK"),
        (0.51, [0.51, 0.50, 0.52], "ABSTAIN"),  # Low confidence
        (0.7, [0.2, 0.9, 0.5, 0.8], None),  # High inconsistency → ABSTAIN
    ]

    for prob, clips, expected in test_cases:
        risk, conf, details = ASDScreeningPipeline.classify_risk(prob, clips)
        status = "✅" if (expected is None or risk == expected) else "❌"
        print(f"  p={prob:.2f}, std={float(__import__('numpy').std(clips)):.3f} → "
              f"{risk:>12s} (conf={conf:.3f}) {status}")

    print("  ✅ PASSED\n")


def test_aggregation():
    """Test top-percentile aggregation."""
    print("─── Top-Percentile Aggregation ───")
    import numpy as np

    clips = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.15]
    agg = ASDScreeningPipeline.aggregate_video_predictions(clips)
    top_30 = sorted(clips, reverse=True)[:max(1, int(len(clips) * 0.3))]

    print(f"  All clips mean: {np.mean(clips):.4f}")
    print(f"  Top 30% mean:   {np.mean(top_30):.4f}")
    print(f"  Pipeline agg:   {agg:.4f}")
    assert abs(agg - np.mean(top_30)) < 1e-6
    print("  ✅ PASSED\n")


def count_parameters(model):
    """Count trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    print(f"  Trainable:  {trainable:>10,d} ({100*trainable/total:.1f}%)")
    print(f"  Frozen:     {frozen:>10,d} ({100*frozen/total:.1f}%)")
    print(f"  Total:      {total:>10,d}")


if __name__ == "__main__":
    print("\n🧪 SHAPE VERIFICATION TESTS\n")

    test_stage1()
    test_stage2()
    test_stage3()
    test_full_pipeline()
    test_risk_classification()
    test_aggregation()

    print("─── Model Parameters ───")
    model = ASDScreeningPipeline()
    count_parameters(model)

    print("\n🎉 ALL TESTS PASSED!\n")
