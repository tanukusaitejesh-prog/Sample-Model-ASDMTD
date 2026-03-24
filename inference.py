"""
Single-Video Inference Script
─────────────────────────────
Runs a video through the full pipeline (all 4 stages) and outputs:
  - Calibrated P(ASD)
  - Risk Level: LOW_RISK / RECHECK / HIGH_RISK / ABSTAIN
  - Confidence score
  - Per-clip breakdown

Supports ensemble mode (averaging across fold models) for improvement #6.

Usage:
    python inference.py --video path/to/video.mp4
    python inference.py --video path/to/video.mp4 --ensemble
    python inference.py --features path/to/features.pt   (pre-extracted)
"""
import os
import sys
import argparse
import glob
import numpy as np
import torch
from collections import defaultdict

import config
from models.pipeline import ASDScreeningPipeline
from models.feature_extractor import ROIExtractor
import torchvision.models as models


def load_ensemble_models(device):
    """Load all fold models for ensemble prediction (improvement #6)."""
    # Check saved_models_2 first
    model_files = sorted(glob.glob(os.path.join(config.SAVED_MODELS_DIR_2, "fold_*.pt")))
    if not model_files:
        model_files = sorted(glob.glob(os.path.join(config.SAVED_MODELS_DIR, "fold_*.pt")))
    
    if not model_files:
        print("❌ No trained models found!")
        sys.exit(1)

    models_list = []
    for mf in model_files:
        checkpoint = torch.load(mf, map_location=device, weights_only=False)
        model = ASDScreeningPipeline().to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        models_list.append(model)
        print(f"  Loaded: {os.path.basename(mf)}")

    return models_list


def extract_features_from_video(video_path, device):
    """Extract features from a single video for inference."""
    from extract_features import load_video_frames, extract_features_from_video as extract_fn

    # Load backbone
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
    backbone.eval().to(device)

    # Initialize ROI extractor
    roi_extractor = ROIExtractor()

    features = extract_fn(video_path, roi_extractor, backbone, None, device)
    roi_extractor.close()

    return features


def compute_frame_level_energy(features, local_window=5):
    """
    Compute per-frame motion energy using a tiny local window.
    This produces a genuinely varying signal unlike flat clip probabilities.
    
    features: [T, 99] array
    Returns: list of {frame, energy, group_energies} dicts
    """
    if features.ndim == 2:
        reshaped = features.reshape(features.shape[0], 33, 3)
    else:
        reshaped = features
    
    T = reshaped.shape[0]
    frame_data = []
    
    for i in range(T):
        start = max(0, i - local_window // 2)
        end = min(T, i + local_window // 2 + 1)
        local_chunk = reshaped[start:end]  # [local_win, 33, 3]
        
        if local_chunk.shape[0] < 2:
            frame_data.append({"frame": i, "energy": 0.0})
            continue
        
        # Frame-to-frame velocity (first derivative)
        velocity = np.diff(local_chunk, axis=0)  # [local_win-1, 33, 3]
        speed = np.linalg.norm(velocity, axis=2)  # [local_win-1, 33]
        
        # Total motion energy for this frame = mean speed across all landmarks
        total_energy = float(np.mean(speed))
        
        # Per-group energy
        groups = {
            "head": list(range(11)),
            "arms": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            "torso": [11, 12, 23, 24],
            "legs": [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        }
        group_energies = {}
        for name, indices in groups.items():
            group_energies[name] = float(np.mean(speed[:, indices]))
        
        frame_data.append({
            "frame": i,
            "energy": total_energy,
            "groups": group_energies
        })
    
    return frame_data


def calculate_motion_signatures(clip_features):
    """
    Calculate motion energy for different body groups.
    clip_features: [win, 33, 3] or [win, 99]
    """
    if clip_features.ndim == 2:
        clip_features = clip_features.reshape(clip_features.shape[0], 33, 3)
    
    # Calculate temporal variance for each landmark
    variances = np.var(clip_features, axis=0)  # [33, 3]
    energies = np.linalg.norm(variances, axis=1)  # [33]
    
    # Body groups (MediaPipe indices)
    groups = {
        "head": list(range(11)),
        "arms": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        "torso": [11, 12, 23, 24],
        "legs": [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    }
    
    signatures = {}
    for name, indices in groups.items():
        # Use MAX energy in group to capture rapid local movement (e.g. wrist flap)
        signatures[name] = float(np.max(energies[indices]))
        
    return signatures


def run_inference(features, model, device):
    """
    Run inference on pre-extracted 3D pose features.
    """
    if features.ndim == 3:
        features = features.reshape(features.shape[0], -1)  # [T, 99]

    T = features.shape[0]
    win = config.WINDOW_SIZE
    stride = config.WINDOW_STRIDE

    clip_probs = []
    temporal_data = []

    # Generate clips via sliding window
    starts = []
    if T < config.MIN_FRAMES:
        starts = [0]
    else:
        for s in range(0, T - win + 1, stride):
            starts.append(s)
        last = T - win
        if last not in starts and last >= 0:
            starts.append(last)

    for start in starts:
        end = min(start + win, T)
        clip_feat = features[start:end]
        pose = torch.tensor(clip_feat, dtype=torch.float32).unsqueeze(0).to(device)

        # Pad if needed
        actual_len = pose.shape[1]
        if actual_len < win:
            pad = win - actual_len
            pose = torch.cat([pose, torch.zeros(1, pad, pose.shape[2]).to(device)], dim=1)

        quality_mask = torch.ones(1, win, dtype=torch.bool).to(device)

        with torch.no_grad():
            _, prob, _ = model(pose, quality_mask)
            p = prob.item()
            clip_probs.append(p)
            
            # Extract motion signature for this clip
            signatures = calculate_motion_signatures(clip_feat)
            
            temporal_data.append({
                "start_frame": int(start),
                "end_frame": int(end),
                "prob": float(p),
                "signatures": signatures
            })

    # Top-percentile aggregation (improvement #3)
    final_prob = ASDScreeningPipeline.aggregate_video_predictions(clip_probs)

    # Risk classification with consistency check (improvements #2, #4)
    risk_level, confidence, details = ASDScreeningPipeline.classify_risk(
        final_prob, clip_probs
    )

    details["n_clips"] = len(clip_probs)
    details["clip_std"] = float(np.std(clip_probs))
    details["temporal_data"] = temporal_data
    details["total_frames"] = T
    
    # Per-frame motion energy for realistic chart
    frame_energies = compute_frame_level_energy(features)
    # Downsample to ~60 points max for chart performance
    step = max(1, len(frame_energies) // 60)
    details["frame_energies"] = frame_energies[::step]

    return final_prob, clip_probs, risk_level, details


def print_result(video_name, final_prob, clip_probs, risk_level, details, ensemble=False):
    """Print a human-readable inference result."""
    prefix = "🔗 ENSEMBLE" if ensemble else "🧠 SINGLE MODEL"

    print(f"\n{'='*60}")
    print(f"  {prefix} RESULT — {video_name}")
    print(f"{'='*60}")

    # Risk level with color-coded emoji
    risk_emoji = {
        "LOW_RISK": "🟢",
        "RECHECK": "🟡",
        "HIGH_RISK": "🔴",
        "ABSTAIN": "⬜",
    }
    print(f"\n  Risk Level:  {risk_emoji.get(risk_level, '  ')} {risk_level}")
    print(f"  P(ASD):      {final_prob:.4f}")
    print(f"  Confidence:  {details['confidence']:.4f}")
    print(f"  Clips:       {details['n_clips']}")
    print(f"  Clip Std:    {details['clip_std']:.4f}")

    if details.get("abstain_reason"):
        print(f"\n  ⚠️  Abstain Reason: {details['abstain_reason']}")

    # Clinical interpretation
    print(f"\n  ─── Clinical Interpretation ───")
    if risk_level == "LOW_RISK":
        print("  Screening suggests typical development.")
        print("  No immediate follow-up recommended.")
    elif risk_level == "RECHECK":
        print("  Result is inconclusive.")
        print("  Recommend rescreening or professional assessment.")
    elif risk_level == "HIGH_RISK":
        print("  Screening suggests elevated ASD indicators.")
        print("  Recommend comprehensive professional evaluation.")
    elif risk_level == "ABSTAIN":
        print("  System confidence is too low for reliable prediction.")
        print("  Recommend manual expert review.")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="ASD Screening — Single Video Inference")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--features", type=str, help="Path to pre-extracted .pt features")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use ensemble of all fold models (improvement #6)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to specific model checkpoint")
    args = parser.parse_args()

    if not args.video and not args.features:
        print("❌ Please provide --video or --features")
        parser.print_help()
        sys.exit(1)

    device = config.DEVICE
    print(f"Device: {device}")

    # Load or extract features
    if args.features:
        print(f"Loading pre-extracted features: {args.features}")
        features = np.load(args.features, allow_pickle=True)
        video_name = os.path.basename(args.features)
    else:
        print(f"Extracting features from video: {args.video}")
        features = extract_features_from_video(args.video, device)
        video_name = os.path.basename(args.video)
        if features is None:
            print("❌ Failed to extract features from video.")
            sys.exit(1)

    print(f"  Frames: {features.shape[0]}")

    if args.ensemble:
        # ─── Ensemble mode (improvement #6) ───
        print("\nLoading ensemble models...")
        models_list = load_ensemble_models(device)

        all_probs = []
        all_clip_probs = []

        for i, model in enumerate(models_list):
            prob, clip_probs, _, _ = run_inference(features, model, device)
            all_probs.append(prob)
            all_clip_probs.extend(clip_probs)

        # Average across models
        ensemble_prob = float(np.mean(all_probs))
        risk_level, confidence, details = ASDScreeningPipeline.classify_risk(
            ensemble_prob, all_clip_probs
        )
        details["n_clips"] = len(all_clip_probs) // len(models_list)
        details["clip_std"] = float(np.std(all_clip_probs))
        details["model_agreement_std"] = float(np.std(all_probs))

        print_result(video_name, ensemble_prob, all_clip_probs, risk_level, details, ensemble=True)

    else:
        # ─── Single model mode ───
        if args.model:
            model_path = args.model
        else:
            # Check saved_models_2 first
            model_files = sorted(glob.glob(os.path.join(config.SAVED_MODELS_DIR_2, "fold_*.pt")))
            if not model_files:
                model_files = sorted(glob.glob(os.path.join(config.SAVED_MODELS_DIR, "fold_*.pt")))
            
            if not model_files:
                print("❌ No trained models found!")
                sys.exit(1)
            model_path = model_files[0]  # Use first fold model

        print(f"Loading model: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model = ASDScreeningPipeline().to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        final_prob, clip_probs, risk_level, details = run_inference(features, model, device)
        print_result(video_name, final_prob, clip_probs, risk_level, details)


if __name__ == "__main__":
    main()
