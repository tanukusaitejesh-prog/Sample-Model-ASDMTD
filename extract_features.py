"""
Feature Pre-Extraction Script
──────────────────────────────
Processes all videos through MediaPipe (ROI detection) and frozen ResNet18
(backbone feature extraction), then caches results as .pt files.

This is run ONCE before training to avoid redundant CNN computation.

Usage:
    python extract_features.py
    python extract_features.py --test-mode   (generates synthetic data for testing)
"""
import os
import sys
import argparse
import glob
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

import config
from models.feature_extractor import ROIExtractor


def load_video_frames(video_path, target_fps=config.FPS_TARGET):
    """
    Load and downsample video frames.

    Returns:
        frames: list of RGB numpy arrays [H, W, 3]
        original_fps: float
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARNING] Cannot open: {video_path}")
        return [], 0

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps <= 0:
        original_fps = 30.0

    frame_interval = max(1, int(round(original_fps / target_fps)))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        frame_idx += 1

    cap.release()
    return frames, original_fps


def extract_features_from_video(video_path, roi_extractor, backbone, transform_fn, device):
    """
    Extract face/pose/hand backbone features from a single video.

    Returns:
        dict with keys: face, pose, hand, quality — each [T, dim]
    """
    frames, fps = load_video_frames(video_path)
    if len(frames) == 0:
        return None

    face_feats = []
    pose_feats = []
    hand_feats = []
    quality_scores = []

    # Default input for missing detections
    default_input = torch.zeros(1, 3, config.FRAME_SIZE, config.FRAME_SIZE).to(device)
    with torch.no_grad():
        default_bb = backbone(default_input).flatten(1)  # [1, 512]

    for frame in tqdm(frames, desc="  Frames", leave=False):
        face_crop, pose_crop, hand_crop, scores = roi_extractor.extract_rois(frame)

        quality_scores.append([scores["face"], scores["pose"], scores["hand"]])

        # Extract backbone features for each stream
        with torch.no_grad():
            if face_crop is not None:
                face_bb = backbone(face_crop.unsqueeze(0).to(device)).flatten(1)
            else:
                face_bb = default_bb

            if pose_crop is not None:
                pose_bb = backbone(pose_crop.unsqueeze(0).to(device)).flatten(1)
            else:
                pose_bb = default_bb

            if hand_crop is not None:
                hand_bb = backbone(hand_crop.unsqueeze(0).to(device)).flatten(1)
            else:
                hand_bb = default_bb

        face_feats.append(face_bb.cpu())
        pose_feats.append(pose_bb.cpu())
        hand_feats.append(hand_bb.cpu())

    if len(face_feats) == 0:
        return None

    result = {
        "face": torch.cat(face_feats, dim=0),     # [T, 512]
        "pose": torch.cat(pose_feats, dim=0),      # [T, 512]
        "hand": torch.cat(hand_feats, dim=0),      # [T, 512]
        "quality": torch.tensor(quality_scores),    # [T, 3]
    }
    return result


def generate_synthetic_data(n_asd=4, n_td=4, T=80):
    """
    Generate synthetic feature data for testing the pipeline.
    Creates fake .pt files with random features.
    """
    print(f"\n🧪 Generating synthetic data: {n_asd} ASD + {n_td} TD videos...")

    for label, subdir, n in [("asd", "asd", n_asd), ("td", "td", n_td)]:
        out_dir = os.path.join(config.FEATURE_DIR, subdir)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(n):
            # Simulate different video lengths
            t = T + np.random.randint(-20, 20)

            # ASD videos have slightly different feature distributions
            if label == "asd":
                face = torch.randn(t, 512) * 0.8 + 0.2
                pose = torch.randn(t, 512) * 1.0 + 0.3
                hand = torch.randn(t, 512) * 0.9 + 0.1
            else:
                face = torch.randn(t, 512) * 0.8 - 0.1
                pose = torch.randn(t, 512) * 1.0 - 0.2
                hand = torch.randn(t, 512) * 0.9 - 0.1

            quality = torch.rand(t, 3) * 0.5 + 0.5  # Decent quality scores

            data = {"face": face, "pose": pose, "hand": hand, "quality": quality}
            save_path = os.path.join(out_dir, f"synthetic_{label}_{i:03d}.pt")
            torch.save(data, save_path)
            print(f"  Created: {save_path} [{t} frames]")

    print("✅ Synthetic data generated!\n")


def main():
    parser = argparse.ArgumentParser(description="Extract features from videos")
    parser.add_argument("--test-mode", action="store_true",
                        help="Generate synthetic data for pipeline testing")
    parser.add_argument("--video-dir", default=None,
                        help="Override video directory (default: config.DATA_DIR)")
    args = parser.parse_args()

    if args.test_mode:
        generate_synthetic_data()
        return

    video_dir = args.video_dir or config.DATA_DIR
    device = config.DEVICE
    print(f"Device: {device}")

    # Load frozen ResNet backbone
    print("Loading ResNet18 backbone...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(*list(resnet.children())[:-1])
    backbone.eval().to(device)

    # Initialize ROI extractor
    print("Initializing MediaPipe ROI extractor...")
    roi_extractor = ROIExtractor()

    # Process each class directory
    for label, subdir in [("ASD", "asd"), ("TD", "td")]:
        input_dir = os.path.join(video_dir, subdir)
        output_dir = os.path.join(config.FEATURE_DIR, subdir)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(input_dir):
            print(f"⚠️  Directory not found: {input_dir}")
            continue

        video_files = []
        for ext in config.VIDEO_EXTENSIONS:
            video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))

        print(f"\n{'='*60}")
        print(f"Processing {len(video_files)} {label} videos from: {input_dir}")
        print(f"{'='*60}")

        for video_path in tqdm(video_files, desc=f"{label} videos"):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(output_dir, f"{video_name}.pt")

            if os.path.exists(save_path):
                print(f"  [SKIP] Already exists: {save_path}")
                continue

            print(f"\n  Processing: {video_name}")
            features = extract_features_from_video(
                video_path, roi_extractor, backbone, None, device
            )

            if features is not None:
                torch.save(features, save_path)
                T = features["face"].shape[0]
                print(f"  ✅ Saved: {save_path} [{T} frames]")
            else:
                print(f"  ❌ Failed: {video_path}")

    roi_extractor.close()
    print("\n🎉 Feature extraction complete!")


if __name__ == "__main__":
    main()
