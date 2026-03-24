"""
Dataset & Data Loading
──────────────────────
- Sliding window clips from pre-extracted features
- TD oversampling for class balance
- Feature-level augmentation (no raw video augmentation needed)
- Returns: (pose_feat, quality_mask, label, video_id)
"""
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import config


class ASDFeatureDataset(Dataset):
    """
    Dataset for pre-extracted features.

    Each video is stored as a .npy array: [T, 33, 3]

    Sliding window + augmentation happen here.
    """

    def __init__(self, feature_files, labels, video_ids, augment=False):
        """
        Args:
            feature_files: list of paths to .npy feature files
            labels: list of int (0=TD, 1=ASD)
            video_ids: list of str (video identifiers)
            augment: whether to apply augmentation
        """
        self.augment = augment
        self.clips = []  # List of (file_path, start_idx, label, video_id)

        for feat_file, label, vid_id in zip(feature_files, labels, video_ids):
            feats = np.load(feat_file, allow_pickle=True)
            T = feats.shape[0]

            # Generate sliding windows
            stride = config.WINDOW_STRIDE
            win = config.WINDOW_SIZE

            if T < config.MIN_FRAMES:
                # Video too short — use all frames, pad later
                self.clips.append((feat_file, 0, T, label, vid_id))
            else:
                for start in range(0, T - win + 1, stride):
                    self.clips.append((feat_file, start, win, label, vid_id))
                # Also add a clip ending at T if not already covered
                last_start = T - win
                if last_start >= 0 and last_start % stride != 0:
                    self.clips.append((feat_file, last_start, win, label, vid_id))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        feat_file, start, length, label, video_id = self.clips[idx]
        feats = np.load(feat_file, allow_pickle=True)

        T_total = feats.shape[0]
        win = config.WINDOW_SIZE

        # Apply temporal jitter (augmentation)
        if self.augment:
            jitter = random.randint(-config.AUG_TEMPORAL_JITTER, config.AUG_TEMPORAL_JITTER)
            start = max(0, min(start + jitter, T_total - length))

        end = min(start + length, T_total)

        pose = torch.tensor(feats[start:end], dtype=torch.float32)
        pose = pose.view(pose.shape[0], -1)  # Flatten to [T, 99]

        # Pad if necessary to reach window size
        actual_len = pose.shape[0]
        if actual_len < win:
            pad = win - actual_len
            pose = torch.cat([pose, torch.zeros(pad, pose.shape[1])], dim=0)

        quality_mask = torch.ones(win, dtype=torch.bool)

        # ─── Feature-level augmentation ───
        if self.augment:
            # Gaussian noise
            pose = pose + torch.randn_like(pose) * config.AUG_GAUSSIAN_NOISE_STD

            # Random feature dropout
            if random.random() < config.AUG_FEATURE_DROPOUT:
                drop_mask = torch.bernoulli(
                    torch.ones_like(pose) * (1 - config.AUG_FEATURE_DROPOUT)
                )
                pose = pose * drop_mask

            # Horizontal (left-right) flip: mirror x-coords and swap L/R pairs
            if random.random() < config.AUG_FLIP_PROB:
                # pose is [T, 99] = [T, 33*3], reshape to [T, 33, 3] for manipulation
                pose_3d = pose.view(win, 33, 3)
                # Mirror x-coordinates (col 0): since coordinates are root-centered at 0, we just negate X.
                pose_3d[:, :, 0] = -pose_3d[:, :, 0]
                # Swap left/right landmark pairs (MediaPipe indices)
                # Shoulders: 11↔12, Elbows: 13↔14, Wrists: 15↔16
                # Hips: 23↔24, Knees: 25↔26, Ankles: 27↔28
                # Pinky: 17↔18, Index: 19↔20, Thumb: 21↔22
                # Foot: 29↔30, Heel: 31↔32
                # Eyes: 2↔5, Ears: 7↔8, Mouth: 9↔10
                lr_pairs = [
                    (2, 5), (7, 8), (9, 10),
                    (11, 12), (13, 14), (15, 16),
                    (17, 18), (19, 20), (21, 22),
                    (23, 24), (25, 26), (27, 28),
                    (29, 30), (31, 32),
                ]
                for l, r in lr_pairs:
                    pose_3d[:, l, :], pose_3d[:, r, :] = (
                        pose_3d[:, r, :].clone(), pose_3d[:, l, :].clone()
                    )
                pose = pose_3d.view(win, -1)

        return {
            "pose": pose,        # [T, 99]
            "quality_mask": quality_mask,  # [T]
            "label": torch.tensor(label, dtype=torch.float32),
            "video_id": video_id,
        }


def get_feature_files_and_labels():
    """
    Scan feature directory for extracted .npy files.

    Expected structure:
        features/
            asd/
                video1.npy
                video2.npy
            td/
                video1.npy
                video2.npy

    Returns:
        files: list of file paths
        labels: list of int (0=TD, 1=ASD)
        video_ids: list of str
    """
    files, labels, video_ids = [], [], []

    # Load from the new straightened feature directory
    asd_dir = os.path.join(config.FEATURE_DIR_2, "asd")
    td_dir = os.path.join(config.FEATURE_DIR_2, "td")

    if os.path.exists(asd_dir):
        for f in sorted(glob.glob(os.path.join(asd_dir, "*.npy"))):
            files.append(f)
            labels.append(1)
            video_ids.append(os.path.splitext(os.path.basename(f))[0])

    if os.path.exists(td_dir):
        for f in sorted(glob.glob(os.path.join(td_dir, "*.npy"))):
            files.append(f)
            labels.append(0)
            video_ids.append(os.path.splitext(os.path.basename(f))[0])

    return files, labels, video_ids


def create_balanced_sampler(labels):
    """
    Create a WeightedRandomSampler for TD oversampling.
    Ensures balanced batches even with imbalanced classes.
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(weights),
        replacement=True,
    )


def build_dataloader(feature_files, labels, video_ids, augment=False, shuffle=True):
    """Build a DataLoader from feature files."""
    dataset = ASDFeatureDataset(feature_files, labels, video_ids, augment=augment)

    if augment:
        # Use balanced sampler for training
        clip_labels = [c[3] for c in dataset.clips]
        sampler = create_balanced_sampler(clip_labels)
        return DataLoader(
            dataset, batch_size=config.BATCH_SIZE,
            sampler=sampler, num_workers=0,
            drop_last=True, pin_memory=True,
        )
    else:
        return DataLoader(
            dataset, batch_size=config.BATCH_SIZE,
            shuffle=shuffle, num_workers=0,
            drop_last=False, pin_memory=True,
        )
