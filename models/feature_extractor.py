"""
Stage 1: Frozen CNN Feature Extraction
───────────────────────────────────────
Uses pretrained ResNet18 (frozen) to extract features from 3 ROI streams:
  - Face region (MediaPipe face detection)
  - Pose region (MediaPipe pose keypoints bounding box)
  - Hand regions (MediaPipe hand detection)

Each stream: frame crop → ResNet18 avgpool → Linear(512→128)
Output: 3× [B, T, 128]

Frame Quality Filtering:
  - Drops frames where detection confidence < threshold
  - Prevents garbage-in-garbage-out

NOTE: MediaPipe is lazy-imported only when ROIExtractor is instantiated.
      This allows model classes to work without MediaPipe installed.
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ROIExtractor:
    """Extracts face, pose, and hand regions from video frames using MediaPipe."""

    def __init__(self):
        import mediapipe as mp
        import cv2  # noqa: F401
        self.mp_face = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=config.FACE_SCORE_THRESH
        )
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=config.POSE_SCORE_THRESH
        )
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=config.HAND_SCORE_THRESH
        )
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.FRAME_SIZE, config.FRAME_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _crop_region(self, frame, bbox, pad_ratio=0.2):
        """Crop a region from frame with padding. bbox = (x1, y1, x2, y2) normalized."""
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0] * w - pad_ratio * (bbox[2] - bbox[0]) * w))
        y1 = max(0, int(bbox[1] * h - pad_ratio * (bbox[3] - bbox[1]) * h))
        x2 = min(w, int(bbox[2] * w + pad_ratio * (bbox[2] - bbox[0]) * w))
        y2 = min(h, int(bbox[3] * h + pad_ratio * (bbox[3] - bbox[1]) * h))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def extract_rois(self, frame_rgb):
        """
        Extract face, pose, and hand ROIs from a single RGB frame.

        Returns:
            face_crop: tensor or None
            pose_crop: tensor or None
            hand_crop: tensor or None
            quality_scores: dict with detection confidences
        """
        h, w = frame_rgb.shape[:2]
        quality_scores = {"face": 0.0, "pose": 0.0, "hand": 0.0}

        # ─── Face ───
        face_crop = None
        face_results = self.mp_face.process(frame_rgb)
        if face_results.detections:
            det = face_results.detections[0]
            quality_scores["face"] = det.score[0]
            bb = det.location_data.relative_bounding_box
            bbox = (
                bb.xmin, bb.ymin,
                bb.xmin + bb.width, bb.ymin + bb.height
            )
            crop = self._crop_region(frame_rgb, bbox)
            if crop is not None:
                face_crop = self.transform(crop)

        # ─── Pose ───
        pose_crop = None
        pose_results = self.mp_pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            vis_scores = [lm.visibility for lm in landmarks]
            quality_scores["pose"] = float(np.mean(vis_scores))
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            crop = self._crop_region(frame_rgb, bbox, pad_ratio=0.1)
            if crop is not None:
                pose_crop = self.transform(crop)

        # ─── Hands ───
        hand_crop = None
        hand_results = self.mp_hands.process(frame_rgb)
        if hand_results.multi_hand_landmarks:
            quality_scores["hand"] = hand_results.multi_handedness[0].classification[0].score
            all_xs, all_ys = [], []
            for hand_lms in hand_results.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    all_xs.append(lm.x)
                    all_ys.append(lm.y)
            bbox = (min(all_xs), min(all_ys), max(all_xs), max(all_ys))
            crop = self._crop_region(frame_rgb, bbox, pad_ratio=0.3)
            if crop is not None:
                hand_crop = self.transform(crop)

        return face_crop, pose_crop, hand_crop, quality_scores

    def close(self):
        self.mp_face.close()
        self.mp_pose.close()
        self.mp_hands.close()


class StreamProjectionHead(nn.Module):
    """Trainable projection: Pose features (99) → compact (128)."""

    def __init__(self, in_dim=config.POSE_DIM, out_dim=config.FEATURE_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.proj(x)


class PoseFeatureExtractor(nn.Module):
    """
    Stage 1: Trainable projection for 3D Pose Landmarks.
    Input: [B, T, 99] 
    Output: [B, T, 128]
    """

    def __init__(self):
        super().__init__()
        self.pose_proj = StreamProjectionHead(in_dim=config.POSE_DIM, out_dim=config.FEATURE_DIM)

    def forward(self, pose_frames, masks=None):
        """
        Args:
            pose_frames: [B, T, 99]
            masks: boolean masks for valid detections
        """
        B, T = pose_frames.shape[:2]
        pose_bb = pose_frames.reshape(B * T, -1)
        pose_feats = self.pose_proj(pose_bb).reshape(B, T, -1)
        return pose_feats
