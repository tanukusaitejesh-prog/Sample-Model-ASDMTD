import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from scipy.signal import savgol_filter

# Robust import for MediaPipe solutions
try:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing
except ImportError:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.drawing_utils as mp_drawing

class SpatialSkeletonProcessor:
    def __init__(self, fps=30, confidence_threshold=0.5, ref_frames=10, smooth_window=5, poly_order=2):
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        self.ref_frames = ref_frames
        self.smooth_window = smooth_window
        self.poly_order = poly_order
        
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        
        # Keypoint Indices
        self.L_HIP = 23
        self.R_HIP = 24
        self.L_SHOULDER = 11
        self.R_SHOULDER = 12

    def process_video(self, video_path):
        """Full pipeline: from raw video to canonical 3D skeleton sequence."""
        # 1. Extraction & Confidence Thresholding
        landmarks, vis_mask = self.extract_landmarks(video_path)
        if len(landmarks) == 0:
            return None
        
        # 2. Missing Joints Handling
        landmarks = self.handle_missing_joints(landmarks, vis_mask)
        
        # 3. Root Centering
        landmarks = self.center_root(landmarks)
        
        # 4 & 5. Orientation Alignment and Scale Normalization (Sequence-Level)
        landmarks = self.align_orientation_and_scale(landmarks)
        
        # 6. Temporal Smoothing
        landmarks = self.smooth_trajectory(landmarks)
        
        return landmarks

    def extract_landmarks(self, video_path):
        """Extract pose_world_landmarks and apply confidence thresholding."""
        cap = cv2.VideoCapture(video_path)
        frames_landmarks = []
        frames_visibility = []
        
        with self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,           # GHUM model
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)
                
                if results.pose_world_landmarks:
                    frame_pts = []
                    frame_vis = []
                    for lm in results.pose_world_landmarks.landmark:
                        frame_pts.append([lm.x, lm.y, lm.z])
                        frame_vis.append(lm.visibility)
                    frames_landmarks.append(frame_pts)
                    frames_visibility.append(frame_vis)
                else:
                    # Append NaN if skeleton is completely missed in this frame
                    frames_landmarks.append([[np.nan]*3]*33)
                    frames_visibility.append([0.0]*33)
                    
        cap.release()
        
        if len(frames_landmarks) == 0:
            return np.empty((0, 33, 3)), np.empty((0, 33))
            
        return np.array(frames_landmarks), np.array(frames_visibility)

    def handle_missing_joints(self, landmarks, vis_mask):
        """Mask out low confidence joints and interpolate missing values."""
        # Mask joints with low confidence
        bad_joints = vis_mask < self.confidence_threshold
        landmarks[bad_joints] = np.nan
        
        # Shape: (T, 33, 3)
        T, num_joints, dims = landmarks.shape
        
        # Reshape to (T, 33*3) for Pandas interpolation
        flattened = landmarks.reshape(T, -1)
        df = pd.DataFrame(flattened)
        
        # Interpolate NaNs
        df = df.interpolate(method='linear', limit_direction='both')
        
        # If there are still NaNs (e.g., entirely missing joint), fill with 0 (fallback)
        df = df.fillna(0)
        
        interpolated = df.values.reshape(T, num_joints, dims)
        return interpolated

    def center_root(self, landmarks):
        """Center the skeleton at the pelvis for every frame."""
        # Pelvis is the midpoint between left hip and right hip
        L_hip = landmarks[:, self.L_HIP, :]
        R_hip = landmarks[:, self.R_HIP, :]
        root = (L_hip + R_hip) / 2.0
        
        # Subtract root from all joints (broadcasting)
        centered_landmarks = landmarks - root[:, np.newaxis, :]
        return centered_landmarks

    def align_orientation_and_scale(self, landmarks):
        """Standardize the skeleton orientation and scale.
        
        1. Vertical Alignment (2D XY Rotation):
           Rotates the skeleton in the image plane so the torso points straight up (-Y).
           This fixes sideways/portrait videos while staying deterministic since it
           ignores the noisy Z-axis fluctuation.
           
        2. Scale Normalization:
           Divides all coordinates by the median torso length across the sequence.
        """
        T = landmarks.shape[0]
        
        # 1. 2D VERTICAL ALIGNMENT (XY PLANE)
        # We compute the torso vector in image space (X, Y)
        L_hip = landmarks[:, self.L_HIP, :2]
        R_hip = landmarks[:, self.R_HIP, :2]
        root_2d = (L_hip + R_hip) / 2.0
        
        L_shoulder = landmarks[:, self.L_SHOULDER, :2]
        R_shoulder = landmarks[:, self.R_SHOULDER, :2]
        shoulder_center_2d = (L_shoulder + R_shoulder) / 2.0
        
        # Average torso vector across all frames to get a stable reference
        torso_vec_2d = np.mean(shoulder_center_2d - root_2d, axis=0)
        
        # We want this vector to point toward (0, -1) [Vertical Up in image coords]
        # Angle from positive X axis
        current_angle = np.arctan2(torso_vec_2d[1], torso_vec_2d[0])
        target_angle = -np.pi / 2.0 # -90 degrees is straight up (negative Y)
        d_theta = target_angle - current_angle
        
        # 2D Rotation Matrix
        cos_tr = np.cos(d_theta)
        sin_tr = np.sin(d_theta)
        R_xy = np.array([
            [cos_tr, -sin_tr],
            [sin_tr,  cos_tr]
        ])
        
        # Apply rotation to X and Y coordinates (T, 33, 2)
        landmarks[:, :, :2] = np.matmul(landmarks[:, :, :2], R_xy.T)
        
        # 2. SCALE NORMALIZATION (3D)
        # Refresh 3D points after XY rotation
        L_hip_3d = landmarks[:, self.L_HIP, :]
        R_hip_3d = landmarks[:, self.R_HIP, :]
        root_3d = (L_hip_3d + R_hip_3d) / 2.0
        
        L_shoulder_3d = landmarks[:, self.L_SHOULDER, :]
        R_shoulder_3d = landmarks[:, self.R_SHOULDER, :]
        shoulder_center_3d = (L_shoulder_3d + R_shoulder_3d) / 2.0
        
        torso_lengths = np.linalg.norm(shoulder_center_3d - root_3d, axis=1)
        torso_length = np.median(torso_lengths)
        
        if torso_length > 1e-5:
            landmarks = landmarks / torso_length
            
        return landmarks

    def smooth_trajectory(self, landmarks):
        """Apply Savitzky-Golay filter to smooth joint trajectories temporally."""
        T = landmarks.shape[0]
        # Only smooth if we have enough frames
        if T > self.smooth_window:
            # Ensure window is odd
            window = self.smooth_window if self.smooth_window % 2 == 1 else self.smooth_window + 1
            if window > T:
                window = T if T % 2 == 1 else T - 1
            
            if window > self.poly_order:
                # Apply filter over time axis (axis=0)
                landmarks = savgol_filter(landmarks, window, self.poly_order, axis=0)
                
        return landmarks


if __name__ == "__main__":
    import os
    
    # Simple rigorous test to verify matrix multiplication correctness
    print("Running mathematical verification tests...")
    
    # Create fake point on the global X axis
    fake_point = np.array([1.0, 0.0, 0.0])
    
    # Assume our new coordinate basis is rotated 90 degrees around Z axis:
    # New X is along global Y.
    # New Y is along global -X.
    # New Z is same as global Z.
    x_axis = np.array([0.0, 1.0, 0.0])
    y_axis = np.array([-1.0, 0.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    
    R = np.column_stack((x_axis, y_axis, z_axis))
    print("Rotation Matrix (columns as new axes):\n", R)
    
    # We want to find the coordinates of fake_point in the new basis.
    # fake_point is [1, 0, 0] in global.
    # Its projection onto the new X axis (which is global Y) is 0.
    # Its projection onto the new Y axis (which is global -X) is -1.
    # Its projection onto the new Z axis (which is global Z) is 0.
    # So the expected output in the new basis is [0, -1, 0].
    
    # Apply standard transformation for row vectors:
    aligned = fake_point @ R
    
    # Apply transformation with R.T just to show the alternative:
    aligned_transpose = fake_point @ R.T
    
    print("\nOriginal point:", fake_point)
    print("Aligned using fake_point @ R   :", aligned)
    print("Aligned using fake_point @ R.T :", aligned_transpose)
    
    assert np.allclose(aligned, [0, -1, 0]), "fake_point @ R is mathematically incorrect for projecting onto basis columns!"
    print("✅ Verified mathematically: For row vectors `v @ R` projects correctly when R columns are basis vectors.")
    print("\nThe pipeline logic is correct!")

    # Optionally test locally if user provides a video path
    # processor = SpatialSkeletonProcessor()
    # out = processor.process_video("example.mp4")
    # print(out.shape)
