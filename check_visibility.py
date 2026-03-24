import numpy as np
from spatial_processor import SpatialSkeletonProcessor
import warnings
warnings.filterwarnings('ignore')

proc = SpatialSkeletonProcessor()
video = r"C:\Users\saite\OneDrive\Desktop\OS\ezgif-3d0f47c951f3ae3e.mp4"

landmarks, vis_mask = proc.extract_landmarks(video)
print(f"Total Frames: {landmarks.shape[0]}")
# Print raw visibility array for a few frames
print(f"Vis Mask Mean: {vis_mask.mean():.4f}")
print(f"Vis Mask Shape: {vis_mask.shape}")
# Joints 15, 16 (Wrists), 23, 24 (Hips), 27, 28 (Ankles)
for i, joint in enumerate(["L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Ankle", "R_Ankle"]):
    idx = [15, 16, 23, 24, 27, 28][i]
    print(f"{joint} Avg Visibility: {vis_mask[:, idx].mean():.4f}")
