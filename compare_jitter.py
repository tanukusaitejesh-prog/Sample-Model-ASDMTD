import numpy as np
from spatial_processor import SpatialSkeletonProcessor
import warnings
warnings.filterwarnings('ignore')

proc = SpatialSkeletonProcessor()

video1 = r"C:\Users\saite\OneDrive\Desktop\OS\ezgif-3d0f47c951f3ae3e.mp4"
video2 = r"data/td/80_video.avi"

print("Extracting EZGIF Video...")
f1 = proc.process_video(video1)
print("Extracting Pristine TD Video...")
f2 = proc.process_video(video2)

def compute_jitter(features):
    # Velocity is frame-to-frame difference
    velocity = np.diff(features, axis=0)
    # Acceleration is frame-to-frame change in velocity (jitter)
    accel = np.diff(velocity, axis=0)
    
    # We want to measure the average magnitude of the jitter
    mag_accel = np.linalg.norm(accel, axis=-1)  # Length of acceleration vectors per joint per frame
    return np.mean(mag_accel)

def compute_missing(features):
    return np.sum(np.isnan(features))

print(f"\n=== EZGIF Video (False Positive) ===")
print(f"Shape: {f1.shape}")
print(f"Missing (NaNs): {compute_missing(f1)}")
print(f"High-Frequency Jitter Score: {compute_jitter(f1):.6f}")

print(f"\n=== PRISTINE TD Video (True Negative) ===")
print(f"Shape: {f2.shape}")
print(f"Missing (NaNs): {compute_missing(f2)}")
print(f"High-Frequency Jitter Score: {compute_jitter(f2):.6f}")

print(f"\n=== EZGIF first 5 frames joint 0 ===")
print(f1[:5, 0, :])
