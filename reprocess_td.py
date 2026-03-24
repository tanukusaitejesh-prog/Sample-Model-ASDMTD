"""Reprocess ONLY TD videos (ASD are already done)."""
import os, glob, numpy as np, warnings
warnings.filterwarnings("ignore")
from spatial_processor import SpatialSkeletonProcessor

processor = SpatialSkeletonProcessor()
input_dir = "data/td"
output_dir = "data/processed/td"
os.makedirs(output_dir, exist_ok=True)

video_files = sorted(glob.glob(os.path.join(input_dir, "*.avi")) + glob.glob(os.path.join(input_dir, "*.mp4")))
print(f"Processing {len(video_files)} TD videos...")

success = fail = 0
for vf in video_files:
    name = os.path.splitext(os.path.basename(vf))[0]
    out_path = os.path.join(output_dir, f"{name}.npy")
    try:
        features = processor.process_video(vf)
        if features is not None and len(features) > 0:
            np.save(out_path, features)
            print(f"  OK: {name} -> {features.shape}")
            success += 1
        else:
            print(f"  FAIL (no features): {name}")
            fail += 1
    except Exception as e:
        print(f"  FAIL: {name} -> {e}")
        fail += 1

print(f"\nTD: {success} success, {fail} failed")
