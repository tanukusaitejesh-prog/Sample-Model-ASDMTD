"""
Reprocess all training videos through the current SpatialSkeletonProcessor.
This ensures training data matches the exact same pipeline used in app.py at inference time.
"""
import os
import glob
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from spatial_processor import SpatialSkeletonProcessor

processor = SpatialSkeletonProcessor()

VIDEO_EXTENSIONS = [".avi", ".mp4", ".mov"]
DATA_DIR = "data"
import config
OUTPUT_DIR = config.FEATURE_DIR_2

for label, subdir in [("ASD", "asd"), ("TD", "td")]:
    input_dir = os.path.join(DATA_DIR, subdir)
    output_dir = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    print(f"\nProcessing {len(video_files)} {label} videos...")
    
    success = 0
    fail = 0
    for vf in sorted(video_files):
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
    
    print(f"  {label}: {success} success, {fail} failed")

print("\nDone! All .npy files regenerated with current SpatialSkeletonProcessor.")
