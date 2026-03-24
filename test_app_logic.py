import os
import glob
import numpy as np
import torch
import config
from spatial_processor import SpatialSkeletonProcessor
from inference import run_inference
from models.pipeline import ASDScreeningPipeline
import warnings
warnings.filterwarnings('ignore')

VIDEO_PATH = r"data/td/80_video.avi"  # Genuinely moving TD video
DEVICE = config.DEVICE

print(f"Testing video: {VIDEO_PATH}")

# 1. Load model just like app.py
model_files = sorted(glob.glob(os.path.join(config.SAVED_MODELS_DIR, "fold_*.pt")))
model_path = model_files[0]
print(f"Using model: {model_path}")
checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
model = ASDScreeningPipeline().to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# 2. Extract features exactly like app.py
processor = SpatialSkeletonProcessor()
features_live = processor.process_video(VIDEO_PATH)

# Verify movement
temporal_std = float(np.mean(np.std(features_live, axis=0)))
print(f"Live features shape: {features_live.shape}, Temporal STD: {temporal_std:.4f}")

# 3. Predict exactly like app.py
final_prob_live, _, _, _ = run_inference(features_live, model, DEVICE)

# 4. Compare with the pre-extracted .npy from training!
npy_path = VIDEO_PATH.replace('data/td', 'data/processed/td').replace('.avi', '.npy').replace('.mp4', '.npy')
if os.path.exists(npy_path):
    features_train = np.load(npy_path, allow_pickle=True)
    
    print(f"\nComparing Live vs Train Matrices:")
    if features_live.shape == features_train.shape:
        diff = np.abs(features_live - features_train).mean()
        print(f"Mean Abs Difference: {diff:.6f}")
        
        print(f"Frame 0, Joint 0 (Live) : {features_live[0, 0, :]}")
        print(f"Frame 0, Joint 0 (Train): {features_train[0, 0, :]}")
        
        # Check specific joints
        print(f"Frame 50, Joint 15 (Live) : {features_live[50, 15, :]}")
        print(f"Frame 50, Joint 15 (Train): {features_train[50, 15, :]}")
        
    else:
        print(f"Shapes differ! Live: {features_live.shape}, Train: {features_train.shape}")
        
    final_prob_train, _, _, _ = run_inference(features_train, model, DEVICE)

print(f"\n=== RESULTS ===")
print(f"LIVE APP PREDICTION:  {final_prob_live:.4f}")
if os.path.exists(npy_path):
    print(f"TRAIN NPY PREDICTION: {final_prob_train:.4f}")
