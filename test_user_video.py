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

VIDEO_PATH = r"C:\Users\saite\OneDrive\Desktop\OS\ezgif-3d0f47c951f3ae3e.mp4"
DEVICE = config.DEVICE

print(f"Testing newly provided video: {VIDEO_PATH}")

# 1. Load ENSEMBLE models just like app.py does for "ensemble" selection
models_dict = {}
model_files = sorted(glob.glob(os.path.join(config.SAVED_MODELS_DIR, "fold_*.pt")))
for mf in model_files:
    name = os.path.basename(mf).replace(".pt", "")
    checkpoint = torch.load(mf, map_location=DEVICE, weights_only=False)
    m = ASDScreeningPipeline().to(DEVICE)
    m.load_state_dict(checkpoint["model_state"])
    m.eval()
    models_dict[name] = m

print(f"Loaded {len(models_dict)} fold models successfully.")

# 2. Extract features exactly like app.py
processor = SpatialSkeletonProcessor()
features_live = processor.process_video(VIDEO_PATH)

if features_live is None or len(features_live) == 0:
    print("Video was entirely empty!")
    exit(1)

temporal_std = float(np.mean(np.std(features_live, axis=0)))
print(f"Live features shape: {features_live.shape}, Temporal STD: {temporal_std:.4f}")

# Check variance block
if temporal_std < 0.05:
    print(f"FAILED SAFETY BLOCK! Temporal std {temporal_std:.4f} < 0.05. This is a frozen ghost video.")
    exit(1)

# 3. Predict exactly like app.py (ensemble)
all_probs = []
models_list = list(models_dict.values())
for model in models_list:
    prob, _, _, _ = run_inference(features_live, model, DEVICE)
    all_probs.append(prob)

final_prob_live = float(np.mean(all_probs))
print(f"\n=== NATIVE RESULTS ON DISK MODELS ===")
print(f"Raw Ensemble Array: {[round(p,4) for p in all_probs]}")
print(f"TRUE LIVE APP PREDICTION: {final_prob_live:.4f}")
