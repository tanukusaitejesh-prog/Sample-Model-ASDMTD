import numpy as np
from spatial_processor import SpatialSkeletonProcessor
import warnings
warnings.filterwarnings('ignore')

proc = SpatialSkeletonProcessor()
ezgif = proc.process_video(r"C:\Users\saite\OneDrive\Desktop\OS\ezgif-3d0f47c951f3ae3e.mp4")
td = proc.process_video(r"data/td/80_video.avi")

# Compare geometric scale (standard deviation across points)
print(f"EZGIF Median joint magnitude from root: {np.mean(np.linalg.norm(ezgif, axis=-1)):.4f}")
print(f"TD Median joint magnitude from root: {np.mean(np.linalg.norm(td, axis=-1)):.4f}")

print(f"EZGIF Standard Deviation of X coordinates: {np.std(ezgif[:,:,0]):.4f}")
print(f"TD Standard Deviation of X coordinates: {np.std(td[:,:,0]):.4f}")

print(f"EZGIF Jitter X: {np.mean(np.abs(np.diff(ezgif[:,:,0], axis=0))):.4f}")
print(f"TD Jitter X: {np.mean(np.abs(np.diff(td[:,:,0], axis=0))):.4f}")

# Look at specific clip scores
from models.pipeline import ASDScreeningPipeline
import torch, glob, os, config
from inference import run_inference
DEVICE = config.DEVICE
model_path = sorted(glob.glob(os.path.join(config.SAVED_MODELS_DIR, "fold_0.pt")))[0]
m = ASDScreeningPipeline().to(DEVICE)
m.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False)["model_state"])
m.eval()

_, clip_probs, _, _ = run_inference(ezgif, m, DEVICE)
print(f"EZGIF Clip Probs: {clip_probs}")

_, clip_probs_td, _, _ = run_inference(td, m, DEVICE)
print(f"TD Clip Probs: {clip_probs_td}")
