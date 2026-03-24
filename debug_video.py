"""Debug script - writes results to file to avoid terminal truncation."""
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from spatial_processor import SpatialSkeletonProcessor

processor = SpatialSkeletonProcessor()
live = processor.process_video(r"C:\Users\saite\OneDrive\Desktop\OS\bbursa-milenabb.mp4")

with open("debug_results.txt", "w") as f:
    f.write("=== LIVE VIDEO ===\n")
    f.write(f"Shape: {live.shape}\n")
    f.write(f"Mean: {live.mean():.6f}\n")
    f.write(f"Std:  {live.std():.6f}\n")
    f.write(f"Min:  {live.min():.6f}\n")
    f.write(f"Max:  {live.max():.6f}\n")
    f.write(f"X mean: {live[:,:,0].mean():.6f}  std: {live[:,:,0].std():.6f}\n")
    f.write(f"Y mean: {live[:,:,1].mean():.6f}  std: {live[:,:,1].std():.6f}\n")
    f.write(f"Z mean: {live[:,:,2].mean():.6f}  std: {live[:,:,2].std():.6f}\n")
    f.write(f"First 3 frames joint 0:\n")
    for i in range(min(3, live.shape[0])):
        f.write(f"  Frame {i}: {live[i, 0]}\n")
    
    f.write("\n=== TRAINING TD VIDEO ===\n")
    train = np.load("data/processed/td/101_video.npy")
    f.write(f"Shape: {train.shape}\n")
    f.write(f"Mean: {train.mean():.6f}\n")
    f.write(f"Std:  {train.std():.6f}\n")
    f.write(f"Min:  {train.min():.6f}\n")
    f.write(f"Max:  {train.max():.6f}\n")
    f.write(f"X mean: {train[:,:,0].mean():.6f}  std: {train[:,:,0].std():.6f}\n")
    f.write(f"Y mean: {train[:,:,1].mean():.6f}  std: {train[:,:,1].std():.6f}\n")
    f.write(f"Z mean: {train[:,:,2].mean():.6f}  std: {train[:,:,2].std():.6f}\n")
    f.write(f"First 3 frames joint 0:\n")
    for i in range(min(3, train.shape[0])):
        f.write(f"  Frame {i}: {train[i, 0]}\n")
    
    f.write("\n=== MODEL INFERENCE ===\n")
    import torch
    import config
    from inference import load_ensemble_models, run_inference
    device = config.DEVICE
    models_list = load_ensemble_models(device)
    
    all_probs = []
    all_clips = []
    for m in models_list:
        p, clips, _, _ = run_inference(live, m, device)
        all_probs.append(p)
        all_clips.append(clips)
    f.write(f"Live video probs: {[round(p,4) for p in all_probs]}\n")
    f.write(f"Live video mean:  {np.mean(all_probs):.4f}\n")
    f.write(f"Live clip details (fold 0): {[round(c,4) for c in all_clips[0]]}\n")
    
    all_probs_train = []
    for m in models_list:
        p, _, _, _ = run_inference(train, m, device)
        all_probs_train.append(p)
    f.write(f"Train video probs: {[round(p,4) for p in all_probs_train]}\n")
    f.write(f"Train video mean:  {np.mean(all_probs_train):.4f}\n")

print("Results written to debug_results.txt")
