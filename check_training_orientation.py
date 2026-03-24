import numpy as np
import glob
import os

def get_spine_vector(npy_path):
    f = np.load(npy_path)
    # T, 33, 3
    # Use middle frame for stability
    frame = f[len(f)//2]
    # L_HIP=23, R_HIP=24, L_SHOULDER=11, R_SHOULDER=12
    root = (frame[23] + frame[24]) / 2.0
    shoulder = (frame[11] + frame[12]) / 2.0
    vec = shoulder - root
    return vec / np.linalg.norm(vec)

asd_files = sorted(glob.glob("data/processed/asd/*.npy"))[:5]
td_files = sorted(glob.glob("data/processed/td/*.npy"))[:5]

print("=== ASD SAMPLES ===")
for f in asd_files:
    v = get_spine_vector(f)
    print(f"{os.path.basename(f)}: Spine Vector: {v}")

print("\n=== TD SAMPLES ===")
for f in td_files:
    v = get_spine_vector(f)
    print(f"{os.path.basename(f)}: Spine Vector: {v}")
