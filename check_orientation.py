import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os

def visualize_npy(npy_path):
    features = np.load(npy_path)
    # features shape: (T, 33, 3)
    frame = features[0] # Take first frame
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x = frame[:, 0]
    y = frame[:, 1]
    z = frame[:, 2]
    
    ax.scatter(x, y, z)
    
    # Draw a "spine" from hip center to shoulder center to see orientation
    # L_HIP=23, R_HIP=24, L_SHOULDER=11, R_SHOULDER=12
    root = (frame[23] + frame[24]) / 2.0
    shoulder = (frame[11] + frame[12]) / 2.0
    ax.plot([root[0], shoulder[0]], [root[1], shoulder[1]], [root[2], shoulder[2]], 'r-', label='Spine')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Orientation: {os.path.basename(npy_path)}")
    plt.legend()
    plt.show() # Note: In this environment, it won't show UI, so I'll print coordinates
    
    print(f"Spine Vector (Shoulder - Root): {shoulder - root}")
    print(f"Magnitude: {np.linalg.norm(shoulder - root)}")

# Find a few processed files
npy_files = glob.glob("data/processed/td/*.npy")[:3]
for nf in npy_files:
    print(f"\nAnalyzing {nf}...")
    visualize_npy(nf)

# If the user's new uploads are in uploads/processed_npy, check those too
uploads = glob.glob("uploads/processed_npy/*.npy")
for uf in uploads:
    print(f"\nAnalyzing upload: {uf}...")
    visualize_npy(uf)
