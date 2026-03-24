import numpy as np
from spatial_processor import SpatialSkeletonProcessor
from models.pipeline import ASDScreeningPipeline
from inference import run_inference
import torch, glob, os, config, json

proc = SpatialSkeletonProcessor()
import warnings; warnings.filterwarnings('ignore')

f1 = proc.process_video(r"C:\Users\saite\OneDrive\Desktop\OS\ezgif-3d0f47c951f3ae3e.mp4")
f2 = proc.process_video(r"data/td/80_video.avi")

DEVICE = config.DEVICE
model_path = sorted(glob.glob(os.path.join(config.SAVED_MODELS_DIR, "fold_0.pt")))[0]
m = ASDScreeningPipeline().to(DEVICE)
m.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False)["model_state"])
m.eval()

_, c1, _, _ = run_inference(f1, m, DEVICE)
_, c2, _, _ = run_inference(f2, m, DEVICE)

output = {
    "EZGIF": {
        "median_mag": float(np.mean(np.linalg.norm(f1, axis=-1))),
        "std_x": float(np.std(f1[:,:,0])),
        "jitter_x": float(np.mean(np.abs(np.diff(f1[:,:,0], axis=0)))),
        "clip_probs": [float(c) for c in c1]
    },
    "TD": {
        "median_mag": float(np.mean(np.linalg.norm(f2, axis=-1))),
        "std_x": float(np.std(f2[:,:,0])),
        "jitter_x": float(np.mean(np.abs(np.diff(f2[:,:,0], axis=0)))),
        "clip_probs": [float(c) for c in c2]
    }
}
with open("diagnostic.json", "w", encoding="utf-8") as out:
    json.dump(output, out, indent=2)
