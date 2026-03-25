import os
import io
import glob
import uvicorn
import numpy as np
import torch
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

import config
from inference import run_inference
from models.pipeline import ASDScreeningPipeline

app = FastAPI(title="ASD Screening Multi-Fold Evaluator")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
os.makedirs(config.UPLOAD_NPY_DIR, exist_ok=True)

DEVICE = config.DEVICE

# Preload models
models_dict = {}
# Use the new orientation-invariant models
model_files = sorted(glob.glob(os.path.join(config.SAVED_MODELS_DIR_2, "fold_*.pt")))
# Fallback to older models if new ones don't exist
if not model_files:
    model_files = sorted(glob.glob(os.path.join(config.SAVED_MODELS_DIR, "fold_*.pt")))
for mf in model_files:
    name = os.path.basename(mf).replace(".pt", "")
    checkpoint = torch.load(mf, map_location=DEVICE, weights_only=False)
    m = ASDScreeningPipeline().to(DEVICE)
    m.load_state_dict(checkpoint["model_state"])
    m.eval()
    models_dict[name] = m

print(f"Loaded {len(models_dict)} fold models: {list(models_dict.keys())}")

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.get("/models")
async def get_models():
    m = list(models_dict.keys())
    m.append("ensemble")
    return {"models": m}

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_selection: str = Form(...)):
    if file.filename.endswith('.npy'):
        try:
            contents = await file.read()
            features = np.load(io.BytesIO(contents), allow_pickle=True)
        except Exception as e:
            return {"error": f"Failed to load .npy file: {str(e)}"}
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        try:
            from spatial_processor import SpatialSkeletonProcessor
        except ImportError as e:
            return {"error": f"Import failed: {str(e)}"}
            
        # Save temp video
        temp_video_path = f"temp_{file.filename}"
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())
            
        try:
            processor = SpatialSkeletonProcessor()
            features = processor.process_video(temp_video_path)
            
            # Reject if empty, None, or entirely zero/NaN
            if features is None or len(features) == 0:
                raise ValueError("No landmarks were extracted.")
            if np.all(features == 0) or np.all(np.isnan(features)):
                raise ValueError("Video is entirely unclear. MediaPipe could not detect a valid person/skeleton in any frame.")
            # Reject if the skeleton is completely frozen across time (meaning MediaPipe interpolated a single static ghost frame)
            temporal_std = float(np.mean(np.std(features, axis=0)))
            if temporal_std < 0.05:
                raise ValueError(f"Skeleton is completely frozen (Movement Variance: {temporal_std:.4f}). MediaPipe could not detect fluid human movement in this incredibly short or static video.")

        except Exception as e:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            return {"error": f"Failed to process video: {str(e)}"}
            
        if os.path.exists(temp_video_path):
            # Save the processed features permanently for the user
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join([c if c.isalnum() or c in "._-" else "_" for c in file.filename])
            save_path = os.path.join(config.UPLOAD_NPY_DIR, f"{timestamp}_{safe_name}.npy")
            np.save(save_path, features)
            
            os.remove(temp_video_path)
    else:
        return {"error": "Please upload a valid .npy or video file (.mp4, .avi, .mov)."}
    
    if model_selection == "ensemble":
        all_probs = []
        all_clip_probs = []
        all_temporal = []
        models_list = list(models_dict.values())
        if not models_list:
            return {"error": "No fold models loaded for ensemble"}
            
        for model in models_list:
            prob, clip_probs, _, fold_details = run_inference(features, model, DEVICE)
            all_probs.append(prob)
            all_clip_probs.extend(clip_probs)
            all_temporal.append(fold_details.get("temporal_data", []))
            
        final_prob = float(np.mean(all_probs))
        risk_level, confidence, details = ASDScreeningPipeline.classify_risk(final_prob, all_clip_probs)
        
        # Aggregate temporal data by averaging probabilities at each frame
        if all_temporal:
            n_models = len(models_list)
            first_fold = all_temporal[0]
            agg_temporal = []
            for i in range(len(first_fold)):
                clip_sum_prob = sum(fold[i]["prob"] for fold in all_temporal)
                agg_temporal.append({
                    "start_frame": first_fold[i]["start_frame"],
                    "end_frame": first_fold[i]["end_frame"],
                    "prob": clip_sum_prob / n_models,
                    "signatures": first_fold[i]["signatures"] 
                })
            details["temporal_data"] = agg_temporal

        details["n_clips"] = len(all_clip_probs) // len(models_list)
        details["model_agreement_std"] = float(np.std(all_probs))
        # Frame-level energy is identical across folds (same raw features)
        # Grab from first fold's already-computed details
        first_fold_details = run_inference(features, models_list[0], DEVICE)[3]
        if "frame_energies" in first_fold_details:
            details["frame_energies"] = first_fold_details["frame_energies"]
    else:
        if model_selection not in models_dict:
            return {"error": "Invalid model selection"}
        model = models_dict[model_selection]
        final_prob, clip_probs, risk_level, details = run_inference(features, model, DEVICE)
        
    return {
        "final_prob": final_prob,
        "risk_level": risk_level,
        "details": details
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
