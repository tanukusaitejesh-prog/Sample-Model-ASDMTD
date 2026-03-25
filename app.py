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

import cv2
import base64
import tempfile
from typing import Any, Optional
from pydantic import BaseModel
from fastapi.responses import FileResponse, StreamingResponse

import config
from inference import run_inference
from models.pipeline import ASDScreeningPipeline

# Report generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

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
async def predict(file: UploadFile = File(...), form_data: dict = Form(...)):
    model_selection = form_data.get("model_selection", "ensemble")
    thumb_b64 = None
    
    if file.filename.lower().endswith('.npy'):
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
        details["thumbnail"] = thumb_b64 if 'thumb_b64' in locals() else None
        
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
        details["thumbnail"] = thumb_b64 if 'thumb_b64' in locals() else None
        
    return {
        "final_prob": final_prob,
        "risk_level": risk_level,
        "details": details
    }

class ReportData(BaseModel):
    filename: str
    risk_level: str
    final_prob: float
    confidence: float
    model_agreement: float
    chart_image: str  # Base64 string
    thumbnail_image: Optional[str] = None  # Base64 string
    n_clips: int
    rapid_motion_detected: bool

def extract_thumbnail(video_path):
    """Saves the 30th frame (or first valid frame) to a temp file for the report."""
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    # Skip first few frames to get a better image
    for _ in range(30):
        s, f = cap.read()
        if s:
            success, frame = s, f
        else:
            break
    
    if success:
        temp_thumb = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(temp_thumb.name, frame)
        cap.release()
        return temp_thumb.name
    cap.release()
    return None

@app.post("/generate-report")
async def generate_report(data: ReportData):
    """Generates a professional clinical PDF report."""
    fd, path = tempfile.mkstemp(suffix=".pdf")
    doc = SimpleDocTemplate(path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor("#2563eb")
    )
    elements.append(Paragraph("ASD Screening Diagnostic Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Decode chart image
    header, encoded = data.chart_image.split(",", 1)
    chart_data = base64.b64decode(encoded)
    chart_temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    chart_temp.write(chart_data)
    chart_temp.close()
    
    # Header Section with Thumbnail
    if data.thumbnail_image:
        try:
            h_thumb, e_thumb = data.thumbnail_image.split(",", 1)
            thumb_data = base64.b64decode(e_thumb)
            thumb_temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            thumb_temp.write(thumb_data)
            thumb_temp.close()
            
            # Create a table for header info + thumbnail
            header_table_data = [[
                [
                    Paragraph("<b>Session Summary</b>", styles['Heading2']),
                    Paragraph(f"<b>Source File:</b> {data.filename}", styles['Normal']),
                    Paragraph(f"<b>Risk Classification:</b> <font color='{'red' if data.risk_level == 'High Risk' else 'green'}'>{data.risk_level}</font>", styles['Normal']),
                    Paragraph(f"<b>ASD Probability:</b> {data.final_prob:.2%}", styles['Normal']),
                ],
                Image(thumb_temp.name, width=120, height=90)
            ]]
            header_table = Table(header_table_data, colWidths=[350, 150])
            header_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))
            elements.append(header_table)
            
            # Cleanup thumb later
            thumb_temp_name = thumb_temp.name
        except:
            elements.append(Paragraph("<b>Session Summary</b>", styles['Heading2']))
            elements.append(Paragraph(f"<b>Source File:</b> {data.filename}", styles['Normal']))
            elements.append(Paragraph(f"<b>Risk Classification:</b> <font color='{'red' if data.risk_level == 'High Risk' else 'green'}'>{data.risk_level}</font>", styles['Normal']))
            elements.append(Paragraph(f"<b>ASD Probability:</b> {data.final_prob:.2%}", styles['Normal']))
            thumb_temp_name = None
    else:
        elements.append(Paragraph("<b>Session Summary</b>", styles['Heading2']))
        elements.append(Paragraph(f"<b>Source File:</b> {data.filename}", styles['Normal']))
        elements.append(Paragraph(f"<b>Risk Classification:</b> <font color='{'red' if data.risk_level == 'High Risk' else 'green'}'>{data.risk_level}</font>", styles['Normal']))
        elements.append(Paragraph(f"<b>ASD Probability:</b> {data.final_prob:.2%}", styles['Normal']))
        thumb_temp_name = None

    elements.append(Spacer(1, 20))
    
    # Results Table
    results_data = [
        ["Metric", "Value", "Notes"],
        ["Confidence Level", f"{data.confidence:.1%}", "Statistical certainty of the model"],
        ["Model Agreement", f"{data.model_agreement:.1%}", "Consensus between ensemble folds"],
        ["Observation Period", f"{data.n_clips} segments", "Number of 3-second clips analyzed"],
        ["Rapid Motion", "YES" if data.rapid_motion_detected else "NO", "Significant hyperactivity detected"]
    ]
    t = Table(results_data, colWidths=[150, 100, 200])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#475569")),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 30))
    
    # Motion Analytics Chart
    elements.append(Paragraph("<b>Temporal Motion Analytics</b>", styles['Heading2']))
    # Decode chart image
    # chart_image is data:image/png;base64,....
    header, encoded = data.chart_image.split(",", 1)
    chart_data = base64.b64decode(encoded)
    chart_temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    chart_temp.write(chart_data)
    chart_temp.close()
    
    img = Image(chart_temp.name, width=500, height=250)
    elements.append(img)
    elements.append(Paragraph("<font size=8 color='grey'>The graph represents per-frame motion energy across body groups (Head, Arms, Legs).</font>", styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Footer
    elements.append(Paragraph("<b>Disclaimer:</b> This report is generated by an AI screening tool and is intended for clinical assistance only. A formal diagnosis should only be made by a qualified healthcare professional.", ParagraphStyle('Disclaimer', fontName='Helvetica-Oblique', fontSize=8, textColor=colors.grey)))
    
    doc.build(elements)
    
    # Cleanup temp chart image
    try: os.unlink(chart_temp.name)
    except: pass
    
    return FileResponse(path, filename=f"ASD_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", media_type='application/pdf')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
