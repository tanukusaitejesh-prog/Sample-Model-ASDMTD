"""
Central configuration for the Small-Data Optimized ASD Screening Pipeline.
All hyperparameters, paths, and thresholds in one place.
"""
import os
import torch

# ──────────────────────── PATHS ────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ASD_DIR = os.path.join(DATA_DIR, "asd")
TD_DIR = os.path.join(DATA_DIR, "td")
FEATURE_DIR = os.path.join(DATA_DIR, "processed")
FEATURE_DIR_2 = os.path.join(DATA_DIR, "processed_2")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
SAVED_MODELS_DIR_2 = os.path.join(ROOT_DIR, "saved_models_2")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
UPLOAD_NPY_DIR = os.path.join(ROOT_DIR, "uploads", "processed_npy")


# ──────────────────────── DEVICE ───────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────── VIDEO / SLIDING WINDOW ───────
FRAME_SIZE = 224                # ResNet input size
FPS_TARGET = 15                 # Downsample videos to this FPS
WINDOW_SIZE = 48                # T = 48 frames per clip (midpoint of 32-64)
WINDOW_STRIDE = 16              # Overlap stride
MIN_FRAMES = 32                 # Minimum frames to form a valid clip

# ──────────────────────── FEATURE EXTRACTION (Stage 1) ─
POSE_DIM = 99                   # 33 landmarks * 3 coords
FEATURE_DIM = 128               # Projection head output dim
NUM_STREAMS = 1                 # Pose only
FACE_SCORE_THRESH = 0.5         # Min face detection confidence
POSE_SCORE_THRESH = 0.5         # Min pose detection confidence
HAND_SCORE_THRESH = 0.5         # Min hand detection confidence

# ──────────────────────── SOFT TEMPORAL ATTENTION (Stage 2)
ATTN_TEMPERATURE = 1.0          # Attention softmax temperature
TOP_K_CLIPS = 16                # Number of salient clips to select

# ──────────────────────── EVENT TRANSFORMER (Stage 3) ──
TRANSFORMER_LAYERS = 3
TRANSFORMER_DIM = 128
TRANSFORMER_HEADS = 4
TRANSFORMER_DROPOUT = 0.4       # Stronger regularization (was 0.3)
TRANSFORMER_FF_DIM = 256        # Feed-forward hidden dim

# ──────────────────────── TRAINING ─────────────────────
NUM_FOLDS = 5                   # Stratified K-fold CV
BATCH_SIZE = 8
NUM_EPOCHS = 300
LEARNING_RATE = 5e-4            # Reduced for small data (was 1e-3)
WEIGHT_DECAY = 1e-4
POS_WEIGHT = 1.0                # Balanced — no ASD bias (was 1.5)
EARLY_STOP_PATIENCE = 30        # Stop earlier to prevent overfitting (was 50)
LR_SCHEDULER = "cosine"         # Cosine annealing
MIXUP_ALPHA = 0.1               # Lighter mixup (was 0.2)
MIXUP_WARMUP_EPOCHS = 30        # Only apply mixup before this epoch
LABEL_SMOOTHING = 0.1           # Prevent overconfident outputs

# ──────────────────────── CALIBRATION (Stage 4) ────────
TAU_LOW = 0.35                  # Below this → Low Risk
TAU_HIGH = 0.65                 # Above this → High Risk
DECISION_THRESHOLD = 0.65       # prob > this → ASD (was implicit 0.5)
INFERENCE_TEMPERATURE = 2.5     # Dampen overconfident logits (pulls 0.95 -> 0.70)
ABSTAIN_CONFIDENCE = 0.2        # |p - 0.5| < this → ABSTAIN
CONSISTENCY_STD_THRESH = 0.25   # Clip-level std > this → ABSTAIN
CONSISTENCY_RATIO_THRESH = 0.3  # Min fraction of clips > threshold for ASD

# ──────────────────────── AGGREGATION ──────────────────
TRIMMED_MEAN_LOW = 0.10         # Drop bottom 10% of clips
TRIMMED_MEAN_HIGH = 0.90        # Drop top 10% of clips
CONSISTENCY_DAMPEN = 0.7        # Dampen score when ratio check fails

# ──────────────────────── AUGMENTATION ─────────────────
AUG_TEMPORAL_JITTER = 4         # Random shift ± frames
AUG_GAUSSIAN_NOISE_STD = 0.01   # Feature-level noise
AUG_FEATURE_DROPOUT = 0.1       # Random feature zeroing
AUG_FLIP_PROB = 0.5             # Horizontal flip probability (swap L/R hands)

# ──────────────────────── HARD EXAMPLE MINING ──────────
HARD_MINING_WEIGHT = 2.0        # Extra weight for hard samples
HARD_MINING_START_EPOCH = 10    # Start mining after N epochs

# ──────────────────────── EVALUATION ───────────────────
EVAL_METRICS = [
    "sensitivity", "specificity", "auc", "f1",
    "accuracy", "false_negative_rate", "ppv", "npv"
]

# ──────────────────────── DIRECTORIES (auto-create) ────
for d in [DATA_DIR, ASD_DIR, TD_DIR, FEATURE_DIR, RESULTS_DIR, SAVED_MODELS_DIR]:
    os.makedirs(d, exist_ok=True)
