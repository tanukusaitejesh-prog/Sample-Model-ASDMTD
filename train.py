"""
Training Script — Clinically Aligned ASD Screening
───────────────────────────────────────────────────
Implements ALL 10 high-impact improvements:

 #1  Sensitivity-optimized loss (pos_weight > 1)
 #2  ABSTAIN as first-class output (strict confidence filter)
 #3  Top-percentile aggregation (top 30% clips)
 #4  Consistency check (high variance → ABSTAIN)
 #5  Stratified 5-fold CV with sensitivity/specificity/AUC tracking
 #6  Ensemble across folds (average predictions from all 5 models)
 #7  Post-hoc temperature calibration on held-out validation
 #8  Hard example mining (upweight misclassified samples)
 #9  Frame quality filtering (drop bad detections)
 #10 Clinical evaluation metrics

Usage:
    python train.py                      (full training)
    python train.py --test-mode          (smoke test with synthetic data)
    python train.py --epochs 50          (custom epoch count)
"""
import os
import sys
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, classification_report,
)
from collections import defaultdict

import config
from dataset import get_feature_files_and_labels, build_dataloader, ASDFeatureDataset
from models.pipeline import ASDScreeningPipeline


def mixup_data(pose, labels, alpha=config.MIXUP_ALPHA):
    """
    Feature-level mixup for regularization.
    Randomly mixes pairs of samples with a Beta-distributed weight.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = pose.size(0)
    index = torch.randperm(batch_size).to(pose.device)

    mixed_pose = lam * pose + (1 - lam) * pose[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]

    return mixed_pose, mixed_labels


def compute_sample_weights(predictions, labels, epoch):
    """
    Hard example mining (improvement #8).
    Upweights misclassified samples after warm-up epochs.

    Returns per-sample weights for the loss function.
    """
    weights = torch.ones_like(labels)

    if epoch >= config.HARD_MINING_START_EPOCH:
        with torch.no_grad():
            pred_labels = (predictions > 0.5).float()
            misclassified = (pred_labels != labels).float()
            weights = 1.0 + misclassified * (config.HARD_MINING_WEIGHT - 1.0)

    return weights


def train_one_epoch(model, dataloader, optimizer, device, epoch, pos_weight_tensor):
    """Train for one epoch with all improvements."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        pose = batch["pose"].to(device)
        quality_mask = batch["quality_mask"].to(device)
        labels = batch["label"].to(device)

        # Mixup augmentation — only during warmup epochs
        if epoch < config.MIXUP_WARMUP_EPOCHS:
            pose, mixed_labels = mixup_data(pose, labels)
        else:
            mixed_labels = labels

        # Apply label smoothing
        smoothed_labels = mixed_labels * (1.0 - config.LABEL_SMOOTHING) + 0.5 * config.LABEL_SMOOTHING

        # Forward
        logit, prob, attn_weights = model(pose, quality_mask)

        # Weighted BCE loss
        base_loss = F.binary_cross_entropy_with_logits(
            logit.squeeze(-1), smoothed_labels,
            pos_weight=pos_weight_tensor
        )

        # Hard example mining weights
        sample_weights = compute_sample_weights(prob.squeeze(-1).detach(), labels, epoch)
        per_sample_loss = F.binary_cross_entropy_with_logits(
            logit.squeeze(-1), smoothed_labels, reduction="none"
        )
        weighted_loss = (per_sample_loss * sample_weights).mean()

        # Combined loss
        loss = 0.5 * base_loss + 0.5 * weighted_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(prob.squeeze(-1).detach().cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_preds, all_labels


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate and collect per-clip and per-video predictions."""
    model.eval()
    total_loss = 0

    clip_preds = []
    clip_labels = []
    clip_video_ids = []
    all_logits = []

    for batch in dataloader:
        pose = batch["pose"].to(device)
        quality_mask = batch["quality_mask"].to(device)
        labels = batch["label"].to(device)

        logit, prob, attn_weights = model(pose, quality_mask)

        loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), labels)
        total_loss += loss.item()

        clip_preds.extend(prob.squeeze(-1).cpu().numpy().tolist())
        clip_labels.extend(labels.cpu().numpy().tolist())
        clip_video_ids.extend(batch["video_id"])
        all_logits.append(logit.cpu())

    avg_loss = total_loss / max(len(dataloader), 1)
    all_logits = torch.cat(all_logits, dim=0) if all_logits else torch.tensor([])

    # ─── Per-video aggregation with top-percentile (improvement #3) ───
    video_preds = defaultdict(list)
    video_labels = {}
    for pred, label, vid_id in zip(clip_preds, clip_labels, clip_video_ids):
        video_preds[vid_id].append(pred)
        video_labels[vid_id] = label

    video_level_preds = {}
    video_level_labels = {}
    for vid_id in video_preds:
        clip_probs = video_preds[vid_id]
        agg_prob = ASDScreeningPipeline.aggregate_video_predictions(clip_probs)
        video_level_preds[vid_id] = agg_prob
        video_level_labels[vid_id] = video_labels[vid_id]

    return {
        "loss": avg_loss,
        "clip_preds": clip_preds,
        "clip_labels": clip_labels,
        "clip_video_ids": clip_video_ids,
        "video_preds": video_level_preds,
        "video_labels": video_level_labels,
        "logits": all_logits,
    }


def compute_clinical_metrics(video_preds, video_labels):
    """
    Compute clinical-grade metrics (improvement #10).

    Returns dict with: sensitivity, specificity, AUC, F1, accuracy,
    false_negative_rate, PPV, NPV, and risk breakdown.
    """
    vids = sorted(video_preds.keys())
    preds = np.array([video_preds[v] for v in vids])
    labels = np.array([video_labels[v] for v in vids])

    # Binary predictions using DECISION_THRESHOLD (not 0.5)
    pred_binary = (preds > config.DECISION_THRESHOLD).astype(int)

    metrics = {}

    # AUC
    try:
        metrics["auc"] = roc_auc_score(labels, preds)
    except ValueError:
        metrics["auc"] = 0.5

    metrics["accuracy"] = accuracy_score(labels, pred_binary)
    metrics["f1"] = f1_score(labels, pred_binary, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, pred_binary, labels=[0, 1]).ravel()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["false_negative_rate"] = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive predictive value
    metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value

    # Risk tier breakdown
    risk_counts = {"LOW_RISK": 0, "RECHECK": 0, "HIGH_RISK": 0, "ABSTAIN": 0}
    risk_correct = {"LOW_RISK": 0, "RECHECK": 0, "HIGH_RISK": 0}

    for vid in vids:
        prob = video_preds[vid]
        label = video_labels[vid]
        clip_probs = None  # Single aggregated prediction

        risk, conf, details = ASDScreeningPipeline.classify_risk(prob)
        risk_counts[risk] += 1

        if risk == "LOW_RISK" and label == 0:
            risk_correct["LOW_RISK"] += 1
        elif risk == "HIGH_RISK" and label == 1:
            risk_correct["HIGH_RISK"] += 1

    metrics["risk_breakdown"] = risk_counts
    metrics["risk_accuracy"] = risk_correct

    return metrics


def train_fold(fold_idx, train_files, train_labels, train_vids,
               val_files, val_labels, val_vids, args, device):
    """Train a single fold and return the best model + metrics."""
    print(f"\n{'='*70}")
    print(f"  FOLD {fold_idx + 1}/{config.NUM_FOLDS}")
    print(f"  Train: {len(train_files)} videos | Val: {len(val_files)} videos")
    print(f"{'='*70}")

    # Build dataloaders
    train_loader = build_dataloader(train_files, train_labels, train_vids, augment=True)
    val_loader = build_dataloader(val_files, val_labels, val_vids, augment=False, shuffle=False)

    # Initialize model
    model = ASDScreeningPipeline().to(device)

    # Optimizer + scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Pos weight for sensitivity-focused loss (improvement #1)
    pos_weight = torch.tensor([config.POS_WEIGHT]).to(device)

    best_composite = 0.0
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_sensitivity": [], "val_specificity": []}

    for epoch in range(args.epochs):
        # Train
        train_loss, _, _ = train_one_epoch(
            model, train_loader, optimizer, device, epoch, pos_weight
        )

        # Validate
        val_results = validate(model, val_loader, device)
        val_metrics = compute_clinical_metrics(
            val_results["video_preds"], val_results["video_labels"]
        )

        scheduler.step()

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_results["loss"])
        history["val_auc"].append(val_metrics["auc"])
        history["val_sensitivity"].append(val_metrics["sensitivity"])
        history["val_specificity"].append(val_metrics["specificity"])

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_results['loss']:.4f} | "
                f"AUC: {val_metrics['auc']:.4f} | "
                f"Sens: {val_metrics['sensitivity']:.3f} | "
                f"Spec: {val_metrics['specificity']:.3f}"
            )

        # Early stopping on composite metric: 0.5*AUC + 0.25*sens + 0.25*spec
        composite = (
            0.5 * val_metrics["auc"]
            + 0.25 * val_metrics["sensitivity"]
            + 0.25 * val_metrics["specificity"]
        )
        if composite > best_composite:
            best_composite = composite
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"  [STOP] Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    model.to(device)

    # ─── Post-hoc temperature calibration (improvement #7) ───
    print("  [CALIBRATION] Calibrating temperature on validation data...")
    val_results = validate(model, val_loader, device)
    if val_results["logits"].numel() > 0:
        val_labels_tensor = torch.tensor(val_results["clip_labels"])
        temp = model.event_transformer.calibrate_temperature(
            val_results["logits"].to(device), val_labels_tensor.to(device)
        )
        print(f"  Temperature: {temp:.4f}")

    # Final validation metrics
    val_results = validate(model, val_loader, device)
    final_metrics = compute_clinical_metrics(
        val_results["video_preds"], val_results["video_labels"]
    )

    # Save model
    save_path = os.path.join(config.SAVED_MODELS_DIR_2, f"fold_{fold_idx}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "metrics": final_metrics,
        "history": history,
        "fold": fold_idx,
    }, save_path)
    print(f"  [SAVED] {save_path}")

    return model, final_metrics, val_results, history


def ensemble_predict(models, dataloader, device):
    """
    Ensemble prediction across all fold models (improvement #6).
    Averages predictions from all models for each sample.
    """
    all_video_preds = defaultdict(list)
    video_labels = {}

    for model in models:
        model.eval()
        val_results = validate(model, dataloader, device)

        for vid_id, pred in val_results["video_preds"].items():
            all_video_preds[vid_id].append(pred)
        video_labels.update(val_results["video_labels"])

    # Average across models
    ensemble_preds = {}
    for vid_id in all_video_preds:
        ensemble_preds[vid_id] = float(np.mean(all_video_preds[vid_id]))

    return ensemble_preds, video_labels


def main():
    parser = argparse.ArgumentParser(description="Train ASD Screening Pipeline")
    parser.add_argument("--test-mode", action="store_true",
                        help="Quick test with reduced epochs/folds")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--folds", type=int, default=config.NUM_FOLDS)
    args = parser.parse_args()

    if args.test_mode:
        args.epochs = 5
        args.folds = 2
        print("TEST MODE: 2 folds, 5 epochs")

    device = config.DEVICE
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs} | Folds: {args.folds}")

    # Ensure output directory exists
    os.makedirs(config.SAVED_MODELS_DIR_2, exist_ok=True)

    # Load feature files
    files, labels, video_ids = get_feature_files_and_labels()
    if len(files) == 0:
        print("❌ No feature files found! Run extract_features.py first.")
        print(f"   Expected directory: {config.FEATURE_DIR}")
        sys.exit(1)

    print(f"\n[DATA] {len(files)} videos ({sum(labels)} ASD, {len(labels) - sum(labels)} TD)")

    # ─── Stratified K-Fold (improvement #5) ───
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    files = np.array(files)
    labels = np.array(labels)
    video_ids = np.array(video_ids)

    fold_models = []
    fold_metrics = []
    all_results = {}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(files, labels)):
        model, metrics, val_results, history = train_fold(
            fold_idx,
            files[train_idx].tolist(), labels[train_idx].tolist(), video_ids[train_idx].tolist(),
            files[val_idx].tolist(), labels[val_idx].tolist(), video_ids[val_idx].tolist(),
            args, device,
        )
        fold_models.append(model)
        fold_metrics.append(metrics)

        # Store per-video predictions
        for vid_id, pred in val_results["video_preds"].items():
            all_results[vid_id] = {
                "prediction": pred,
                "label": val_results["video_labels"][vid_id],
                "fold": fold_idx,
            }

    # ─── Aggregate results across folds ───
    print(f"\n{'='*70}")
    print("  CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")

    metric_names = ["auc", "sensitivity", "specificity", "f1", "accuracy", "false_negative_rate"]
    for metric in metric_names:
        values = [m[metric] for m in fold_metrics]
        print(f"  {metric:>25s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    # ─── Risk tier analysis ───
    print(f"\n  Risk Tier Breakdown (aggregated):")
    total_risk = defaultdict(int)
    for m in fold_metrics:
        for tier, count in m["risk_breakdown"].items():
            total_risk[tier] += count
    for tier, count in total_risk.items():
        print(f"    {tier}: {count}")

    # Save overall results
    results_path = os.path.join(config.RESULTS_DIR, "training_results.json")
    summary = {
        "n_videos": len(files),
        "n_asd": int(sum(labels)),
        "n_td": int(len(labels) - sum(labels)),
        "n_folds": args.folds,
        "n_epochs": args.epochs,
        "fold_metrics": fold_metrics,
        "mean_metrics": {
            metric: {
                "mean": float(np.mean([m[metric] for m in fold_metrics])),
                "std": float(np.std([m[metric] for m in fold_metrics])),
            }
            for metric in metric_names
        },
        "per_video_results": all_results,
    }
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  [RESULTS] Saved: {results_path}")

    print(f"\n[DONE] Training complete! Models saved in: {config.SAVED_MODELS_DIR}")
    print("   Run `python evaluate.py` for detailed evaluation plots.")
    print("   Run `python inference.py --video <path>` for single-video inference.")


if __name__ == "__main__":
    main()
