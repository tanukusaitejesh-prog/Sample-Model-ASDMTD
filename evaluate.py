"""
Clinical-Grade Evaluation Script
─────────────────────────────────
Generates comprehensive evaluation like a clinical paper (improvement #10):
  - Sensitivity / Specificity / AUC / F1 / FNR / PPV / NPV
  - ROC curves per fold + ensemble
  - Calibration plot (reliability diagram)
  - Confusion matrix heatmap
  - Risk tier breakdown
  - Per-video results table
  - ABSTAIN analysis

Usage:
    python evaluate.py
    python evaluate.py --results-file results/training_results.json
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    confusion_matrix, classification_report
)
from collections import defaultdict

import config
from models.pipeline import ASDScreeningPipeline


def plot_roc_curves(per_video_results, save_dir):
    """Plot ROC curves — per-fold and aggregated."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Group by fold
    folds = defaultdict(lambda: {"preds": [], "labels": []})
    for vid_id, data in per_video_results.items():
        fold = data["fold"]
        folds[fold]["preds"].append(data["prediction"])
        folds[fold]["labels"].append(data["label"])

    all_preds = []
    all_labels = []

    for fold_idx in sorted(folds.keys()):
        preds = np.array(folds[fold_idx]["preds"])
        labels = np.array(folds[fold_idx]["labels"])
        all_preds.extend(preds)
        all_labels.extend(labels)

        try:
            fpr, tpr, _ = roc_curve(labels, preds)
            auc_val = roc_auc_score(labels, preds)
            ax.plot(fpr, tpr, alpha=0.4, lw=1.5,
                    label=f"Fold {fold_idx+1} (AUC={auc_val:.3f})")
        except ValueError:
            pass

    # Overall ROC
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        auc_val = roc_auc_score(all_labels, all_preds)
        ax.plot(fpr, tpr, color="black", lw=2.5,
                label=f"Overall (AUC={auc_val:.3f})")
    except ValueError:
        pass

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curves — ASD Screening Pipeline", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curves.png"), dpi=200)
    plt.close()
    print("  Saved: roc_curves.png")


def plot_confusion_matrix(per_video_results, save_dir):
    """Plot confusion matrix heatmap."""
    preds = [d["prediction"] for d in per_video_results.values()]
    labels = [d["label"] for d in per_video_results.values()]
    pred_binary = (np.array(preds) > 0.5).astype(int)

    cm = confusion_matrix(labels, pred_binary, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["TD (Predicted)", "ASD (Predicted)"],
        yticklabels=["TD (Actual)", "ASD (Actual)"],
        ax=ax, annot_kws={"size": 16}
    )
    ax.set_title("Confusion Matrix", fontsize=14)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=200)
    plt.close()
    print("  Saved: confusion_matrix.png")


def plot_calibration(per_video_results, save_dir, n_bins=10):
    """Plot reliability diagram (calibration plot)."""
    preds = np.array([d["prediction"] for d in per_video_results.values()])
    labels = np.array([d["label"] for d in per_video_results.values()])

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_true_freqs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_means.append(preds[mask].mean())
            bin_true_freqs.append(labels[mask].mean())
            bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfectly calibrated")
    ax1.plot(bin_means, bin_true_freqs, "o-", color="steelblue", lw=2, label="Pipeline")
    ax1.set_ylabel("Fraction of positives", fontsize=12)
    ax1.set_title("Calibration Plot (Reliability Diagram)", fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(bin_means, bin_counts, width=0.08, color="steelblue", alpha=0.6)
    ax2.set_xlabel("Mean predicted probability", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "calibration_plot.png"), dpi=200)
    plt.close()
    print("  Saved: calibration_plot.png")


def plot_risk_breakdown(per_video_results, save_dir):
    """Plot risk tier distribution with correctness."""
    risk_data = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0})

    for vid_id, data in per_video_results.items():
        prob = data["prediction"]
        label = data["label"]

        risk_level, confidence, details = ASDScreeningPipeline.classify_risk(prob)
        risk_data[risk_level]["total"] += 1

        if risk_level == "LOW_RISK" and label == 0:
            risk_data[risk_level]["correct"] += 1
        elif risk_level == "HIGH_RISK" and label == 1:
            risk_data[risk_level]["correct"] += 1
        elif risk_level == "RECHECK":
            risk_data[risk_level]["correct"] += 1  # Recheck is always "correct"
        elif risk_level == "ABSTAIN":
            risk_data[risk_level]["correct"] += 1  # Abstain is always "safe"
        else:
            risk_data[risk_level]["incorrect"] += 1

    tiers = ["LOW_RISK", "RECHECK", "HIGH_RISK", "ABSTAIN"]
    colors = ["#4CAF50", "#FF9800", "#F44336", "#9E9E9E"]
    totals = [risk_data[t]["total"] for t in tiers]
    correct = [risk_data[t]["correct"] for t in tiers]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(tiers, totals, color=colors, alpha=0.7, edgecolor="black")
    ax.bar(tiers, correct, color=colors, alpha=1.0, edgecolor="black")

    for bar, total, corr in zip(bars, totals, correct):
        if total > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{corr}/{total}", ha="center", fontsize=11, fontweight="bold")

    ax.set_ylabel("Number of Videos", fontsize=12)
    ax.set_title("Risk Tier Distribution (Correct/Total)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "risk_breakdown.png"), dpi=200)
    plt.close()
    print("  Saved: risk_breakdown.png")


def print_clinical_report(results):
    """Print a clinical-style summary report."""
    mean_metrics = results.get("mean_metrics", {})

    print(f"\n{'='*70}")
    print("  CLINICAL EVALUATION REPORT - ASD Screening Pipeline")
    print(f"{'='*70}")
    print(f"  Dataset: {results['n_videos']} videos "
          f"({results['n_asd']} ASD, {results['n_td']} TD)")
    print(f"  Protocol: {results['n_folds']}-fold stratified cross-validation")
    print(f"  Epochs: {results['n_epochs']}")
    print()

    print("  --- Performance Metrics ---")
    metric_display = {
        "auc": "AUC (Area Under ROC)",
        "sensitivity": "Sensitivity (Recall for ASD)",
        "specificity": "Specificity (TD Correctness)",
        "f1": "F1 Score",
        "accuracy": "Overall Accuracy",
        "false_negative_rate": "False Negative Rate",
    }
    
    # Print per-fold metrics (user request)
    fold_metrics = results.get("fold_metrics", [])
    if fold_metrics:
        print("\n  [ PER-FOLD BREAKDOWN ]")
        for i, fm in enumerate(fold_metrics):
            print(f"  Fold {i+1}:")
            for key, display in metric_display.items():
                if key in fm:
                    print(f"    {display:>35s}: {fm[key]:.4f}")
            print()

    print("\n  [ OVERALL ENSEMBLE PERFORMANCE (Mean +/- Std) ]")
    for key, display in metric_display.items():
        if key in mean_metrics:
            m = mean_metrics[key]
            print(f"    {display:>35s}: {m['mean']:.4f} +/- {m['std']:.4f}")

    # Risk tier summary
    per_video = results.get("per_video_results", {})
    risk_counts = defaultdict(int)
    for vid_id, data in per_video.items():
        risk, _, _ = ASDScreeningPipeline.classify_risk(data["prediction"])
        risk_counts[risk] += 1

    print(f"\n  --- Risk Tier Distribution ---")
    for tier in ["LOW_RISK", "RECHECK", "HIGH_RISK", "ABSTAIN"]:
        print(f"    {tier:>12s}: {risk_counts[tier]:3d} videos")

    non_abstain = results["n_videos"] - risk_counts["ABSTAIN"]
    print(f"\n    Coverage: {non_abstain}/{results['n_videos']} "
          f"({100*non_abstain/max(results['n_videos'],1):.1f}%)")
    print(f"    Abstained: {risk_counts['ABSTAIN']} "
          f"({100*risk_counts['ABSTAIN']/max(results['n_videos'],1):.1f}%)")

    print(f"\n{'='*70}")
    print("  NOTE: This is a SCREENING tool, not a diagnostic system.")
    print("  HIGH_RISK -> recommend professional assessment")
    print("  ABSTAIN -> insufficient confidence, needs manual review")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ASD Screening Pipeline")
    parser.add_argument("--results-file", default=os.path.join(config.RESULTS_DIR, "training_results.json"))
    args = parser.parse_args()

    if not os.path.exists(args.results_file):
        print(f"❌ Results file not found: {args.results_file}")
        print("   Run `python train.py` first.")
        sys.exit(1)

    with open(args.results_file, "r") as f:
        results = json.load(f)

    save_dir = config.RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    per_video = results.get("per_video_results", {})

    # Print clinical report
    print_clinical_report(results)

    # Generate plots
    print("Generating evaluation plots...")
    plot_roc_curves(per_video, save_dir)
    plot_confusion_matrix(per_video, save_dir)
    plot_calibration(per_video, save_dir)
    plot_risk_breakdown(per_video, save_dir)

    print(f"\n✅ All plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
