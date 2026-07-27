"""
Plotting and metrics-logging utilities for SFP + LoRA training.

Produces (all saved under a run's output directory):
  - loss_curve.png            : train loss (and val loss, if tracked) vs epoch
  - val_accuracy_curve.png    : validation accuracy vs epoch, best epoch marked
  - combined_curve.png        : loss + accuracy on shared x-axis (twin y-axes)
  - snip_saliency.png         : per-block SNIP saliency bar chart, selected block highlighted
  - param_breakdown.png       : pie chart of parameter counts by category
  - history.csv               : raw per-epoch metrics
  - metrics_summary.json      : final run summary (config + best/final metrics + param counts)
"""

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_lr_schedule(history: dict, save_dir: str, dataset_name: str = "") -> str:
    """
    Plots per-epoch learning rate(s). Expects history to contain "epoch" and one or
    more of "lr_main", "lr_lora".
    """
    ensure_dir(save_dir)
    epochs = history["epoch"]

    plt.figure(figsize=(7, 5))
    if "lr_main" in history:
        plt.plot(epochs, history["lr_main"], label="Main LR (filter block / LN / head)", color="tab:blue")
    if "lr_lora" in history and any(v is not None for v in history["lr_lora"]):
        plt.plot(epochs, history["lr_lora"], label="LoRA LR", color="tab:purple")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(f"LR Schedule{' - ' + dataset_name if dataset_name else ''}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "lr_schedule.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_training_curves(history: dict, save_dir: str, dataset_name: str = "") -> dict:
    """
    Expects history = {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...]}
    val_loss entries may be None if not tracked.
    """
    ensure_dir(save_dir)
    epochs = history["epoch"]
    has_val_loss = "val_loss" in history and any(v is not None for v in history["val_loss"])

    # --- Loss curve ---
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="tab:blue")
    if has_val_loss:
        plt.plot(epochs, history["val_loss"], label="Val Loss", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve{' - ' + dataset_name if dataset_name else ''}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()

    # --- Accuracy curve ---
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["val_acc"], label="Val Accuracy", color="tab:green")
    best_val = max(history["val_acc"])
    best_epoch = epochs[history["val_acc"].index(best_val)]
    plt.scatter([best_epoch], [best_val], color="red", zorder=5,
                label=f"Best: {best_val:.2f}% @ epoch {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Validation Accuracy{' - ' + dataset_name if dataset_name else ''}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(save_dir, "val_accuracy_curve.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()

    # --- Combined dual-axis curve ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(epochs, history["train_loss"], color="tab:blue", label="Train Loss")
    if has_val_loss:
        ax1.plot(epochs, history["val_loss"], color="tab:orange", label="Val Loss")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Val Accuracy (%)", color="tab:green")
    ax2.plot(epochs, history["val_acc"], color="tab:green", label="Val Accuracy")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle(f"Training Curves{' - ' + dataset_name if dataset_name else ''}")
    fig.tight_layout()
    combined_path = os.path.join(save_dir, "combined_curve.png")
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)

    return {"loss_curve": loss_path, "val_accuracy_curve": acc_path, "combined_curve": combined_path}


def plot_snip_saliency(saliencies: dict, selected_idx: int, save_dir: str) -> str:
    ensure_dir(save_dir)
    items = sorted(saliencies.items(), key=lambda kv: kv[0])
    indices = [i for i, _ in items]
    scores = [s for _, s in items]
    colors = ["red" if i == selected_idx else "tab:blue" for i in indices]

    plt.figure(figsize=(9, 5))
    plt.bar(indices, scores, color=colors)
    plt.xlabel("Block Index")
    plt.ylabel("SNIP Saliency Score")
    plt.title(f"Per-Block SNIP Saliency (selected block {selected_idx} shown in red)")
    plt.xticks(indices)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir, "snip_saliency.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_param_breakdown(param_counts: dict, save_dir: str) -> str:
    """
    param_counts: dict of category -> count, e.g.
      {"filter_block": ..., "lora": ..., "layernorm": ..., "head": ..., "frozen_backbone": ...}
    Categories with zero count are dropped from the chart.
    """
    ensure_dir(save_dir)
    filtered = {k: v for k, v in param_counts.items() if v > 0}
    labels = list(filtered.keys())
    values = list(filtered.values())
    total = sum(values)

    def autopct_fmt(pct):
        count = int(round(pct / 100.0 * total))
        return f"{pct:.1f}%\n({count:,})"

    plt.figure(figsize=(7, 7))
    plt.pie(values, labels=labels, autopct=autopct_fmt, startangle=90)
    plt.title("Parameter Breakdown by Category")
    plt.tight_layout()
    path = os.path.join(save_dir, "param_breakdown.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def save_history_csv(history: dict, save_dir: str) -> str:
    ensure_dir(save_dir)
    path = os.path.join(save_dir, "history.csv")
    keys = list(history.keys())
    n = len(history["epoch"])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([history[k][i] for k in keys])
    return path


def save_metrics_summary(summary: dict, save_dir: str) -> str:
    ensure_dir(save_dir)
    path = os.path.join(save_dir, "metrics_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path