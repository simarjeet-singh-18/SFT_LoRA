"""
Diagnostic: measures how much each LoRA adapter actually learned, by comparing the
magnitude of its effective weight delta (scaling * B @ A) against the magnitude of
the frozen base layer weight it's attached to.

Answers: "did the LoRA adapters wake up and learn something substantial during
training, or did they stay close to their zero-initialized starting point?"
This disambiguates two very different explanations for LoRA not improving accuracy:
  - tiny deltas  -> adapters never really trained (short schedule / low effective LR
    for the LoRA path / undertrained), not a lack-of-capacity-being-useful problem
  - large deltas -> adapters ARE learning substantial updates, so if accuracy still
    isn't improving, the issue is more likely overfitting / wrong placement / the
    filter block already saturating what's learnable from this data, not "LoRA
    failed to train at all"

Usage:
    python3 lora_delta_diagnostic.py <run_output_dir>

<run_output_dir> must be the output directory of a --lora-rank > 0 training run,
containing metrics_summary.json and the saved best-checkpoint .pt file. Reuses
substitute_filter_block() / inject_lora() / freeze_non_trainable() from
single_filter_lora.py to reconstruct the exact same wrapped architecture that was
trained (supports single-block and multi-block --num-filter-blocks runs,
--filter-block-layers 1 and >1, and --filter-residual-hidden-dim 0 and >0), so the
checkpoint loads correctly.

Handles both current metrics_summary.json format (pruned_block_idx as a list) and
older formats (pruned_block_idx as a single int, filter_block_layers /
filter_residual_hidden_dim absent -> default to 1 / 0 respectively, matching what
existed before those fields were added).

ASSUMPTION: reconstructs using the default target_keywords ["qkv", "proj", "fc1",
"fc2"] -- this matches every run produced by the current train_sfp_lora.py (that
parameter isn't exposed as a CLI flag / recorded in metrics_summary.json, so this
script assumes it wasn't changed between training and now).
"""

import argparse
import json
import os

import timm
import torch

from single_filter_lora import LoRALinear, substitute_filter_block, inject_lora, freeze_non_trainable


def load_run_model(output_dir: str, device: str = "cpu"):
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"metrics_summary.json not found in {output_dir}")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    if summary.get("full_finetune"):
        raise ValueError("This run used --full-finetune -- there are no LoRA modules to inspect.")

    lora_rank = summary.get("lora_rank")
    if not lora_rank or lora_rank <= 0:
        raise ValueError(f"This run used lora_rank={lora_rank} -- there are no LoRA modules to inspect.")

    # pruned_block_idx is a list in current metrics_summary.json (multi-block support),
    # but was a single int in older summaries -- normalize to a list either way.
    raw_pruned = summary["pruned_block_idx"]
    pruned_block_indices = [raw_pruned] if isinstance(raw_pruned, int) else list(raw_pruned)

    # filter_block_layers wasn't recorded in older summaries -- default to 1
    # (the only option that existed before this field was added).
    filter_block_layers = summary.get("filter_block_layers", 1)

    # filter_residual_hidden_dim wasn't recorded in older summaries either --
    # default to 0 (no residual branch, the only option that existed before this
    # field was added). Must match what was actually trained with, or
    # load_state_dict below will fail (missing/unexpected keys for the residual
    # branch's fc1/fc2 params).
    filter_residual_hidden_dim = summary.get("filter_residual_hidden_dim", 0) or 0
    filter_residual_alpha = summary.get("filter_residual_alpha") or 1.0
    filter_residual_dropout = summary.get("filter_residual_dropout") or 0.0

    lora_alpha = summary.get("lora_alpha", 32.0)
    lora_dropout = summary.get("lora_dropout", 0.0) or 0.0
    num_classes = summary["num_classes"]
    checkpoint_path = summary.get("checkpoint_path") or os.path.join(
        output_dir, f"best_sfp_lora_{summary['dataset']}.pt"
    )
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Diagnostic] Rebuilding architecture: pruned_block_indices={pruned_block_indices}, "
          f"filter_block_layers={filter_block_layers}, "
          f"filter_residual_hidden_dim={filter_residual_hidden_dim}, "
          f"lora_rank={lora_rank}, lora_alpha={lora_alpha}, "
          f"lora_dropout={lora_dropout}, num_classes={num_classes}")

    # pretrained=False is safe AND faster here: every weight gets immediately
    # overwritten by the checkpoint's state_dict below, so the initial pretrained
    # weights are never actually used -- no need to download them.
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)

    # Reconstruct the architecture: substitute each filter block (no pinv init needed
    # here -- the checkpoint's state_dict below overwrites all these weights anyway,
    # we just need matching shapes/parameter names), then inject LoRA + freeze.
    for idx in pruned_block_indices:
        substitute_filter_block(
            model, idx, num_layers=filter_block_layers,
            residual_hidden_dim=filter_residual_hidden_dim,
            residual_alpha=filter_residual_alpha, residual_dropout=filter_residual_dropout,
        )
    inject_lora(model, pruned_block_indices, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    freeze_non_trainable(model, pruned_block_indices)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, summary


def compute_lora_delta_report(model: torch.nn.Module):
    """
    For every LoRALinear module with rank > 0, computes:
      - delta_norm: Frobenius norm of the effective LoRA update (scaling * B @ A)
      - base_norm : Frobenius norm of the frozen base layer's weight
      - ratio_pct : delta_norm / base_norm, as a percentage -- how large the LoRA
        update is relative to the weight matrix it's modifying
    """
    rows = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.rank > 0:
            with torch.no_grad():
                delta = module.scaling * (module.lora_B @ module.lora_A)  # (out_features, in_features)
                delta_norm = delta.norm().item()
                base_norm = module.base_layer.weight.norm().item()
                ratio_pct = 100.0 * delta_norm / base_norm if base_norm > 0 else float("nan")
            rows.append({
                "layer": name,
                "delta_norm": delta_norm,
                "base_norm": base_norm,
                "ratio_pct": ratio_pct,
            })
    return rows


def print_report(rows: list):
    if not rows:
        print("[Diagnostic] No LoRA modules found in this model.")
        return

    print(f"\n{'Layer':<45} {'||delta||':>12} {'||base||':>12} {'delta/base %':>14}")
    print("-" * 85)
    for r in sorted(rows, key=lambda r: -r["ratio_pct"]):
        print(f"{r['layer']:<45} {r['delta_norm']:>12.4f} {r['base_norm']:>12.4f} {r['ratio_pct']:>13.2f}%")

    ratios = [r["ratio_pct"] for r in rows]
    mean_ratio = sum(ratios) / len(ratios)
    max_ratio = max(ratios)
    min_ratio = min(ratios)
    print("-" * 85)
    print(f"{'MEAN':<45} {'':>12} {'':>12} {mean_ratio:>13.2f}%")
    print(f"Min: {min_ratio:.2f}% | Max: {max_ratio:.2f}% | N layers: {len(rows)}")

    print()
    if mean_ratio < 1.0:
        print("[Diagnostic] Interpretation: LoRA deltas are TINY (<1% of base weight norm, "
              "on average). The adapters barely moved from their zero-init starting point --\n"
              "they likely never really 'woke up' enough to learn anything substantial. This "
              "points toward under-training (too-short a schedule, or too-low an effective LR\n"
              "for the LoRA path) as the main bottleneck, rather than the adapters having "
              "learned unhelpful features.")
    elif mean_ratio < 5.0:
        print("[Diagnostic] Interpretation: LoRA deltas are small-to-moderate (1-5% of base "
              "weight norm). Some real learning happened, but the adapters are still operating\n"
              "as a fairly gentle perturbation on top of the frozen backbone.")
    else:
        print("[Diagnostic] Interpretation: LoRA deltas are substantial (>5% of base weight "
              "norm). The adapters ARE learning meaningfully-sized updates -- if accuracy still\n"
              "isn't improving despite this, the issue is more likely that this capacity isn't "
              "being used productively (overfitting / placement / the filter block already\n"
              "saturating what's learnable from this data), not that LoRA failed to train at all.")


def main():
    parser = argparse.ArgumentParser(
        description="Measure how much each LoRA adapter actually learned (delta magnitude vs base weight norm)."
    )
    parser.add_argument("run_output_dir", type=str,
                         help="Output dir of a --lora-rank > 0 training run (must contain "
                              "metrics_summary.json and the saved checkpoint).")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model, summary = load_run_model(args.run_output_dir, device=args.device)
    print(f"[Diagnostic] Loaded checkpoint for dataset={summary.get('dataset')}, "
          f"best_val_acc={summary.get('best_val_acc')}, final_test_acc={summary.get('final_test_acc')}")

    rows = compute_lora_delta_report(model)
    print_report(rows)


if __name__ == "__main__":
    main()