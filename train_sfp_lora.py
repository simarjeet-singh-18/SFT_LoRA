import argparse
import csv
import os

# Must be set BEFORE torch is imported / any CUDA context is created. Without this,
# cuBLAS matmul kernels (Linear layers, attention, AdamW internals) can still use
# non-deterministic reduction order even with cudnn.deterministic=True, since that
# flag only governs cuDNN convolution algorithm selection, not cuBLAS.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import timm

from data import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from single_filter_lora import (
    apply_single_filter_and_lora,
    count_parameter_breakdown,
    substitute_filter_block,
    inject_lora,
    freeze_non_trainable,
    compute_lora_orthogonality_loss,
)
from snip_selection import select_block_with_snip, select_blocks_with_snip
from run_naming import build_run_folder_name
from plotting import (
    ensure_dir,
    plot_training_curves,
    plot_snip_saliency,
    plot_param_breakdown,
    plot_lr_schedule,
    save_history_csv,
    save_metrics_summary,
)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seeds every RNG the pipeline touches (python random, numpy, torch CPU/CUDA),
    for reproducible data splits, model/LoRA init, and DataLoader shuffle order.

    deterministic=True additionally:
      - enables full cuDNN determinism (fixed conv algorithm, no autotuning)
      - calls torch.use_deterministic_algorithms(True), which forces deterministic
        implementations across ALL of torch (not just cuDNN conv) -- this is what
        actually pins down cuBLAS matmul / AdamW kernel behavior, which
        cudnn.deterministic alone does NOT cover
      - requires CUBLAS_WORKSPACE_CONFIG to have been set before torch was imported
        (done at the top of this file) for the cuBLAS part to take effect
      - forces PyTorch's scaled_dot_product_attention (used internally by timm's ViT
        attention blocks) onto its "math" backend, disabling the flash-attention and
        memory-efficient-attention fused kernels. Those fused kernels use atomic-add
        based reductions in their backward pass that are NOT deterministic even under
        use_deterministic_algorithms(True) -- this was the actual remaining source of
        run-to-run divergence observed even with the other determinism settings on
        (visible as tiny ~1e-6 relative differences in SNIP saliency scores between
        otherwise-identical runs, which then compound over training). The math
        backend is slower but fully deterministic.

    warn_only=True means any OTHER operation without a deterministic implementation
    will print a warning and fall back to its normal (possibly non-deterministic)
    kernel, rather than raising and killing the run. Check the console for such
    warnings if you still see run-to-run variance with deterministic=True -- they'll
    name the exact op that's still non-deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

        # Disable non-deterministic fused attention kernels (flash / mem-efficient),
        # forcing the deterministic "math" SDPA backend. Guarded with hasattr since
        # these toggles were added in newer torch versions.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    else:
        torch.backends.cudnn.benchmark = True

    print(f"[Seed] Global seed set to {seed} (deterministic={deterministic}).")


def extract_block_inputs_outputs(model: nn.Module, dataloader: DataLoader, block_idx: int, device: str):
    """
    Hooks block inputs/outputs to initialize single filter block via Ridge Pseudo-Inverse.
    """
    model.eval()
    inputs_list, outputs_list = [], []

    def hook_fn(module, input_tensor, output_tensor):
        inputs_list.append(input_tensor[0].detach().cpu())
        outputs_list.append(output_tensor.detach().cpu())

    hook_handle = model.blocks[block_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            x = batch[0].to(device)  # Handle datasets returning tuples/lists
            _ = model(x)
            if i >= 10:  # Collect 10 batches for stable pseudo-inverse fit
                break

    hook_handle.remove()
    return torch.cat(inputs_list, dim=0), torch.cat(outputs_list, dim=0)


def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            preds = model(x).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return (correct / total) * 100.0


def evaluate_full(model: nn.Module, dataloader: DataLoader, device: str, criterion: nn.Module):
    """
    Like evaluate(), but also returns average loss so we can plot a val-loss curve
    alongside train loss.
    """
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            preds = out.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    avg_loss = loss_sum / total
    acc = (correct / total) * 100.0
    return avg_loss, acc


def denormalize(tensor: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """Undoes transforms.Normalize so images can be saved as viewable PNGs."""
    mean_t = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    return (tensor * std_t + mean_t).clamp(0, 1)


def evaluate_and_save_misclassified(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    criterion: nn.Module,
    output_dir: str,
    class_names: list = None,
    mean: list = None,
    std: list = None,
    max_images: int = 200,
):
    """
    Same as evaluate_full (loss + accuracy over dataloader), but additionally saves
    every misclassified image as a PNG under <output_dir>/misclassified/, plus a CSV
    log (misclassified_log.csv) with true/predicted labels and confidence.

    max_images caps how many images get saved (<=0 means unlimited) to avoid dumping
    thousands of files on large test sets; loss/accuracy are still computed over the
    FULL dataset regardless of the cap. Mode-agnostic: works identically whether the
    model came from SFP, SFP+LoRA, or --full-finetune, since it only touches the
    final test-evaluation step, which is already shared across all three.
    """
    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    misclassified_dir = ensure_dir(os.path.join(output_dir, "misclassified"))
    csv_path = os.path.join(misclassified_dir, "misclassified_log.csv")

    model.eval()
    correct, total, loss_sum, saved_count, global_idx = 0, 0, 0.0, 0, 0

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "sample_index", "true_label_idx", "true_label_name",
                          "pred_label_idx", "pred_label_name", "confidence"])

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x)
                loss = criterion(out, y)
                loss_sum += loss.item() * x.size(0)

                probs = torch.softmax(out, dim=-1)
                confs, preds = probs.max(dim=-1)
                correct += (preds == y).sum().item()
                total += y.size(0)

                mismatches = (preds != y).nonzero(as_tuple=True)[0]
                if mismatches.numel() > 0:
                    imgs_denorm = denormalize(x[mismatches].detach(), mean, std).cpu()
                    for local_i, sample_i in enumerate(mismatches.tolist()):
                        if max_images > 0 and saved_count >= max_images:
                            continue
                        true_idx = int(y[sample_i].item())
                        pred_idx = int(preds[sample_i].item())
                        conf = float(confs[sample_i].item())
                        true_name = str(class_names[true_idx]) if class_names else str(true_idx)
                        pred_name = str(class_names[pred_idx]) if class_names else str(pred_idx)

                        safe_true = true_name.replace("/", "-").replace(" ", "_")
                        safe_pred = pred_name.replace("/", "-").replace(" ", "_")
                        fname = f"idx{global_idx + sample_i:05d}_true-{safe_true}_pred-{safe_pred}_conf{conf:.2f}.png"

                        save_image(imgs_denorm[local_i], os.path.join(misclassified_dir, fname))
                        writer.writerow([fname, global_idx + sample_i, true_idx, true_name,
                                          pred_idx, pred_name, f"{conf:.4f}"])
                        saved_count += 1

                global_idx += y.size(0)

    avg_loss = loss_sum / total
    acc = (correct / total) * 100.0
    total_misclassified = total - correct
    print(f"[SFP] Misclassified images: saved {saved_count} / {total_misclassified} total "
          f"misclassified on test set -> {misclassified_dir}")
    if max_images > 0 and total_misclassified > max_images:
        print(f"[SFP] Note: capped at --max-misclassified-images={max_images}; "
              f"{total_misclassified - saved_count} additional misclassified samples were not saved.")
    print(f"[SFP] Misclassified log CSV -> {csv_path}")

    return avg_loss, acc, saved_count, csv_path, misclassified_dir


def main():
    parser = argparse.ArgumentParser(description="SFP Single Filter + LoRA Fine-Tuning")
    parser.add_argument("--dataset", type=str, default="pets",
                         choices=["pets", "svhn", "flowers102", "dtd", "caltech101", "cifar100",
                                  "pcam", "clevr", "dsprites-loc", "dsprites-ori"])
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--use-full-dataset", action="store_true")
    parser.add_argument("--pruned-block", type=int, default=-1,
                         help="-1 runs SNIP auto-search. Only valid when --num-filter-blocks 1 "
                              "(manually specifying a fixed set of multiple blocks isn't supported "
                              "yet -- use SNIP auto-search via --num-filter-blocks for that).")
    parser.add_argument("--num-filter-blocks", type=int, default=1,
                         help="How many blocks (top-N by LOWEST SNIP saliency) to replace with filter "
                              "blocks. 1 = original single-block SFP behavior. N>1 replaces the N "
                              "least-salient blocks sequentially (each one's pseudoinverse init is "
                              "computed AFTER earlier substitutions in the set, so later filter blocks "
                              "correctly learn from the already-modified preceding representations). "
                              "LoRA (if --lora-rank > 0) is injected into every block NOT in this set. "
                              "Caution: the paper's own ablation found 2-block substitution slightly "
                              "underperforming 1-block on their subtractive-only method -- more filter "
                              "blocks isn't automatically better, treat this as a real hyperparameter.")
    parser.add_argument("--filter-block-layers", type=int, default=1,
                         help="Number of stacked FC layers per filter block (default 1 = paper's "
                              "original single-layer construction). N>1 uses a generalized "
                              "pseudoinverse init: N-1 layers start as identity, 1 layer gets the "
                              "solved pseudoinverse matrix -- so the WHOLE STACK is mathematically "
                              "identical to the N=1 case at initialization, regardless of depth. NOTE: "
                              "since layers are purely linear (no activation between them), stacking "
                              "them does NOT add expressivity beyond a single layer by itself -- this "
                              "is for experimenting with parameterization/depth, not a capacity boost.")
    parser.add_argument("--filter-residual-hidden-dim", type=int, default=0,
                         help="Hidden dim of an OPTIONAL nonlinear zero-init residual branch attached "
                              "to each filter block: fc2(GELU(fc1(x))), with fc2 zero-initialized so "
                              "the branch contributes exactly 0 at init (same safe-start trick as "
                              "LoRA's zero-init B matrix) -- doesn't disturb the pseudoinverse-inherited "
                              "behavior. 0 (default) = no residual branch (unchanged from all previous "
                              "versions). This IS a genuine capacity boost (has a real nonlinearity), "
                              "unlike --filter-block-layers which is linear-only.")
    parser.add_argument("--filter-residual-alpha", type=float, default=1.0,
                         help="Scaling for the residual branch's output = alpha / hidden_dim (mirrors "
                              "LoRA's alpha/rank normalization). Only used when "
                              "--filter-residual-hidden-dim > 0.")
    parser.add_argument("--filter-residual-dropout", type=float, default=0.0,
                         help="Dropout on the residual branch's output. Only used when "
                              "--filter-residual-hidden-dim > 0.")
    parser.add_argument("--full-finetune", action="store_true",
                         help="Baseline comparison mode: bypass SNIP search, filter block substitution, "
                              "and LoRA entirely, and fine-tune the ENTIRE pretrained backbone + head "
                              "instead (i.e. the paper's 'Full' baseline in Table 1/2). "
                              "--pruned-block, --num-filter-blocks, --filter-block-layers, "
                              "--filter-residual-hidden-dim, --lora-rank, --lora-alpha, --lora-dropout "
                              "are all ignored when this is set.")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                         help="Dropout applied to LoRA adapter input (0.0 disables it). "
                              "Useful regularization when training on small subsets (e.g. VTAB-1k-style splits).")
    parser.add_argument("--lora-ortho-lambda1", type=float, default=0.0,
                         help="Weight (lambda1) on the ||A @ A.T - I||_F^2 orthogonality penalty applied to "
                              "every LoRA lora_A matrix. Pushes each adapter's rank rows to be mutually "
                              "orthonormal, i.e. discourages them from learning linearly-dependent / redundant "
                              "directions. 0.0 (default) disables it entirely -- identical behavior to before "
                              "this flag existed.")
    parser.add_argument("--lora-ortho-lambda2", type=float, default=0.0,
                         help="Weight (lambda2) on the ||B.T @ B - I||_F^2 orthogonality penalty applied to "
                              "every LoRA lora_B matrix (columns instead of rows, mirroring lambda1 for A). "
                              "0.0 (default) disables it.")
    parser.add_argument("--lr", type=float, default=1e-3, help="LR for filter block, LayerNorm, and head")
    parser.add_argument("--lora-lr", type=float, default=3e-4, help="LR for LoRA adapter params (A/B matrices)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=-1,
                         help="Number of training epochs. Set to -1 to enable early stopping "
                              "instead of a fixed epoch count (applies identically to all three "
                              "modes: --full-finetune, SFT-only, SFT+LoRA) -- training then runs "
                              "up to --max-epochs, stopping early if val accuracy hasn't improved "
                              "for --patience consecutive epochs.")
    parser.add_argument("--patience", type=int, default=10,
                         help="Early stopping patience (epochs without val-acc improvement before "
                              "stopping). Only used when --epochs -1.")
    parser.add_argument("--max-epochs", type=int, default=200,
                         help="Safety ceiling on total epochs when --epochs -1 (early stopping "
                              "mode). Also used as the cosine LR schedule's horizon in that mode, "
                              "since the actual stopping epoch isn't known in advance. Ignored "
                              "when --epochs is set to a positive value.")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                         help="Linear LR warmup epochs before cosine decay begins. 0 disables warmup.")
    parser.add_argument("--min-lr-ratio", type=float, default=0.0,
                         help="Cosine decay floor as a fraction of each group's peak LR (e.g. 0.01 = decay to 1% of peak).")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                         help="Max gradient norm for clipping (0.0 disables clipping). Cheap safety net "
                              "against LR spikes destabilizing training, especially in --full-finetune mode.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed (data splits, model init, "
                                                               "LoRA init, DataLoader shuffle order).")
    parser.add_argument("--fast", action="store_true",
                         help="Disable full determinism for faster training (enables cuDNN autotuning "
                              "and allows non-deterministic algorithms). Reproducibility is ON by default "
                              "(RNG seeding + cuDNN determinism + torch.use_deterministic_algorithms); "
                              "pass --fast only if you don't need bit-exact reproducibility and want speed.")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                         help="Root directory for plots, CSV history, and metrics summary. "
                              "A per-dataset subfolder is created automatically.")
    parser.add_argument("--save-misclassified-images", action="store_true",
                         help="Save every misclassified test-set image (denormalized PNG) into "
                              "<output_dir>/misclassified/, plus a CSV log of true/predicted labels "
                              "and confidence. Works identically for SFP, SFP+LoRA, and --full-finetune.")
    parser.add_argument("--max-misclassified-images", type=int, default=-1,
                         help="Cap on how many misclassified images to save to disk (<=0 = unlimited). "
                              "Test accuracy/loss are still computed over the full test set regardless. "
                              "Default is unlimited so misclassified_log.csv is always complete -- capping "
                              "would make comparisons across runs ambiguous (can't tell 'correct' apart "
                              "from 'wrong but not logged').")
    args = parser.parse_args()

    if args.epochs != -1 and args.epochs <= 0:
        parser.error("--epochs must be a positive integer, or exactly -1 to enable early stopping.")
    if args.patience <= 0:
        parser.error("--patience must be a positive integer.")
    if args.num_filter_blocks < 1:
        parser.error("--num-filter-blocks must be >= 1.")
    if args.filter_block_layers < 1:
        parser.error("--filter-block-layers must be >= 1.")
    if args.filter_residual_hidden_dim < 0:
        parser.error("--filter-residual-hidden-dim must be >= 0 (0 disables the residual branch).")
    if args.num_filter_blocks > 1 and args.pruned_block != -1:
        parser.error("--pruned-block (a single manual block index) can't be combined with "
                      "--num-filter-blocks > 1 (multi-block SNIP auto-selection). Leave "
                      "--pruned-block at its default (-1) when using --num-filter-blocks > 1.")

    early_stopping_enabled = (args.epochs == -1)
    effective_epochs = args.max_epochs if early_stopping_enabled else args.epochs

    # The --lr default (1e-3) was tuned for SFP's tiny filter block + LN + head
    # (~0.6-2.8M params). Applied to the ENTIRE pretrained backbone in --full-finetune
    # mode, it's aggressive enough to cause a destructive/catastrophic-forgetting step
    # once warmup ramps up to peak LR (visible as a train-loss spike right as warmup
    # ends). Auto-lower it for full-finetune runs, but only if the user didn't
    # explicitly pass --lr themselves.
    lr_passed_explicitly = any(tok == "--lr" or tok.startswith("--lr=") for tok in sys.argv[1:])
    if args.full_finetune and not lr_passed_explicitly:
        old_lr = args.lr
        args.lr = 1e-4
        print(f"[SFP] --full-finetune set without an explicit --lr: lowering default LR "
              f"from {old_lr} to {args.lr} (the {old_lr} default was tuned for the tiny "
              f"SFP filter block, not full-backbone fine-tuning). Pass --lr explicitly to override.")

    set_seed(args.seed, deterministic=not args.fast)

    run_name = build_run_folder_name(sys.argv[1:])
    output_dir = ensure_dir(os.path.join(args.output_dir, run_name))
    print(f"[SFP] Run folder name (from CLI args passed): {run_name}")
    print(f"[SFP] Outputs (plots, CSV, metrics JSON) will be saved to: {output_dir}")

    # 1. Load Data
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(args)

    # 2. Load Pretrained Backbone
    print(f"[SFP] Loading ViT backbone for {num_classes} output classes...")
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    model.to(args.device)

    # 3. Either: (a) SFP path - SNIP search -> pseudoinverse filter block(s) -> optional LoRA, or
    #    (b)  Full-finetune baseline - skip all of that, unfreeze the entire model.
    pruned_block_indices = None   # list of ints once determined (None only for full-finetune)
    snip_saliencies = None
    snip_plot_path = None
    filter_blocks = []

    if args.full_finetune:
        print("[SFP] --full-finetune set: skipping SNIP search, filter block substitution, "
              "and LoRA injection. The ENTIRE backbone + head will be trained "
              "(this is the paper's 'Full' fine-tuning baseline).")
        for p in model.parameters():
            p.requires_grad = True

    elif args.num_filter_blocks == 1:
        # Single-block path: identical behavior/logging to all previous versions.
        pruned_block_idx = args.pruned_block
        if pruned_block_idx < 0:
            print("[SFP] No block index provided. Running SNIP search...")
            pruned_block_idx, snip_saliencies = select_block_with_snip(
                model, train_loader, device=args.device, keep="low", return_scores=True
            )
            snip_plot_path = plot_snip_saliency(snip_saliencies, pruned_block_idx, output_dir)
            print(f"[SFP] Saved SNIP saliency plot -> {snip_plot_path}")

        print(f"[SFP] Extracting representations for Pseudo-Inverse Init at block {pruned_block_idx}...")
        X_in, X_out = extract_block_inputs_outputs(model, train_loader, pruned_block_idx, args.device)

        filter_block = apply_single_filter_and_lora(
            model,
            pruned_block_idx=pruned_block_idx,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            filter_num_layers=args.filter_block_layers,
            filter_residual_hidden_dim=args.filter_residual_hidden_dim,
            filter_residual_alpha=args.filter_residual_alpha,
            filter_residual_dropout=args.filter_residual_dropout,
        )
        filter_block.init_from_pinv(X_in.to(args.device), X_out.to(args.device))
        filter_blocks = [filter_block]
        pruned_block_indices = [pruned_block_idx]

    else:
        # Multi-block path (--num-filter-blocks > 1): SNIP auto-selects the N
        # least-salient blocks, substituted SEQUENTIALLY (increasing index order)
        # so each later filter block's pseudoinverse init is computed from the
        # model's already-partially-modified state -- matches the paper's own
        # sequential dual-layer construction (Fig. 3), just generalized to N.
        print(f"[SFP] Running SNIP search for {args.num_filter_blocks} filter block(s)...")
        pruned_block_indices, snip_saliencies = select_blocks_with_snip(
            model, train_loader, device=args.device, num_blocks=args.num_filter_blocks,
            keep="low", return_scores=True
        )
        snip_plot_path = plot_snip_saliency(snip_saliencies, pruned_block_indices, output_dir)
        print(f"[SFP] Saved SNIP saliency plot -> {snip_plot_path}")

        for idx in pruned_block_indices:
            print(f"[SFP] Extracting representations for Pseudo-Inverse Init at block {idx}...")
            X_in, X_out = extract_block_inputs_outputs(model, train_loader, idx, args.device)
            fb = substitute_filter_block(
                model, idx, num_layers=args.filter_block_layers,
                residual_hidden_dim=args.filter_residual_hidden_dim,
                residual_alpha=args.filter_residual_alpha,
                residual_dropout=args.filter_residual_dropout,
            )
            fb.init_from_pinv(X_in.to(args.device), X_out.to(args.device))
            filter_blocks.append(fb)
            layer_desc = "Single" if args.filter_block_layers <= 1 else f"{args.filter_block_layers}-Layer"
            residual_desc = f" + residual(hidden={args.filter_residual_hidden_dim})" if args.filter_residual_hidden_dim > 0 else ""
            print(f"[SFP-MultiFilter] Substituted block {idx} with {layer_desc} Filter Block{residual_desc}.")

        lora_params = inject_lora(model, pruned_block_indices, args.lora_rank, args.lora_alpha, args.lora_dropout)
        ln_params = freeze_non_trainable(model, pruned_block_indices)
        print(f"[SFP-MultiFilter] Injected {lora_params:,} LoRA parameters "
              f"(rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}) "
              f"across all blocks except {pruned_block_indices}.")
        print(f"[SFP-MultiFilter] Unfroze {ln_params:,} LayerNorm parameters across all blocks.")

    # 6. Optimization Loop
    model.to(args.device)

    # Split trainable params into two groups since LoRA adapters and the filter
    # block / LayerNorm / head have very different scales and typically want
    # different learning rates (LoRA is usually tuned lower, e.g. 1e-4 to 3e-4,
    # while the full-rank filter block and LN/head can tolerate a higher LR).
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    main_params = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]

    optimizer = torch.optim.AdamW(
        [
            {"params": main_params, "lr": args.lr},
            {"params": lora_params, "lr": args.lora_lr},
        ],
        weight_decay=args.weight_decay,
    )
    n_main = sum(p.numel() for p in main_params)
    n_lora = sum(p.numel() for p in lora_params)
    print(f"[SFP] Optimizer groups -> main: {n_main:,} params @ lr={args.lr} | lora: {n_lora:,} params @ lr={args.lora_lr}")

    # Cosine LR schedule, matching the paper's CosineLRScheduler + AdamW protocol.
    # Each param group decays from its own peak LR down to min_lr_ratio * peak, over
    # (epochs - warmup_epochs) steps, with an optional linear warmup beforehand.
    # eta_min is set per-group since main_params and lora_params can have different peak LRs.
    warmup_epochs = min(args.warmup_epochs, max(effective_epochs - 1, 0))
    cosine_epochs = max(effective_epochs - warmup_epochs, 1)
    eta_mins = [args.lr * args.min_lr_ratio, args.lora_lr * args.min_lr_ratio]

    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=0.0
    )
    # CosineAnnealingLR ignores per-group eta_min unless passed as a list in newer torch;
    # to stay compatible across torch versions, we manually floor the LR after each step instead.

    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
        )
    else:
        scheduler = cosine_sched

    print(f"[SFP] LR schedule: {warmup_epochs} warmup epoch(s) -> cosine decay over {cosine_epochs} epoch(s), "
          f"min_lr_ratio={args.min_lr_ratio}")

    criterion = nn.CrossEntropyLoss()

    if early_stopping_enabled:
        print(f"[SFP] Starting training on {args.dataset.upper()} with EARLY STOPPING "
              f"(patience={args.patience}, max_epochs={effective_epochs})...")
    else:
        print(f"[SFP] Starting training for {args.epochs} epochs on {args.dataset.upper()}...")
    best_val, best_epoch, best_path = 0.0, 0, os.path.join(output_dir, f"best_sfp_lora_{args.dataset}.pt")
    epochs_without_improvement = 0

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "lr_main": [], "lr_lora": []}
    ortho_enabled = (args.lora_ortho_lambda1 != 0.0) or (args.lora_ortho_lambda2 != 0.0)
    if ortho_enabled:
        history["train_ortho_loss"] = []
        print(f"[SFP] LoRA orthogonality regularization ENABLED: "
              f"lambda1={args.lora_ortho_lambda1} (||A@A.T - I||^2), "
              f"lambda2={args.lora_ortho_lambda2} (||B.T@B - I||^2)")

    for epoch in range(1, effective_epochs + 1):
        model.train()
        running_loss = 0.0
        running_ortho_loss = 0.0
        for batch in train_loader:
            x, y = batch[0].to(args.device), batch[1].to(args.device)
            optimizer.zero_grad()
            out = model(x)
            task_loss = criterion(out, y)
            # L = general_loss + lambda1 * ||A @ A.T - I||_F^2 + lambda2 * ||B.T @ B - I||_F^2,
            # summed over every LoRA adapter in the model. See
            # single_filter_lora.compute_lora_orthogonality_loss / LoRALinear.orthogonality_penalty
            # for why A@A.T / B.T@B (not A.T@A / B@B.T) are the correct shapes here.
            # Returns exactly 0.0 (no graph) when both lambdas are 0, so this is a
            # strict no-op unless the new flags are explicitly passed.
            ortho_loss = compute_lora_orthogonality_loss(
                model, args.lora_ortho_lambda1, args.lora_ortho_lambda2
            )
            loss = task_loss + ortho_loss
            loss.backward()
            if args.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(main_params + lora_params, max_norm=args.grad_clip)
            optimizer.step()
            running_loss += task_loss.item() * x.size(0)
            if ortho_enabled:
                running_ortho_loss += ortho_loss.item() * x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_ortho_loss = running_ortho_loss / len(train_loader.dataset) if ortho_enabled else None
        val_loss, val_acc = evaluate_full(model, val_loader, args.device, criterion)

        # Record current LR *before* stepping the scheduler for this epoch's log line,
        # then step + apply the manual min-LR floor for next epoch.
        lr_main_now = optimizer.param_groups[0]["lr"]
        lr_lora_now = optimizer.param_groups[1]["lr"] if n_lora > 0 else None

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr_main"].append(lr_main_now)
        history["lr_lora"].append(lr_lora_now)
        if ortho_enabled:
            history["train_ortho_loss"].append(epoch_ortho_loss)

        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), best_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        lr_lora_display = f"{lr_lora_now:.2e}" if lr_lora_now is not None else "n/a"
        epoch_display = f"{epoch:03d}/{effective_epochs:03d}" + (" (max)" if early_stopping_enabled else "")
        patience_display = f" | No-improve: {epochs_without_improvement}/{args.patience}" if early_stopping_enabled else ""
        ortho_display = f" | Ortho Loss: {epoch_ortho_loss:.4f}" if ortho_enabled else ""
        print(f"[Epoch {epoch_display}] Train Loss: {epoch_loss:.4f}{ortho_display} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Best Val: {best_val:.2f}%"
              f"{patience_display} | LR(main/lora): {lr_main_now:.2e}/{lr_lora_display}")

        scheduler.step()
        # Manually floor each group's LR at min_lr_ratio * its own peak, since
        # CosineAnnealingLR's built-in eta_min doesn't support per-group floors
        # across all torch versions.
        for group, peak_lr, eta_min in zip(optimizer.param_groups, [args.lr, args.lora_lr], eta_mins):
            if group["lr"] < eta_min:
                group["lr"] = eta_min

        if early_stopping_enabled and epochs_without_improvement >= args.patience:
            print(f"[SFP] Early stopping triggered: no val-acc improvement for {args.patience} "
                  f"epoch(s) (best={best_val:.2f}% @ epoch {best_epoch}). Stopping at epoch {epoch}.")
            break

    # Save curves + raw per-epoch history as soon as training finishes, so they exist
    # even if something later (checkpoint reload, test eval) fails.
    curve_paths = plot_training_curves(history, output_dir, dataset_name=args.dataset)
    lr_plot_path = plot_lr_schedule(history, output_dir, dataset_name=args.dataset)
    csv_path = save_history_csv(history, output_dir)
    print(f"[SFP] Saved training curves -> {curve_paths}")
    print(f"[SFP] Saved LR schedule plot -> {lr_plot_path}")
    print(f"[SFP] Saved per-epoch history CSV -> {csv_path}")

    # Load best checkpoint and evaluate test set
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        print(f"[SFP] Restored best model checkpoint (val_acc={best_val:.2f}%, epoch {best_epoch})")

    test_loss, test_acc, misclassified_saved, misclassified_csv, misclassified_dir = (
        None, None, None, None, None
    )
    if args.save_misclassified_images:
        test_loss, test_acc, misclassified_saved, misclassified_csv, misclassified_dir = (
            evaluate_and_save_misclassified(
                model, test_loader, args.device, criterion, output_dir,
                class_names=class_names, max_images=args.max_misclassified_images,
            )
        )
    else:
        test_loss, test_acc = evaluate_full(model, test_loader, args.device, criterion)

    # Parameter breakdown (filter block / LoRA / LayerNorm / head / frozen backbone)
    param_breakdown = count_parameter_breakdown(model, pruned_block_indices)
    param_plot_path = plot_param_breakdown(param_breakdown, output_dir)
    print(f"[SFP] Saved parameter breakdown plot -> {param_plot_path}")

    summary = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "num_samples": args.num_samples if not args.use_full_dataset else "full",
        "full_finetune": args.full_finetune,
        "pruned_block_idx": pruned_block_indices,
        "num_filter_blocks": args.num_filter_blocks,
        "filter_block_layers": args.filter_block_layers,
        "filter_residual_hidden_dim": args.filter_residual_hidden_dim,
        "filter_residual_alpha": args.filter_residual_alpha if args.filter_residual_hidden_dim > 0 else None,
        "filter_residual_dropout": args.filter_residual_dropout if args.filter_residual_hidden_dim > 0 else None,
        "lora_rank": args.lora_rank if not args.full_finetune else None,
        "lora_alpha": args.lora_alpha if not args.full_finetune else None,
        "lora_dropout": args.lora_dropout if not args.full_finetune else None,
        "seed": args.seed,
        "deterministic": not args.fast,
        "lr_main": args.lr,
        "lr_lora": args.lora_lr,
        "warmup_epochs": warmup_epochs,
        "min_lr_ratio": args.min_lr_ratio,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "early_stopping_enabled": early_stopping_enabled,
        "patience": args.patience if early_stopping_enabled else None,
        "max_epochs": effective_epochs if early_stopping_enabled else None,
        "epochs_trained": len(history["epoch"]),
        "batch_size": args.batch_size,
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "final_test_acc": test_acc,
        "final_test_loss": test_loss,
        "param_breakdown": param_breakdown,
        "plots": {
            **curve_paths,
            "lr_schedule": lr_plot_path,
            "snip_saliency": snip_plot_path,
            "param_breakdown": param_plot_path,
        },
        "history_csv": csv_path,
        "checkpoint_path": best_path,
        "misclassified_images": {
            "enabled": args.save_misclassified_images,
            "saved_count": misclassified_saved,
            "max_images_cap": args.max_misclassified_images if args.save_misclassified_images else None,
            "csv": misclassified_csv,
            "dir": misclassified_dir,
        },
    }
    summary_path = save_metrics_summary(summary, output_dir)

    print(f"\n==================================================")
    print(f"[SFP] Dataset: {args.dataset.upper()}")
    if args.full_finetune:
        print(f"[SFP] Mode: FULL FINE-TUNE (baseline, no SFP/LoRA)")
    else:
        layer_desc = "Single" if args.filter_block_layers <= 1 else f"{args.filter_block_layers}-Layer"
        print(f"[SFP] Mode: SFP | Replaced Block(s): {pruned_block_indices} ({layer_desc} Filter Block) | "
              f"LoRA rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"[SFP] Trainable Params: {param_breakdown['trainable_params']:,} / "
          f"{param_breakdown['total_params']:,} ({param_breakdown['trainable_pct']:.2f}%)")
    print(f"[SFP]   - Filter block      : {param_breakdown['filter_block']:,}")
    print(f"[SFP]   - LoRA adapters     : {param_breakdown['lora']:,}")
    print(f"[SFP]   - LayerNorm         : {param_breakdown['layernorm']:,}")
    print(f"[SFP]   - Head              : {param_breakdown['head']:,}")
    print(f"[SFP]   - Trainable backbone: {param_breakdown['trainable_backbone']:,}")
    print(f"[SFP] Best Val Acc: {best_val:.2f}% (epoch {best_epoch})")
    print(f"[SFP] Final Test Acc: {test_acc:.2f}% | Final Test Loss: {test_loss:.4f}")
    if args.save_misclassified_images:
        print(f"[SFP] Misclassified images saved: {misclassified_saved} -> {misclassified_dir}")
    print(f"[SFP] All plots, CSV history, and metrics JSON saved under: {output_dir}")
    print(f"[SFP] Metrics summary JSON -> {summary_path}")
    print(f"==================================================")


if __name__ == "__main__":
    main()