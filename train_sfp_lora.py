import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm

from data import get_dataloaders
from single_filter_lora import apply_single_filter_and_lora, count_parameter_breakdown
from snip_selection import select_block_with_snip
from plotting import (
    ensure_dir,
    plot_training_curves,
    plot_snip_saliency,
    plot_param_breakdown,
    save_history_csv,
    save_metrics_summary,
)


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


def main():
    parser = argparse.ArgumentParser(description="SFP Single Filter + LoRA Fine-Tuning")
    parser.add_argument("--dataset", type=str, default="pets", choices=["pets", "svhn", "flowers102", "dtd", "caltech101", "cifar100"])
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--use-full-dataset", action="store_true")
    parser.add_argument("--pruned-block", type=int, default=-1, help="-1 runs SNIP search automatically")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=1e-3, help="LR for filter block, LayerNorm, and head")
    parser.add_argument("--lora-lr", type=float, default=3e-4, help="LR for LoRA adapter params (A/B matrices)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                         help="Root directory for plots, CSV history, and metrics summary. "
                              "A per-dataset subfolder is created automatically.")
    args = parser.parse_args()

    output_dir = ensure_dir(os.path.join(args.output_dir, args.dataset))
    print(f"[SFP] Outputs (plots, CSV, metrics JSON) will be saved to: {output_dir}")

    # 1. Load Data
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(args)

    # 2. Load Pretrained Backbone
    print(f"[SFP] Loading ViT backbone for {num_classes} output classes...")
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)
    model.to(args.device)

    # 3. Determine target block index via SNIP search if not specified
    pruned_block_idx = args.pruned_block
    if pruned_block_idx < 0:
        print("[SFP] No block index provided. Running SNIP search...")
        pruned_block_idx, snip_saliencies = select_block_with_snip(
            model, train_loader, device=args.device, keep="low", return_scores=True
        )
        snip_plot_path = plot_snip_saliency(snip_saliencies, pruned_block_idx, output_dir)
        print(f"[SFP] Saved SNIP saliency plot -> {snip_plot_path}")
    else:
        snip_saliencies = None

    # 4. Extract Block Input/Output features for Pseudo-Inverse Init
    print(f"[SFP] Extracting representations for Pseudo-Inverse Init at block {pruned_block_idx}...")
    X_in, X_out = extract_block_inputs_outputs(model, train_loader, pruned_block_idx, args.device)

    # 5. Substitute Single Filter & Inject LoRA
    filter_block = apply_single_filter_and_lora(
        model,
        pruned_block_idx=pruned_block_idx,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )

    # Initialize single filter weights via Ridge Pseudo-Inverse
    filter_block.init_from_pinv(X_in.to(args.device), X_out.to(args.device))

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

    criterion = nn.CrossEntropyLoss()

    print(f"[SFP] Starting training for {args.epochs} epochs on {args.dataset.upper()}...")
    best_val, best_epoch, best_path = 0.0, 0, os.path.join(output_dir, f"best_sfp_lora_{args.dataset}.pt")

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x, y = batch[0].to(args.device), batch[1].to(args.device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate_full(model, val_loader, args.device, criterion)

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), best_path)

        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] Train Loss: {epoch_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Best Val: {best_val:.2f}%")

    # Save curves + raw per-epoch history as soon as training finishes, so they exist
    # even if something later (checkpoint reload, test eval) fails.
    curve_paths = plot_training_curves(history, output_dir, dataset_name=args.dataset)
    csv_path = save_history_csv(history, output_dir)
    print(f"[SFP] Saved training curves -> {curve_paths}")
    print(f"[SFP] Saved per-epoch history CSV -> {csv_path}")

    # Load best checkpoint and evaluate test set
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        print(f"[SFP] Restored best model checkpoint (val_acc={best_val:.2f}%, epoch {best_epoch})")

    test_loss, test_acc = evaluate_full(model, test_loader, args.device, criterion)

    # Parameter breakdown (filter block / LoRA / LayerNorm / head / frozen backbone)
    param_breakdown = count_parameter_breakdown(model, pruned_block_idx)
    param_plot_path = plot_param_breakdown(param_breakdown, output_dir)
    print(f"[SFP] Saved parameter breakdown plot -> {param_plot_path}")

    summary = {
        "dataset": args.dataset,
        "num_classes": num_classes,
        "num_samples": args.num_samples if not args.use_full_dataset else "full",
        "pruned_block_idx": pruned_block_idx,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lr_main": args.lr,
        "lr_lora": args.lora_lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "best_val_acc": best_val,
        "best_epoch": best_epoch,
        "final_test_acc": test_acc,
        "final_test_loss": test_loss,
        "param_breakdown": param_breakdown,
        "plots": {
            **curve_paths,
            "snip_saliency": snip_plot_path if snip_saliencies is not None else None,
            "param_breakdown": param_plot_path,
        },
        "history_csv": csv_path,
        "checkpoint_path": best_path,
    }
    summary_path = save_metrics_summary(summary, output_dir)

    print(f"\n==================================================")
    print(f"[SFP] Dataset: {args.dataset.upper()}")
    print(f"[SFP] Replaced Block: {pruned_block_idx} | LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"[SFP] Trainable Params: {param_breakdown['trainable_params']:,} / "
          f"{param_breakdown['total_params']:,} ({param_breakdown['trainable_pct']:.2f}%)")
    print(f"[SFP]   - Filter block : {param_breakdown['filter_block']:,}")
    print(f"[SFP]   - LoRA adapters: {param_breakdown['lora']:,}")
    print(f"[SFP]   - LayerNorm    : {param_breakdown['layernorm']:,}")
    print(f"[SFP]   - Head         : {param_breakdown['head']:,}")
    print(f"[SFP] Best Val Acc: {best_val:.2f}% (epoch {best_epoch})")
    print(f"[SFP] Final Test Acc: {test_acc:.2f}% | Final Test Loss: {test_loss:.4f}")
    print(f"[SFP] All plots, CSV history, and metrics JSON saved under: {output_dir}")
    print(f"[SFP] Metrics summary JSON -> {summary_path}")
    print(f"==================================================")


if __name__ == "__main__":
    main()