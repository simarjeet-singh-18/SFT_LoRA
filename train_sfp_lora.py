import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm

from data import get_dataloaders
from single_filter_lora import apply_single_filter_and_lora
from snip_selection import select_block_with_snip


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


def main():
    parser = argparse.ArgumentParser(description="SFP Single Filter + LoRA Fine-Tuning")
    parser.add_argument("--dataset", type=str, default="pets", choices=["pets", "svhn", "flowers102", "dtd", "caltech101", "cifar100"])
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--use-full-dataset", action="store_true")
    parser.add_argument("--pruned-block", type=int, default=-1, help="-1 runs SNIP search automatically")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

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
        pruned_block_idx = select_block_with_snip(model, train_loader, device=args.device, keep="low")

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
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f"[SFP] Starting training for {args.epochs} epochs on {args.dataset.upper()}...")
    best_val, best_path = 0.0, f"best_sfp_lora_{args.dataset}.pt"

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
        val_acc = evaluate(model, val_loader, args.device)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_path)

        print(f"[Epoch {epoch:03d}/{args.epochs:03d}] Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}% | Best Val: {best_val:.2f}%")

    # Load best checkpoint and evaluate test set
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path))
        print(f"[SFP] Restored best model checkpoint (val_acc={best_val:.2f}%)")

    test_acc = evaluate(model, test_loader, args.device)
    print(f"\n==================================================")
    print(f"[SFP] Dataset: {args.dataset.upper()}")
    print(f"[SFP] Final Test Accuracy: {test_acc:.2f}% (Best Val: {best_val:.2f}%)")
    print(f"==================================================")


if __name__ == "__main__":
    main()