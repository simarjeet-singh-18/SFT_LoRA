import torch
import torch.nn as nn
from typing import Dict, List, Tuple

def compute_snip_saliency_for_blocks(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda"
) -> Dict[int, float]:
    """
    Computes SNIP saliency per block across a dataloader:
    S_l = sum | g_w * w |
    """
    model.to(device)
    model.eval()

    # Enable gradients for base weights temporarily to measure saliency
    for p in model.parameters():
        p.requires_grad = True

    block_saliencies = {i: 0.0 for i in range(len(model.blocks))}
    criterion = nn.CrossEntropyLoss()

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        model.zero_grad()

        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        with torch.no_grad():
            for i, block in enumerate(model.blocks):
                block_score = 0.0
                for p in block.parameters():
                    if p.grad is not None:
                        block_score += torch.sum(torch.abs(p.grad * p)).item()
                block_saliencies[i] += block_score

    return block_saliencies


def select_block_with_snip(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    keep: str = "low"
) -> int:
    """
    Selects block index with lowest (keep='low') or highest (keep='high') SNIP score.
    Default keep='low' selects candidate block for replacement (redundant/replaceable block).
    """
    saliencies = compute_snip_saliency_for_blocks(model, dataloader, device)

    sorted_blocks = sorted(saliencies.items(), key=lambda item: item[1])
    selected_idx = sorted_blocks[0][0] if keep == "low" else sorted_blocks[-1][0]

    print("[SNIP Search] Block Saliency Scores:")
    for idx, score in sorted_blocks:
        print(f"  - Block {idx:02d}: {score:.6f}")
    print(f"[SNIP Search] Selected Block {selected_idx} (keep='{keep}')")

    return selected_idx


if __name__ == "__main__":
    # Internal execution test
    print("[snip_selection] Execution script initialized.")