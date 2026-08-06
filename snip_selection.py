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
    keep: str = "low",
    return_scores: bool = False,
):
    """
    Selects block index with lowest (keep='low') or highest (keep='high') SNIP score.
    Default keep='low' selects candidate block for replacement (redundant/replaceable block).

    If return_scores=True, returns (selected_idx, saliencies_dict) so the caller can
    plot the full per-block saliency profile. Default behavior (return_scores=False)
    is unchanged for backward compatibility.
    """
    saliencies = compute_snip_saliency_for_blocks(model, dataloader, device)

    sorted_blocks = sorted(saliencies.items(), key=lambda item: item[1])
    selected_idx = sorted_blocks[0][0] if keep == "low" else sorted_blocks[-1][0]

    print("[SNIP Search] Block Saliency Scores:")
    for idx, score in sorted_blocks:
        print(f"  - Block {idx:02d}: {score:.6f}")
    print(f"[SNIP Search] Selected Block {selected_idx} (keep='{keep}')")

    if return_scores:
        return selected_idx, saliencies
    return selected_idx


def select_blocks_with_snip(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    num_blocks: int = 1,
    keep: str = "low",
    return_scores: bool = False,
):
    """
    Generalizes select_block_with_snip to multiple blocks: selects the num_blocks
    block indices with lowest (keep='low') or highest (keep='high') SNIP saliency.

    Returns indices sorted in INCREASING order -- required for sequential
    filter-block substitution, since block k's pseudoinverse init must be computed
    from the model's CURRENT state, i.e. after any earlier (lower-index) blocks in
    the selection have already been substituted.

    num_blocks=1 behaves identically to select_block_with_snip (same underlying
    saliency computation, same selection rule).
    """
    saliencies = compute_snip_saliency_for_blocks(model, dataloader, device)
    total_blocks = len(saliencies)
    if num_blocks > total_blocks:
        print(f"[SNIP Search] Warning: requested num_blocks={num_blocks} exceeds total "
              f"block count ({total_blocks}); clamping to {total_blocks}.")
        num_blocks = total_blocks
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")

    sorted_blocks = sorted(saliencies.items(), key=lambda item: item[1])
    if keep == "low":
        selected = [idx for idx, _ in sorted_blocks[:num_blocks]]
    else:
        selected = [idx for idx, _ in sorted_blocks[-num_blocks:]]
    selected_sorted = sorted(selected)
    selected_set = set(selected_sorted)

    print("[SNIP Search] Block Saliency Scores:")
    for idx, score in sorted_blocks:
        marker = "  <== SELECTED FOR FILTER SUBSTITUTION" if idx in selected_set else ""
        print(f"  - Block {idx:02d}: {score:.6f}{marker}")
    print(f"[SNIP Search] Selected {len(selected_sorted)} block(s) for filter substitution: "
          f"{selected_sorted} (keep='{keep}')")

    if return_scores:
        return selected_sorted, saliencies
    return selected_sorted


if __name__ == "__main__":
    # Internal execution test
    print("[snip_selection] Execution script initialized.")