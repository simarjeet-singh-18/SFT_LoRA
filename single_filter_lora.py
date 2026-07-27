import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear layer with a low-rank adapter (LoRA):
    W_new = W_base + (alpha / rank) * (W_B @ W_A)
    """
    def __init__(self, base_layer: nn.Linear, rank: int = 16, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, base_layer.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, rank))
            self.reset_parameters()

    def reset_parameters(self):
        if self.rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self.rank > 0:
            lora_out = F.linear(self.dropout(x), self.lora_A)
            lora_out = F.linear(lora_out, self.lora_B)
            result = result + self.scaling * lora_out
        return result


class SingleFilterBlock(nn.Module):
    """
    Single linear filter block replacing a pruned Transformer block.
    Initialized via Ridge Pseudo-Inverse (paper Eq. 3-4).
    """
    def __init__(self, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.dropout = nn.Dropout(p=dropout)

    def init_from_pinv(self, X_in: torch.Tensor, X_out: torch.Tensor):
        """
        Fits matrix W such that X_in @ W ≈ X_out
        """
        with torch.no_grad():
            X_in_flat = X_in.reshape(-1, X_in.size(-1)).float()
            X_out_flat = X_out.reshape(-1, X_out.size(-1)).float()

            eye = torch.eye(X_in_flat.size(1), device=X_in.device) * 1e-4
            pinv = torch.linalg.solve(X_in_flat.T @ X_in_flat + eye, X_in_flat.T @ X_out_flat)

            self.weight.copy_(pinv.T.to(self.weight.dtype))
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.linear(x, self.weight, self.bias))


def count_parameter_breakdown(model: nn.Module, pruned_block_idx: int) -> dict:
    """
    Categorizes every parameter in the model for reporting/plotting purposes.
    Categories (checked in this order, first match wins):
      - filter_block   : the SingleFilterBlock replacing model.blocks[pruned_block_idx]
      - lora           : LoRA A/B matrices injected into remaining blocks
      - layernorm      : unfrozen LN params (norm1/norm2/final norm)
      - head           : classifier head
      - frozen_backbone: everything else (untouched, frozen pretrained weights)
    Also returns total_params, trainable_params, and trainable_pct.
    """
    breakdown = {
        "filter_block": 0,
        "lora": 0,
        "layernorm": 0,
        "head": 0,
        "frozen_backbone": 0,
    }
    total = 0
    trainable = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n

        if f"blocks.{pruned_block_idx}" in name:
            breakdown["filter_block"] += n
        elif "lora_" in name:
            breakdown["lora"] += n
        elif "norm" in name.lower():
            breakdown["layernorm"] += n
        elif "head" in name:
            breakdown["head"] += n
        else:
            breakdown["frozen_backbone"] += n

    breakdown["total_params"] = total
    breakdown["trainable_params"] = trainable
    breakdown["trainable_pct"] = (100.0 * trainable / total) if total > 0 else 0.0
    return breakdown


def apply_single_filter_and_lora(
    model: nn.Module,
    pruned_block_idx: int,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    target_keywords: list = ["qkv", "proj", "fc1", "fc2"]
) -> SingleFilterBlock:
    """
    1. Replaces block at pruned_block_idx with SingleFilterBlock.
    2. Inject LoRA adapters into designated linear projections across remaining blocks.
    3. Sets requires_grad status appropriately.
    """
    embed_dim = getattr(model, "embed_dim", 768)

    # 1. Substitute target Transformer block with Single Filter Block
    filter_block = SingleFilterBlock(embed_dim=embed_dim)
    model.blocks[pruned_block_idx] = filter_block

    # 2. Inject LoRA adapters into all remaining blocks
    lora_params = 0
    for idx, block in enumerate(model.blocks):
        if idx == pruned_block_idx:
            continue

        for name, module in list(block.named_modules()):
            if any(kw in name for kw in target_keywords) and isinstance(module, nn.Linear):
                parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = block if parent_name == "" else block.get_submodule(parent_name)

                lora_layer = LoRALinear(module, rank=lora_rank, alpha=lora_alpha)
                setattr(parent, attr_name, lora_layer)
                lora_params += lora_rank * (module.in_features + module.out_features)

    # 3. Freeze base parameters and enable gradients for Filter, LoRA, LayerNorm, and Classifier Head
    # LN unfreezing follows the paper's ablation (Table 4): substitution induces activation-distribution
    # shifts, and retraining LN params (~0.038M) is needed to realign feature geometry.
    ln_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for p in module.parameters():
                ln_params += p.numel()

    for name, param in model.named_parameters():
        if (
            "lora_" in name
            or f"blocks.{pruned_block_idx}" in name
            or "head" in name
            or "norm" in name.lower()
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    print(f"[SFP-SingleFilter] Substituted block {pruned_block_idx} with Single Filter Block.")
    print(f"[SFP-SingleFilter] Injected {lora_params:,} LoRA parameters (rank={lora_rank}, alpha={lora_alpha}).")
    print(f"[SFP-SingleFilter] Unfroze {ln_params:,} LayerNorm parameters across all blocks.")
    return filter_block