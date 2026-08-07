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

    def orthogonality_penalty(self):
        """
        Returns (A_term, B_term), the (rank x rank) residual matrices whose squared
        Frobenius norm penalizes lora_A / lora_B for being far from row/column-
        orthonormal. None if rank <= 0 (no LoRA params to regularize on this layer).

        Shapes: lora_A is (rank, in_features), lora_B is (out_features, rank), with
        rank << in_features/out_features. Because of that, lora_A can only ever have
        AT MOST `rank` linearly independent directions among its in_features-dim rows
        -- so the meaningful orthogonality constraint is on those `rank` ROWS being
        mutually orthonormal, i.e. A @ A.T ~ I_rank (a (rank x rank) identity is
        achievable; forcing A.T @ A, which is (in_features x in_features) and has
        rank <= rank < in_features, could never equal an identity matrix).
        Symmetrically for lora_B, whose `rank` COLUMNS are the quantity that can be
        made orthonormal: B.T @ B ~ I_rank.

        Driving both toward the identity pushes each adapter's `rank` update
        directions to be linearly independent of one another, i.e. discourages the
        adapter from wasting capacity by learning redundant (near-parallel) columns.
        """
        if self.rank <= 0:
            return None
        eye_r = torch.eye(self.rank, device=self.lora_A.device, dtype=self.lora_A.dtype)
        A_term = self.lora_A @ self.lora_A.t() - eye_r   # (rank, rank)
        B_term = self.lora_B.t() @ self.lora_B - eye_r   # (rank, rank)
        return A_term, B_term


def compute_lora_orthogonality_loss(model: nn.Module, lambda1: float = 0.0, lambda2: float = 0.0) -> torch.Tensor:
    """
    Sums the orthogonality regularizer lambda1 * ||A@A.T - I||_F^2 + lambda2 * ||B.T@B - I||_F^2
    over every LoRALinear submodule in `model` (see LoRALinear.orthogonality_penalty
    for why A@A.T / B.T@B, rather than A.T@A / B@B.T, are the well-posed choices
    given LoRA's (rank, in_features) / (out_features, rank) shapes).

    Returns a 0-dim tensor on the same device as the model's parameters, so it can
    always be added directly to the task loss (returns exactly 0.0, with no graph
    attached to lora_A/lora_B, when both lambdas are 0 -- the default -- so existing
    training runs that don't pass either flag are numerically unaffected).
    """
    device = next(model.parameters()).device
    total = torch.zeros((), device=device)
    if lambda1 == 0.0 and lambda2 == 0.0:
        return total

    for module in model.modules():
        if isinstance(module, LoRALinear) and module.rank > 0:
            terms = module.orthogonality_penalty()
            if terms is None:
                continue
            A_term, B_term = terms
            if lambda1 != 0.0:
                total = total + lambda1 * torch.sum(A_term * A_term)
            if lambda2 != 0.0:
                total = total + lambda2 * torch.sum(B_term * B_term)
    return total


class FilterResidualMLP(nn.Module):
    """
    Optional nonlinear residual branch for a filter block: fc2(GELU(fc1(x))).

    fc2 is ZERO-INITIALIZED (weight and bias), so this branch contributes EXACTLY
    ZERO at initialization -- the same safe-start trick LoRALinear already uses for
    its own lora_B matrix. This means attaching this branch to a filter block does
    NOT disturb that block's pseudoinverse-inherited behavior at step 1; the branch
    can only start contributing once gradients move fc2 away from zero during
    training.

    fc1 keeps its default (random) init. This is fine precisely because fc2 starts
    at zero: whatever fc1 outputs gets multiplied by zero at fc2 regardless, so a
    random fc1 can't destabilize anything at initialization.

    Unlike the purely-linear --filter-block-layers stack, this branch genuinely
    adds expressivity (the GELU nonlinearity means fc2(GELU(fc1(x))) is NOT
    reducible to a single linear map) -- this is the mechanism for real added
    capacity in the filter block, implemented as a safe zero-init residual rather
    than by making the block's main path nonlinear (which would break the
    closed-form pseudoinverse init and risk destabilizing the inherited behavior).

    scaling = alpha / hidden_dim, mirroring LoRALinear's alpha/rank normalization:
    keeps the branch's effective update magnitude comparable across different
    hidden_dim choices once it does start contributing.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.scaling = alpha / hidden_dim if hidden_dim > 0 else 1.0

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc2(self.act(self.fc1(x)))
        return self.scaling * self.dropout(out)


class SingleFilterBlock(nn.Module):
    """
    Single linear filter block replacing a pruned Transformer block.
    Initialized via Ridge Pseudo-Inverse (paper Eq. 3-4).

    Optionally attaches a nonlinear zero-init residual branch (FilterResidualMLP)
    when residual_hidden_dim > 0 -- see that class's docstring for why this is the
    safe way to add genuine nonlinear capacity without disturbing the pseudoinverse
    init. Default (residual_hidden_dim=0) is UNCHANGED from all previous versions:
    no residual submodule is constructed at all, so state_dict keys for existing
    checkpoints trained without this feature still match exactly.
    """
    def __init__(self, embed_dim: int, dropout: float = 0.0,
                 residual_hidden_dim: int = 0, residual_alpha: float = 1.0, residual_dropout: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(embed_dim))
        self.bias = nn.Parameter(torch.zeros(embed_dim))
        self.dropout = nn.Dropout(p=dropout)

        self.residual = None
        if residual_hidden_dim > 0:
            self.residual = FilterResidualMLP(
                embed_dim=embed_dim, hidden_dim=residual_hidden_dim,
                alpha=residual_alpha, dropout=residual_dropout,
            )

    def init_from_pinv(self, X_in: torch.Tensor, X_out: torch.Tensor):
        """
        Fits matrix W such that X_in @ W ≈ X_out. Only touches the main linear
        path (self.weight/self.bias) -- the residual branch (if present) stays at
        its zero-init starting point regardless, exactly as intended.
        """
        with torch.no_grad():
            X_in_flat = X_in.reshape(-1, X_in.size(-1)).float()
            X_out_flat = X_out.reshape(-1, X_out.size(-1)).float()

            eye = torch.eye(X_in_flat.size(1), device=X_in.device) * 1e-4
            pinv = torch.linalg.solve(X_in_flat.T @ X_in_flat + eye, X_in_flat.T @ X_out_flat)

            self.weight.copy_(pinv.T.to(self.weight.dtype))
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.residual is not None:
            out = out + self.residual(x)
        return self.dropout(out)


class MultiLayerFilterBlock(nn.Module):
    """
    N-layer (N >= 2) generalization of SingleFilterBlock's linear filter block.

    IMPORTANT: the stacked layers themselves are purely linear (no activation
    between them), so stacking them does NOT add expressivity beyond a single
    layer -- the composed transformation is mathematically still just one linear
    map (matrix product collapses). This class exists to let you experiment with
    depth/parameterization while preserving the EXACT SAME inheritance property as
    the single-layer case. For genuine added capacity, attach a nonlinear
    zero-init residual branch instead (residual_hidden_dim > 0 -- see
    FilterResidualMLP's docstring); that's a separate, safer mechanism than making
    this stack itself nonlinear, since it doesn't disturb the closed-form
    pseudoinverse init.

    Weight initialization generalizes the paper's pseudoinverse trick (Eq. 3-4) to
    N layers as follows:
      - Solve the SAME problem as the single-layer case: min_W ||X_in@W - X_out||_F^2
        -> W = X_in^+ @ X_out
      - Initialize exactly ONE layer (the last one, closest to the block's output)
        with W and zero bias -- identical to SingleFilterBlock's own init
      - Initialize every OTHER layer to the identity transform (identity weight
        matrix, zero bias)
      - Composing identity maps changes nothing, so the WHOLE STACK's product is
        IDENTICAL to a single layer initialized with W, regardless of N. This
        exactly preserves the "inherits the original block's behavior" property
        at any depth.
    """

    def __init__(self, embed_dim: int, num_layers: int, dropout: float = 0.0,
                 residual_hidden_dim: int = 0, residual_alpha: float = 1.0, residual_dropout: float = 0.0):
        super().__init__()
        assert num_layers >= 2, "MultiLayerFilterBlock requires num_layers >= 2 (use SingleFilterBlock for 1)"
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)])
        # Identity-init every layer up front, so the block is at least a
        # well-behaved no-op even before init_from_pinv() is called (rather than
        # torch's default random nn.Linear init).
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(torch.eye(embed_dim))
                layer.bias.zero_()

        self.dropout = nn.Dropout(p=dropout)

        self.residual = None
        if residual_hidden_dim > 0:
            self.residual = FilterResidualMLP(
                embed_dim=embed_dim, hidden_dim=residual_hidden_dim,
                alpha=residual_alpha, dropout=residual_dropout,
            )

    def init_from_pinv(self, X_in: torch.Tensor, X_out: torch.Tensor):
        """
        Fits W such that X_in @ W ~= X_out (same as SingleFilterBlock.init_from_pinv),
        assigns W to the LAST layer, and resets every other layer to identity. See
        class docstring for why this exactly generalizes the single-layer inheritance
        property to any depth.
        """
        with torch.no_grad():
            X_in_flat = X_in.reshape(-1, X_in.size(-1)).float()
            X_out_flat = X_out.reshape(-1, X_out.size(-1)).float()

            eye = torch.eye(X_in_flat.size(1), device=X_in.device) * 1e-4
            pinv = torch.linalg.solve(X_in_flat.T @ X_in_flat + eye, X_in_flat.T @ X_out_flat)
            W = pinv.T.to(self.layers[0].weight.dtype)

            for layer in self.layers[:-1]:
                layer.weight.copy_(torch.eye(self.embed_dim, device=W.device, dtype=W.dtype))
                layer.bias.zero_()
            self.layers[-1].weight.copy_(W)
            self.layers[-1].bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_input = x  # residual branch sees the block's ORIGINAL input, not the stacked-layer intermediate
        for layer in self.layers:
            x = layer(x)
        if self.residual is not None:
            x = x + self.residual(residual_input)
        return self.dropout(x)


def make_filter_block(embed_dim: int, num_layers: int = 1, dropout: float = 0.0,
                       residual_hidden_dim: int = 0, residual_alpha: float = 1.0, residual_dropout: float = 0.0):
    """
    Factory: returns SingleFilterBlock for num_layers==1 (unchanged, backward
    compatible with all existing checkpoints), or MultiLayerFilterBlock for
    num_layers > 1. residual_hidden_dim > 0 attaches a nonlinear zero-init
    residual branch to either (see FilterResidualMLP); default 0 = no residual
    branch, fully backward compatible.
    """
    if num_layers <= 1:
        return SingleFilterBlock(embed_dim=embed_dim, dropout=dropout,
                                  residual_hidden_dim=residual_hidden_dim,
                                  residual_alpha=residual_alpha, residual_dropout=residual_dropout)
    return MultiLayerFilterBlock(embed_dim=embed_dim, num_layers=num_layers, dropout=dropout,
                                  residual_hidden_dim=residual_hidden_dim,
                                  residual_alpha=residual_alpha, residual_dropout=residual_dropout)


def _normalize_indices(block_idx_or_indices) -> set:
    """Accepts a single int or any iterable of ints; always returns a set of ints."""
    if isinstance(block_idx_or_indices, int):
        return {block_idx_or_indices}
    return set(block_idx_or_indices)


def count_parameter_breakdown(model: nn.Module, pruned_block_idx=None) -> dict:
    """
    Categorizes every parameter in the model for reporting/plotting purposes.
    Categories (checked in this order, first match wins):
      - filter_block     : any SingleFilterBlock/MultiLayerFilterBlock replacing
                            model.blocks[idx] for idx in pruned_block_idx
      - lora              : LoRA A/B matrices injected into remaining blocks
      - layernorm         : unfrozen LN params (norm1/norm2/final norm)
      - head              : classifier head
      - trainable_backbone: anything else that IS trainable (e.g. full-finetune mode,
                             where the whole backbone is unfrozen and there's no filter
                             block / LoRA to separate out)
      - frozen_backbone   : anything else that is NOT trainable
    Also returns total_params, trainable_params, and trainable_pct.

    pruned_block_idx: None (full-finetune runs, no block substitution happened),
    a single int (legacy single-block runs), or a list/set of ints (multi-block runs).
    """
    pruned_indices = _normalize_indices(pruned_block_idx) if pruned_block_idx is not None else set()

    breakdown = {
        "filter_block": 0,
        "lora": 0,
        "layernorm": 0,
        "head": 0,
        "trainable_backbone": 0,
        "frozen_backbone": 0,
    }
    total = 0
    trainable = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n

        if any(f"blocks.{idx}" in name for idx in pruned_indices):
            breakdown["filter_block"] += n
        elif "lora_" in name:
            breakdown["lora"] += n
        elif "norm" in name.lower():
            breakdown["layernorm"] += n
        elif "head" in name:
            breakdown["head"] += n
        elif param.requires_grad:
            breakdown["trainable_backbone"] += n
        else:
            breakdown["frozen_backbone"] += n

    breakdown["total_params"] = total
    breakdown["trainable_params"] = trainable
    breakdown["trainable_pct"] = (100.0 * trainable / total) if total > 0 else 0.0
    return breakdown


def substitute_filter_block(model: nn.Module, block_idx: int, num_layers: int = 1, dropout: float = 0.0,
                             residual_hidden_dim: int = 0, residual_alpha: float = 1.0, residual_dropout: float = 0.0):
    """
    Replaces model.blocks[block_idx] with a fresh filter block (SingleFilterBlock
    if num_layers==1, MultiLayerFilterBlock otherwise). Does NOT perform the
    pseudoinverse init -- call .init_from_pinv(X_in, X_out) on the returned module
    afterward, using block inputs/outputs extracted from the model's CURRENT state.

    For multi-block runs, substitute blocks in INCREASING index order and extract
    each block's I/O data AFTER earlier substitutions have already happened, so
    later filter blocks correctly learn to map from the already-modified preceding
    representations (mirrors the paper's own sequential dual-layer construction,
    Fig. 3).

    residual_hidden_dim > 0 attaches a nonlinear zero-init residual branch to the
    filter block (see FilterResidualMLP) -- default 0 = no residual branch.
    """
    embed_dim = getattr(model, "embed_dim", 768)
    filter_block = make_filter_block(embed_dim=embed_dim, num_layers=num_layers, dropout=dropout,
                                      residual_hidden_dim=residual_hidden_dim,
                                      residual_alpha=residual_alpha, residual_dropout=residual_dropout)
    model.blocks[block_idx] = filter_block
    return filter_block


def inject_lora(
    model: nn.Module,
    excluded_block_indices,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    target_keywords: list = ["qkv", "proj", "fc1", "fc2"],
) -> int:
    """
    Wraps target linear layers with LoRALinear in every block EXCEPT those in
    excluded_block_indices (the filter-substituted blocks). Returns total number
    of LoRA parameters injected.
    """
    excluded = _normalize_indices(excluded_block_indices)
    lora_params = 0
    for idx, block in enumerate(model.blocks):
        if idx in excluded:
            continue
        for name, module in list(block.named_modules()):
            if any(kw in name for kw in target_keywords) and isinstance(module, nn.Linear):
                parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = block if parent_name == "" else block.get_submodule(parent_name)

                lora_layer = LoRALinear(module, rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
                setattr(parent, attr_name, lora_layer)
                lora_params += lora_rank * (module.in_features + module.out_features)
    return lora_params


def freeze_non_trainable(model: nn.Module, filter_block_indices) -> int:
    """
    Sets requires_grad=True for LoRA params, filter-block params (any block index
    in filter_block_indices), LayerNorm params, and the classifier head;
    requires_grad=False for everything else. Returns total LayerNorm param count
    (for logging). LN unfreezing follows the paper's ablation (Table 4): substitution
    induces activation-distribution shifts, and retraining LN params is needed to
    realign feature geometry.
    """
    indices = _normalize_indices(filter_block_indices)

    ln_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for p in module.parameters():
                ln_params += p.numel()

    for name, param in model.named_parameters():
        if (
            "lora_" in name
            or any(f"blocks.{idx}" in name for idx in indices)
            or "head" in name
            or "norm" in name.lower()
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    return ln_params


def apply_single_filter_and_lora(
    model: nn.Module,
    pruned_block_idx: int,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.0,
    filter_num_layers: int = 1,
    filter_residual_hidden_dim: int = 0,
    filter_residual_alpha: float = 1.0,
    filter_residual_dropout: float = 0.0,
    target_keywords: list = ["qkv", "proj", "fc1", "fc2"]
):
    """
    Backward-compatible convenience wrapper for the SINGLE-block case, built on top
    of substitute_filter_block / inject_lora / freeze_non_trainable. Behavior is
    unchanged from all previous versions of this function when filter_num_layers=1
    and filter_residual_hidden_dim=0 (both defaults) -- both are new, letting the
    filter block use more than one (purely linear) layer and/or a nonlinear
    zero-init residual branch; see MultiLayerFilterBlock's and FilterResidualMLP's
    docstrings for details.

    Does NOT perform the pseudoinverse init itself -- caller still does that
    afterward via filter_block.init_from_pinv(X_in, X_out), exactly as before.

    For multi-block runs (more than one filter-substituted block), don't use this
    function -- call substitute_filter_block / inject_lora / freeze_non_trainable
    directly in a loop instead (see train_sfp_lora.py's main() for the pattern).
    """
    filter_block = substitute_filter_block(
        model, pruned_block_idx, num_layers=filter_num_layers,
        residual_hidden_dim=filter_residual_hidden_dim,
        residual_alpha=filter_residual_alpha, residual_dropout=filter_residual_dropout,
    )
    lora_params = inject_lora(model, pruned_block_idx, lora_rank, lora_alpha, lora_dropout, target_keywords)
    ln_params = freeze_non_trainable(model, pruned_block_idx)

    layer_desc = "Single" if filter_num_layers <= 1 else f"{filter_num_layers}-Layer"
    residual_desc = f" + residual(hidden={filter_residual_hidden_dim})" if filter_residual_hidden_dim > 0 else ""
    print(f"[SFP-SingleFilter] Substituted block {pruned_block_idx} with {layer_desc} Filter Block{residual_desc}.")
    print(f"[SFP-SingleFilter] Injected {lora_params:,} LoRA parameters "
          f"(rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}).")
    print(f"[SFP-SingleFilter] Unfroze {ln_params:,} LayerNorm parameters across all blocks.")
    return filter_block