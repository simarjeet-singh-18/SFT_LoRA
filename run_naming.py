"""
Builds a run/output folder name directly from sys.argv, using only the flags
that were actually passed on the command line (not the full argparse
namespace, which would include every default too).

Example:
  python3 train_sfp_lora.py --dataset pets --use-full-dataset --lora-rank 0 \
      --epochs 30 --lora-dropout 0.1 --warmup-epochs 2 --min-lr-ratio 0.01

  -> "dataset_pets_use-full-dataset_lora-rank_0_epochs_30_lora-dropout_0_1_" \
     "warmup-epochs_2_min-lr-ratio_0_01"

Rules:
  - Only tokens starting with "--" are considered.
  - "--flag value" and "--flag=value" both produce "flag_value".
  - A bare boolean flag (e.g. store_true, no following value) produces just "flag".
  - "." in values is replaced with "_" (so 0.1 -> 0_1, 0.01 -> 0_01).
  - Order matches the order the flags were passed in.
  - "--output-dir" is excluded, since it names *where* runs are stored, not a
    hyperparameter of the run itself, and including it would be circular
    (its value would end up nested inside itself).
"""

import re

EXCLUDED_KEYS = {"output-dir"}


def _sanitize(token: str) -> str:
    """Replace filesystem-unfriendly characters. Decimals become underscores."""
    token = token.replace(".", "_")
    token = re.sub(r"[^\w\-]", "-", token)  # anything not alnum/underscore/hyphen -> hyphen
    return token


def build_run_folder_name(argv: list) -> str:
    """
    argv: typically sys.argv[1:] (i.e. excluding the script name).
    Returns a folder-name-safe string built from the passed flags, in order.
    Falls back to "default_run" if no flags were passed.
    """
    parts = []
    i = 0
    n = len(argv)

    while i < n:
        token = argv[i]
        if not token.startswith("--"):
            i += 1
            continue

        raw = token[2:]

        if "=" in raw:
            key, value = raw.split("=", 1)
            if key not in EXCLUDED_KEYS:
                parts.append(_sanitize(key))
                parts.append(_sanitize(value))
            i += 1
            continue

        key = raw
        has_value = (i + 1 < n) and (not argv[i + 1].startswith("--"))

        if key in EXCLUDED_KEYS:
            i += 2 if has_value else 1
            continue

        if has_value:
            value = argv[i + 1]
            parts.append(_sanitize(key))
            parts.append(_sanitize(value))
            i += 2
        else:
            # Boolean/store_true flag with no following value
            parts.append(_sanitize(key))
            i += 1

    if not parts:
        return "default_run"
    return "_".join(parts)