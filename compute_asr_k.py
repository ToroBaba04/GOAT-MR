import json, os, re, sys
from collections import defaultdict
import config


def compute_asr_k(config_name: str, max_k: int = 10):
    results_dir = f"{config.RESULTS_DIRS[config_name]}_asrk"

    if not os.path.isdir(results_dir):
        print(f"No directory found: {results_dir}")
        return

    files = sorted(f for f in os.listdir(results_dir)
                   if re.match(r"asrk_b\d+_r\d+\.json", f))

    by_behavior = defaultdict(dict)
    category_by_behavior = {}

    for fname in files:
        m = re.match(r"asrk_b(\d+)_r(\d+)\.json", fname)
        b_idx, r_num = int(m.group(1)), int(m.group(2))
        with open(os.path.join(results_dir, fname), encoding="utf-8") as f:
            data = json.load(f)
        by_behavior[b_idx][r_num] = data.get("success", False)
        category_by_behavior[b_idx] = data.get("category", "Unknown")

    n_behaviors = len(by_behavior)
    print(f"\n{'='*50}")
    print(f"ASR@k report — {config_name}")
    print(f"{'='*50}")
    print(f"Behaviors found: {n_behaviors}\n")

    for k in range(1, max_k + 1):
        successes = sum(
            1 for rounds in by_behavior.values()
            if any(rounds.get(r, False) for r in range(1, k + 1))
        )
        completeness = sum(
            1 for rounds in by_behavior.values()
            if all(r in rounds for r in range(1, k + 1))
        )
        asr_k = 100.0 * successes / n_behaviors if n_behaviors else 0
        flag = ("" if completeness == n_behaviors
                else f"  (⚠ only {completeness}/{n_behaviors} behaviors have all {k} rounds so far)")
        print(f"  ASR@{k:<2d} : {asr_k:5.1f}% ({successes}/{n_behaviors}){flag}")

    # Per-category ASR@1 and ASR@10 for the final table
    print(f"\n=== Per-category ASR@1 vs ASR@{max_k} ===")
    by_cat = defaultdict(list)
    for b_idx, rounds in by_behavior.items():
        cat = category_by_behavior[b_idx]
        asr1 = rounds.get(1, False)
        asrk = any(rounds.get(r, False) for r in range(1, max_k + 1))
        by_cat[cat].append((asr1, asrk))

    for cat in sorted(by_cat.keys()):
        vals = by_cat[cat]
        n = len(vals)
        r1 = 100 * sum(v[0] for v in vals) / n
        rk = 100 * sum(v[1] for v in vals) / n
        print(f"  {cat:35s} ASR@1={r1:5.1f}%  ASR@{max_k}={rk:5.1f}%  (n={n})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 compute_asr_k.py <baseline|goat_mr> [max_k]")
        sys.exit(1)
    config_name = sys.argv[1]
    max_k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    compute_asr_k(config_name, max_k)