import json, os, re
from collections import defaultdict
import math
import config


def load_success_by_behavior(config_name: str) -> dict:
    """Returns {behavior_idx: bool} — True if ANY of the 10 rounds succeeded."""
    results_dir = config.ASRK_RESULTS_DIRS[config_name]
    files = sorted(f for f in os.listdir(results_dir)
                   if re.match(r"asrk_b\d+_r\d+\.json", f))

    by_behavior = defaultdict(list)
    for fname in files:
        m = re.match(r"asrk_b(\d+)_r\d+\.json", fname)
        b_idx = int(m.group(1))
        with open(os.path.join(results_dir, fname), encoding="utf-8") as f:
            data = json.load(f)
        by_behavior[b_idx].append(data.get("success", False))

    return {b: any(rounds) for b, rounds in by_behavior.items()}


def mcnemar_test(config_a: str, config_b: str):
    succ_a = load_success_by_behavior(config_a)
    succ_b = load_success_by_behavior(config_b)

    common = sorted(set(succ_a) & set(succ_b))
    both = sum(1 for i in common if succ_a[i] and succ_b[i])
    a_only = sum(1 for i in common if succ_a[i] and not succ_b[i])
    b_only = sum(1 for i in common if not succ_a[i] and succ_b[i])
    neither = sum(1 for i in common if not succ_a[i] and not succ_b[i])

    print(f"n comportements comparés : {len(common)}")
    print(f"  Réussite dans les deux        : {both}")
    print(f"  Réussite {config_a} uniquement : {a_only}")
    print(f"  Réussite {config_b} uniquement : {b_only}")
    print(f"  Échec dans les deux            : {neither}")

    discordant = a_only + b_only
    if discordant == 0:
        print("\nAucune paire discordante, test non applicable.")
        return

    if discordant < 25:
        # exact binomial test on discordant pairs
        from math import comb
        n, k = discordant, min(a_only, b_only)
        p_value = 2 * sum(comb(n, i) * 0.5**n for i in range(0, k + 1))
        p_value = min(p_value, 1.0)
        print(f"\nTest exact (binomial), n discordant={discordant}")
    else:
        chi2 = (abs(a_only - b_only) - 1) ** 2 / discordant  # continuity correction
        from math import erf, sqrt
        z = math.sqrt(chi2)
        p_value = 1 - 0.5 * (1 + erf(z / sqrt(2)))
        p_value *= 2
        print(f"\nTest McNemar (chi2 avec correction de continuité)")

    print(f"p-value = {p_value:.4f} -> "
          f"{'SIGNIFICATIF' if p_value < 0.05 else 'non significatif'}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 mcnemar_test.py <config_a> <config_b>")
        sys.exit(1)
    mcnemar_test(sys.argv[1], sys.argv[2])