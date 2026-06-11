from datasets import load_dataset
from collections import defaultdict

def load_jailbreakbench_behaviors(subset_size: int = None) -> list:
    """Charge tous les comportements harmful."""
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["harmful"]
    behaviors = []
    for i, row in enumerate(ds):
        behaviors.append({
            "index": i,
            "goal": row["Goal"],
            "category": row["Category"],
            "behavior": row["Behavior"],
        })
    if subset_size is not None:
        behaviors = behaviors[:subset_size]
    return behaviors


def load_stratified_sample(per_category: int = 1) -> list:
    """
    Échantillonne `per_category` comportement(s) par catégorie distincte.
    Retourne donc 10 comportements si per_category=1 (10 catégories dans JBB).
    """
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")["harmful"]
    by_category = defaultdict(list)
    for i, row in enumerate(ds):
        by_category[row["Category"]].append({
            "index": i,
            "goal": row["Goal"],
            "category": row["Category"],
            "behavior": row["Behavior"],
        })

    sample = []
    for category, items in sorted(by_category.items()):
        sample.extend(items[:per_category])
    return sample


if __name__ == "__main__":
    sample = load_stratified_sample(per_category=1)
    print(f"Total: {len(sample)} behaviors across {len(set(b['category'] for b in sample))} categories\n")
    for b in sample:
        print(f"[{b['index']:3d}] {b['category']:30s} | {b['goal'][:80]}")