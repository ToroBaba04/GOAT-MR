from datasets import load_dataset

def load_jailbreakbench_behaviors(subset_size: int = None) -> list:
    """
    Charge les comportements harmful de JailbreakBench.
    Retourne une liste de dicts : {"goal", "category", "behavior", "index"}
    """
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


if __name__ == "__main__":
    behaviors = load_jailbreakbench_behaviors(subset_size=5)
    print(f"Total chargé : {len(behaviors)}")
    for b in behaviors:
        print(f"\n[{b['index']}] Catégorie : {b['category']}")
        print(f"  Goal     : {b['goal']}")
        print(f"  Behavior : {b['behavior']}")