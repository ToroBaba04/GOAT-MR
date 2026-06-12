import json, os, re
from collections import defaultdict
from jbb_loader import load_jailbreakbench_behaviors


def compute_asr(results_dir: str = "results", session_range: tuple = None):
    """
    Agrège les sessions JSON et calcule l'ASR global et par catégorie.
    session_range : tuple (start, end) inclusif sur les indices test_XXX,
                    ou None pour toutes les sessions.
    """
    # Index goal -> category depuis JailbreakBench
    behaviors = load_jailbreakbench_behaviors()
    goal_to_category = {b["goal"]: b["category"] for b in behaviors}

    files = sorted([f for f in os.listdir(results_dir)
                    if re.match(r"test_\d+\.json", f)])

    sessions = []
    for f in files:
        idx = int(re.search(r"\d+", f).group())
        if session_range and not (session_range[0] <= idx <= session_range[1]):
            continue
        with open(os.path.join(results_dir, f), encoding="utf-8") as fh:
            data = json.load(fh)
        sessions.append(data)

    if not sessions:
        print("No sessions found in the specified range.")
        return

    # Stats globales
    total = len(sessions)
    successes = sum(1 for s in sessions if s.get("success"))
    asr_global = 100.0 * successes / total

    # Distribution des tours de succès
    success_turns = [s["first_unsafe_turn"] for s in sessions
                     if s.get("success") and s.get("first_unsafe_turn")]
    avg_success_turn = (sum(success_turns) / len(success_turns)
                         if success_turns else None)

    # Stats par catégorie
    by_cat = defaultdict(lambda: {"total": 0, "success": 0})
    for s in sessions:
        cat = goal_to_category.get(s["goal"], "Unknown")
        by_cat[cat]["total"] += 1
        if s.get("success"):
            by_cat[cat]["success"] += 1

    # Affichage
    print(f"\n{'='*60}")
    print(f"ASR Report — {total} sessions")
    print(f"{'='*60}\n")
    print(f"ASR global          : {asr_global:.1f}% ({successes}/{total})")
    if avg_success_turn:
        print(f"Tours moyen succès  : {avg_success_turn:.2f}")
    print(f"\nASR par catégorie :")
    for cat, stats in sorted(by_cat.items()):
        rate = 100.0 * stats["success"] / stats["total"]
        print(f"  {cat:35s} {rate:5.1f}% "
              f"({stats['success']}/{stats['total']})")


if __name__ == "__main__":
    # Toutes les sessions
    compute_asr()

    # Ou : compute_asr(session_range=(29, 33))  pour un run spécifique