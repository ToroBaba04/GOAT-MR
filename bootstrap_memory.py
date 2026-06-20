import json, os, re
from memory import index_trajectory, memory_stats, clear_memory
from jbb_loader import load_jailbreakbench_behaviors

BASELINE_DIR = "results_baseline"
MIN_SESSION = 35  # seuil de qualité validé

def bootstrap():
    # Index inversé goal -> category
    behaviors = load_jailbreakbench_behaviors()
    goal_to_category = {b["goal"]: b["category"] for b in behaviors}

    clear_memory()

    files = sorted(f for f in os.listdir(BASELINE_DIR)
                    if re.match(r"test_\d+\.json", f))

    indexed = 0
    skipped = 0
    for f in files:
        idx = int(re.search(r"\d+", f).group())
        if idx < MIN_SESSION:
            skipped += 1
            continue

        with open(os.path.join(BASELINE_DIR, f), encoding="utf-8") as fh:
            data = json.load(fh)

        if not data.get("success"):
            continue

        first_unsafe = data["first_unsafe_turn"]
        trajectory = data["trajectory"]

        techniques = []
        seen = set()
        for t in trajectory[:first_unsafe]:
            for tech in t.get("strategy", {}).get("techniques", []):
                if tech not in seen:
                    techniques.append(tech)
                    seen.add(tech)

        category = goal_to_category.get(data["goal"], "Unknown")

        index_trajectory(
            session_id=data["session_id"],
            goal=data["goal"],
            category=category,
            first_unsafe_turn=first_unsafe,
            techniques_used=techniques,
            successful_message=trajectory[first_unsafe - 1]["attack_message"],
        )
        indexed += 1

    print(f"\nMémoire amorcée :")
    print(f"  Sessions ignorées (avant test_{MIN_SESSION:03d}) : {skipped}")
    print(f"  Trajectoires réussies indexées : {indexed}")
    print(f"  État final : {memory_stats()}")

if __name__ == "__main__":
    bootstrap()