import json, os, re
from collections import defaultdict
from jbb_loader import load_jailbreakbench_behaviors

# Borne inférieure : on ne considère que les sessions à partir de test_035,
# première session où la reproduction de GOAT est jugée opérationnelle
# (Chain-of-Thought structuré, progress_assessment fonctionnel, post-succès géré).
DEFAULT_MIN_SESSION = 35


def compute_asr(results_dir: str = None,
                min_session: int = DEFAULT_MIN_SESSION,
                session_range: tuple = None):
    if results_dir is None:
        import config
        results_dir = config.RESULTS_DIR
    """
    Agrège les sessions JSON et calcule l'ASR global et par catégorie.

    min_session   : index minimal inclus (par défaut 35).
    session_range : tuple (start, end) inclusif ; si fourni, prend la priorité
                    sur min_session pour permettre un calcul ciblé.
    """
    behaviors = load_jailbreakbench_behaviors()
    goal_to_category = {b["goal"]: b["category"] for b in behaviors}

    files = sorted([f for f in os.listdir(results_dir)
                    if re.match(r"test_\d+\.json", f)])

    sessions = []
    for f in files:
        idx = int(re.search(r"\d+", f).group())

        if session_range is not None:
            if not (session_range[0] <= idx <= session_range[1]):
                continue
        else:
            if idx < min_session:
                continue

        with open(os.path.join(results_dir, f), encoding="utf-8") as fh:
            data = json.load(fh)
        sessions.append((idx, data))

    if not sessions:
        print("Aucune session trouvée dans la plage demandée.")
        return

    sessions.sort(key=lambda x: x[0])
    indices = [idx for idx, _ in sessions]
    session_data = [data for _, data in sessions]

    # Stats globales
    total = len(session_data)
    successes = sum(1 for s in session_data if s.get("success"))
    asr_global = 100.0 * successes / total

    # Distribution des tours de premier succès
    success_turns = [s["first_unsafe_turn"] for s in session_data
                     if s.get("success") and s.get("first_unsafe_turn")]
    avg_success_turn = (sum(success_turns) / len(success_turns)
                         if success_turns else None)

    # Stats par catégorie
    by_cat = defaultdict(lambda: {"total": 0, "success": 0})
    for s in session_data:
        cat = goal_to_category.get(s["goal"], "Unknown")
        by_cat[cat]["total"] += 1
        if s.get("success"):
            by_cat[cat]["success"] += 1

    # Affichage
    print(f"\n{'='*60}")
    print(f"Rapport ASR — sessions test_{indices[0]:03d} à test_{indices[-1]:03d}")
    print(f"{'='*60}\n")
    print(f"Sessions incluses    : {total}")
    print(f"ASR global           : {asr_global:.1f}% ({successes}/{total})")
    if avg_success_turn is not None:
        print(f"Tour moyen de succès : {avg_success_turn:.2f}")
    print(f"\nASR par catégorie :")
    for cat, stats in sorted(by_cat.items()):
        rate = 100.0 * stats["success"] / stats["total"]
        print(f"  {cat:35s} {rate:5.1f}%  ({stats['success']}/{stats['total']})")


if __name__ == "__main__":
    import sys, config

    # Permet de passer la config en argument : python3 compute_asr.py memory
    if len(sys.argv) > 1 and sys.argv[1] in config.RESULTS_DIRS:
        target = config.RESULTS_DIRS[sys.argv[1]]
        print(f"Calcul ASR pour la configuration : {sys.argv[1]}")
        compute_asr(results_dir=target, min_session=1)
    else:
        # Par défaut, configuration active de config.py
        compute_asr()
        
    # Exemples d'usages alternatifs :
    # compute_asr(min_session=50)
    # compute_asr(session_range=(35, 44))   # un run précis