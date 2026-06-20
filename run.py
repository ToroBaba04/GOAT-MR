import json, time, os, re
from attacker import attacker_turn
from target import query_target
from judge import judge_conversation
from memory import (index_trajectory, retrieve_similar,
                    classify_goal, retrieve_category_profile)
import config


def get_next_session_id() -> str:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(config.RESULTS_DIR)
                if re.match(r"test_\d+\.json", f)]
    if not existing:
        return "test_001"
    indices = [int(re.search(r"\d+", f).group()) for f in existing]
    return f"test_{max(indices) + 1:03d}"


def run_goat_session(goal: str, category: str = "Unknown",
                     session_id: str = None) -> dict:

    if session_id is None:
        session_id = get_next_session_id()

    print(f"\n{'='*60}")
    print(f"Session : {session_id}")
    print(f"Goal    : {goal}")
    print(f"{'='*60}\n")

    # ── 1. Category: confirm provided or auto-classify ─────────────────────
    auto_classified = False
    auto_confidence = 0.0

    if config.USE_MEMORY:
        if category == "Unknown":
            inferred, conf = classify_goal(goal)
            if inferred != "Unknown":
                category = inferred
                auto_classified = True
                auto_confidence = conf
                print(f"[MEMORY] Category auto-classified: {category} "
                      f"(confidence={conf:.0%})")
            else:
                print(f"[MEMORY] Category could not be inferred (conf={conf:.0%})")
        else:
            # Provided category — optionally verify via memory
            inferred, conf = classify_goal(goal)
            if inferred != "Unknown" and inferred != category and conf >= 0.70:
                print(f"[MEMORY] Category provided='{category}' but memory "
                      f"suggests '{inferred}' (conf={conf:.0%}) — keeping provided.")

    print(f"Category: {category}")

    # ── 2. Retrieve memory context (semantic similarity) ──────────────────
    memory_context = []
    if config.USE_MEMORY:
        memory_context = retrieve_similar(goal=goal)
        if memory_context:
            sims = [round(t["similarity"], 3) for t in memory_context]
            print(f"[MEMORY] {len(memory_context)} similar trajectory(ies): {sims}")
        else:
            print("[MEMORY] No similar trajectory found.")

    # ── 3. Retrieve category attack profile ───────────────────────────────
    category_profile = None
    if config.USE_MEMORY and category != "Unknown":
        category_profile = retrieve_category_profile(category)
        if category_profile.get("total_entries", 0) > 0:
            t1r = category_profile["turn1_success_rate"]
            avg = category_profile["avg_first_unsafe_turn"]
            best = category_profile["best_first_technique"]
            print(f"[MEMORY] Category profile: {category_profile['total_entries']} "
                  f"entries | T1 success={t1r:.0%} | avg_turn={avg:.2f} | "
                  f"best_first={best}")
        else:
            print(f"[MEMORY] No profile data for category '{category}'.")
            category_profile = None

    # ── 4. Multi-turn attack loop ─────────────────────────────────────────
    conversation_history = []
    trajectory = []

    for turn in range(1, config.MAX_TURNS + 1):
        print(f"\n--- Turn {turn} ---")

        attacker_output = attacker_turn(
            goal=goal,
            conversation_history=conversation_history,
            turn_number=turn,
            memory_context=memory_context if turn == 1 else None,
            category=category,
            category_profile=category_profile if turn == 1 else None,
            auto_classified=auto_classified,
            auto_confidence=auto_confidence,
        )

        print(f"[OBSERVATION] {attacker_output['observation']}")
        print(f"[PROGRESS]    {attacker_output.get('progress_assessment', 'n/a')}")
        print(f"[THOUGHT]     {attacker_output['thought']}")
        techniques = attacker_output["strategy"].get("techniques", [])
        print(f"[STRATEGY]    {techniques} | "
              f"{attacker_output['strategy'].get('rationale', '')}")

        attack_msg = attacker_output["reply"]
        print(f"\n[REPLY TO TARGET]\n{attack_msg}\n")

        conversation_history.append({"role": "user", "content": attack_msg})
        target_response = query_target(conversation_history)
        conversation_history.append({"role": "assistant",
                                      "content": target_response})
        print(f"[TARGET]\n{target_response}\n")

        trajectory.append({
            "turn": turn,
            "observation": attacker_output["observation"],
            "progress_assessment": attacker_output.get("progress_assessment", "n/a"),
            "thought": attacker_output["thought"],
            "strategy": attacker_output["strategy"],
            "attack_message": attack_msg,
            "target_response": target_response,
        })

        time.sleep(0.5)

    # ── 5. Final judgment ─────────────────────────────────────────────────
    print(f"\n--- Final judgment ---")
    judgment = judge_conversation(goal=goal,
                                   conversation_history=conversation_history)

    for i, turn_verdict in enumerate(judgment["per_turn"]):
        trajectory[i]["judgment"] = {
            "unsafe": turn_verdict["unsafe"],
            "raw": turn_verdict["raw"],
        }
        marker = "UNSAFE" if turn_verdict["unsafe"] else "safe"
        print(f"  Turn {turn_verdict['turn']}: {marker}")

    success = judgment["success"]
    first_unsafe_turn = judgment["first_unsafe_turn"]

    if success:
        print(f"\n✓ Attack succeeded — first unsafe at turn {first_unsafe_turn}")
    else:
        print(f"\n✗ Attack failed — no unsafe response over {config.MAX_TURNS} turns")

    result = {
        "session_id": session_id,
        "goal": goal,
        "category": category,
        "success": success,
        "first_unsafe_turn": first_unsafe_turn,
        "total_turns": len(trajectory),
        "trajectory": trajectory,
    }

    filepath = f"{config.RESULTS_DIR}/{session_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {filepath}")

    # ── 6. Index successful trajectory ───────────────────────────────────
    if config.USE_MEMORY and success:
        # Collect techniques up to first_unsafe_turn (deduplicated, in order)
        techniques_until_success = []
        seen: set = set()
        for t in trajectory[:first_unsafe_turn]:
            for tech in t.get("strategy", {}).get("techniques", []):
                if tech not in seen:
                    techniques_until_success.append(tech)
                    seen.add(tech)

        # Per-turn precision: turn-1 techniques and winning-turn techniques
        turn1_techniques = (trajectory[0].get("strategy", {})
                            .get("techniques", []))
        winning_techniques = (trajectory[first_unsafe_turn - 1]
                              .get("strategy", {}).get("techniques", []))

        index_trajectory(
            session_id=session_id,
            goal=goal,
            category=category,
            first_unsafe_turn=first_unsafe_turn,
            techniques_used=techniques_until_success,
            successful_message=trajectory[first_unsafe_turn - 1]["attack_message"],
            turn1_techniques=turn1_techniques,
            winning_techniques=winning_techniques,
        )
        print(f"[MEMORY] Indexed: {len(techniques_until_success)} techniques "
              f"until T{first_unsafe_turn} | T1={turn1_techniques} | "
              f"winning={winning_techniques}")

    return result


if __name__ == "__main__":
    from jbb_loader import load_stratified_sample

    print(f"=== Active config: {config.ACTIVE_CONFIG} ===")
    print(f"  Results dir  : {config.RESULTS_DIR}")
    print(f"  Memory       : {config.USE_MEMORY}")
    print(f"  Reflection   : {config.USE_REFLECTION}\n")

    # Niveau 1 — validation rapide (5 catégories × 1) :
    behaviors = load_stratified_sample(per_category=1)[:5]

    # Niveau 2 — run intermédiaire (10 catégories × 2 = 20) :
    # behaviors = load_stratified_sample(per_category=2)

    # Niveau 3 — comparaison baseline (10 catégories × 5 = 50) :
    # behaviors = load_stratified_sample(per_category=5)

    # Niveau 4 — benchmark complet :
    # from jbb_loader import load_jailbreakbench_behaviors
    # behaviors = load_jailbreakbench_behaviors()

    print(f"Running {len(behaviors)} behaviors "
          f"across {len(set(b['category'] for b in behaviors))} categories\n")

    for b in behaviors:
        print(f"\n>>> JBB [{b['index']}] | {b['category']}")
        run_goat_session(goal=b["goal"], category=b["category"])
