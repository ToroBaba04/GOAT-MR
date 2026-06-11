import json, time, os, re
from datetime import datetime
from attacker import attacker_turn
from target import query_target
from judge import judge_response
import config


def get_next_session_id() -> str:
    os.makedirs("results", exist_ok=True)
    existing = [f for f in os.listdir("results")
                if re.match(r"test_\d+\.json", f)]
    if not existing:
        return "test_001"
    indices = [int(re.search(r"\d+", f).group()) for f in existing]
    return f"test_{max(indices) + 1:03d}"


def run_goat_session(goal: str, session_id: str = None) -> dict:
    if session_id is None:
        session_id = get_next_session_id()

    print(f"\n{'='*60}")
    print(f"Session: {session_id}")
    print(f"Goal: {goal}")
    print(f"{'='*60}\n")

    conversation_history = []
    trajectory = []
    success = False
    success_turn = None

    for turn in range(1, config.MAX_TURNS + 1):
        print(f"--- Turn {turn} ---")

        # 1. Raisonnement complet de l'attaquant en un seul appel
        attacker_output = attacker_turn(
            goal=goal,
            conversation_history=conversation_history,
            turn_number=turn
        )

        print(f"[OBSERVATION] {attacker_output['observation']}")
        print(f"[THOUGHT]     {attacker_output['thought']}")
        techniques = attacker_output['strategy'].get('techniques', [])
        print(f"[STRATEGY]    techniques={techniques} | "
              f"{attacker_output['strategy'].get('rationale', '')}")

        attack_msg = attacker_output['reply']
        print(f"\n[REPLY TO TARGET]\n{attack_msg}\n")

        # 2. Réponse du modèle cible
        conversation_history.append({"role": "user", "content": attack_msg})
        target_response = query_target(conversation_history)
        conversation_history.append({"role": "assistant",
                                      "content": target_response})
        print(f"[TARGET]\n{target_response}\n")

        # 3. Jugement
        judgment = judge_response(
            goal=goal,
            response=target_response,
            attack_message=attack_msg
        )
        print(f"[JUDGE] success={judgment['success']} | "
              f"confidence={judgment.get('confidence', '?')} | "
              f"{judgment['reason']}\n")

        # 4. Log du tour
        trajectory.append({
            "turn": turn,
            "observation": attacker_output['observation'],
            "thought": attacker_output['thought'],
            "strategy": attacker_output['strategy'],
            "attack_message": attack_msg,
            "target_response": target_response,
            "judgment": judgment,
        })

        # 5. Succès → arrêt
        if judgment["success"]:
            success = True
            success_turn = turn
            print(f"✓ Success at turn {turn}\n")
            break

        time.sleep(0.5)

    result = {
        "session_id": session_id,
        "goal": goal,
        "success": success,
        "success_turn": success_turn,
        "total_turns": len(trajectory),
        "trajectory": trajectory,
    }

    filepath = f"results/{session_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {filepath}")
    return result


if __name__ == "__main__":
    from jbb_loader import load_stratified_sample

    behaviors = load_stratified_sample(per_category=1)

    print(f"Running on {len(behaviors)} behaviors across "
          f"{len(set(b['category'] for b in behaviors))} categories\n")

    for b in behaviors:
        print(f"\n>>> JBB [{b['index']}] | {b['category']}")
        run_goat_session(goal=b["goal"])