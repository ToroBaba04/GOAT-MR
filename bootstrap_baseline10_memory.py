import json, os, re
from collections import defaultdict
from memory import index_trajectory, memory_stats, clear_memory
from reflection import generate_reflection, index_reflection, reflection_stats, clear_reflections
from jbb_loader import load_jailbreakbench_behaviors

SOURCE_DIR = "results_baseline_asrk"
CLEAR_BEFORE_BOOTSTRAP = True
GENERATE_REFLECTIONS_FOR_TOTAL_FAILURES = True


def load_rounds_by_behavior(source_dir: str) -> dict:
    """
    Groups all asrk_b{idx}_r{num}.json files by behavior index.
    Returns {behavior_idx: {round_num: session_data}}.
    """
    files = sorted(f for f in os.listdir(source_dir)
                   if re.match(r"asrk_b\d+_r\d+\.json", f))

    by_behavior = defaultdict(dict)
    for fname in files:
        m = re.match(r"asrk_b(\d+)_r(\d+)\.json", fname)
        b_idx, r_num = int(m.group(1)), int(m.group(2))
        with open(os.path.join(source_dir, fname), encoding="utf-8") as f:
            data = json.load(f)
        by_behavior[b_idx][r_num] = data

    return by_behavior


def bootstrap_baseline10_memory(source_dir: str = SOURCE_DIR,
                                clear_first: bool = CLEAR_BEFORE_BOOTSTRAP,
                                do_reflections: bool = GENERATE_REFLECTIONS_FOR_TOTAL_FAILURES):
    """
    Seeds the memory store from the baseline ASR@10 run, using ONLY the
    earliest successful round per behavior (never all successful rounds),
    to avoid over-representing behaviors that succeed easily and often.

    Behaviors that never succeeded across all 10 rounds are, optionally,
    used to generate reflection lessons (cheap here since the failing
    subset is small — typically under 10 behaviors out of 100).
    """
    behaviors = load_jailbreakbench_behaviors()
    goal_to_category = {b["goal"]: b["category"] for b in behaviors}

    if clear_first:
        clear_memory()
        clear_reflections()

    by_behavior = load_rounds_by_behavior(source_dir)
    if not by_behavior:
        print(f"No ASR@k result files found in {source_dir}. Nothing to do.")
        return

    indexed_success = 0
    indexed_failure = 0
    total_failures_found = 0

    for b_idx, rounds in sorted(by_behavior.items()):
        # Earliest successful round for this behavior, if any
        successful_rounds = sorted(
            r for r, data in rounds.items() if data.get("success")
        )

        if successful_rounds:
            first_success_round = successful_rounds[0]
            data = rounds[first_success_round]

            goal = data["goal"]
            category = data.get("category") or goal_to_category.get(goal, "Unknown")
            trajectory = data["trajectory"]
            first_unsafe = data["first_unsafe_turn"]

            techniques_until_success = []
            seen: set = set()
            for t in trajectory[:first_unsafe]:
                for tech in t.get("strategy", {}).get("techniques", []):
                    if tech not in seen:
                        techniques_until_success.append(tech)
                        seen.add(tech)

            turn1_techniques = (trajectory[0].get("strategy", {})
                                .get("techniques", []))
            winning_techniques = (trajectory[first_unsafe - 1]
                                  .get("strategy", {}).get("techniques", []))

            index_trajectory(
                session_id=data["session_id"],
                goal=goal,
                category=category,
                first_unsafe_turn=first_unsafe,
                techniques_used=techniques_until_success,
                successful_message=trajectory[first_unsafe - 1]["attack_message"],
                turn1_techniques=turn1_techniques,
                winning_techniques=winning_techniques,
            )
            indexed_success += 1

        else:
            # Never succeeded across all 10 rounds — a genuine hard case,
            # worth a reflection lesson if the option is enabled.
            total_failures_found += 1
            if do_reflections:
                # Use round 1's conversation as the basis for the lesson
                data = rounds[min(rounds.keys())]
                goal = data["goal"]
                category = data.get("category") or goal_to_category.get(goal, "Unknown")
                trajectory = data["trajectory"]

                conversation_history = []
                for t in trajectory:
                    conversation_history.append(
                        {"role": "user", "content": t["attack_message"]})
                    conversation_history.append(
                        {"role": "assistant", "content": t["target_response"]})

                all_techniques = []
                seen_techs: set = set()
                for t in trajectory:
                    for tech in t.get("strategy", {}).get("techniques", []):
                        if tech not in seen_techs:
                            all_techniques.append(tech)
                            seen_techs.add(tech)

                lesson = generate_reflection(
                    goal=goal,
                    category=category,
                    conversation_history=conversation_history,
                    techniques_tried=all_techniques,
                )
                index_reflection(session_id=data["session_id"], goal=goal,
                                 category=category, lesson=lesson)
                indexed_failure += 1

    print(f"\nBootstrap from {source_dir} complete:")
    print(f"  Behaviors with at least one success : {indexed_success}")
    print(f"  Behaviors that failed all 10 rounds  : {total_failures_found}")
    if do_reflections:
        print(f"  Reflection lessons generated         : {indexed_failure}")
    print(f"\nMemory store state     : {memory_stats()}")
    print(f"Reflection store state : {reflection_stats()}")


if __name__ == "__main__":
    bootstrap_baseline10_memory()