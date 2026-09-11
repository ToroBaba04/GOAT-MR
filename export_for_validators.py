import json, os, csv, random
from pathlib import Path

RESULTS_DIRS = {
    "baseline": "results_baseline",
    "memory": "results_memory",
    "reflection": "results_reflection",
    "goat_mr": "results_goat_mr",
}

OUTPUT_DIR = "validator_export"
OVERLAP_RATIO = 0.15
N_VALIDATORS = 3
RANDOM_SEED = 42  # reproducibility of the shuffle, to document in the memoir


def load_all_conversations() -> list:
    conversations = []
    for config_name, rdir in RESULTS_DIRS.items():
        if not os.path.isdir(rdir):
            continue
        for fname in sorted(os.listdir(rdir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(rdir, fname), encoding="utf-8") as f:
                data = json.load(f)
            conversations.append({
                "true_config": config_name,
                "true_session_id": data["session_id"],
                "goal": data["goal"],
                "category": data.get("category", "Unknown"),
                "trajectory": data["trajectory"],
                "judge_success": data.get("success"),
            })
    return conversations


def format_transcript(trajectory: list) -> str:
    lines = []
    for t in trajectory:
        lines.append(f"--- Turn {t['turn']} ---")
        lines.append(f"[ATTACKER MESSAGE]\n{t['attack_message']}\n")
        lines.append(f"[TARGET RESPONSE]\n{t['target_response']}\n")
    return "\n".join(lines)


def anonymize_and_split(conversations: list):
    random.seed(RANDOM_SEED)
    shuffled = conversations[:]
    random.shuffle(shuffled)

    for i, c in enumerate(shuffled, 1):
        c["anon_id"] = f"CONV_{i:04d}"

    n_total = len(shuffled)
    n_overlap = round(n_total * OVERLAP_RATIO)
    overlap_pool = shuffled[:n_overlap]
    unique_pool = shuffled[n_overlap:]

    chunk_size = len(unique_pool) // N_VALIDATORS
    validator_unique = []
    for v in range(N_VALIDATORS):
        start = v * chunk_size
        end = (start + chunk_size if v < N_VALIDATORS - 1 else len(unique_pool))
        validator_unique.append(unique_pool[start:end])

    validator_exports = []
    for v in range(N_VALIDATORS):
        combined = overlap_pool + validator_unique[v]
        random.shuffle(combined)
        validator_exports.append(combined)

    answer_key = {c["anon_id"]: {
        "true_config": c["true_config"],
        "true_session_id": c["true_session_id"],
        "judge_success": c["judge_success"],
        "category": c["category"],
        "in_overlap": c in overlap_pool,
    } for c in shuffled}

    return validator_exports, answer_key, [c["anon_id"] for c in overlap_pool]


def write_validator_csv(export_list: list, filepath: str):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "anon_id", "goal", "transcript",
            "turn1_verdict", "turn2_verdict", "turn3_verdict",
            "turn4_verdict", "turn5_verdict",
            "overall_verdict (safe/unsafe)", "notes",
        ])
        for c in export_list:
            writer.writerow([
                c["anon_id"],
                c["goal"],
                format_transcript(c["trajectory"]),
                "", "", "", "", "",
                "",
                "",
            ])


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    conversations = load_all_conversations()
    print(f"Loaded {len(conversations)} conversations across "
          f"{len(RESULTS_DIRS)} configurations.")

    validator_exports, answer_key, overlap_ids = anonymize_and_split(conversations)

    for i, export_list in enumerate(validator_exports, 1):
        filepath = os.path.join(OUTPUT_DIR, f"validator_{i}.csv")
        write_validator_csv(export_list, filepath)
        n_overlap_in_this = sum(1 for c in export_list if c["anon_id"] in overlap_ids)
        print(f"validator_{i}.csv : {len(export_list)} conversations "
              f"({n_overlap_in_this} shared for Fleiss' kappa)")

    with open(os.path.join(OUTPUT_DIR, "ANSWER_KEY_SECRET.json"),
              "w", encoding="utf-8") as f:
        json.dump({
            "answer_key": answer_key,
            "overlap_ids": overlap_ids,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nOverlap pool size: {len(overlap_ids)} "
          f"({100*len(overlap_ids)/len(conversations):.1f}% of total)")
    print(f"Secret answer key saved to "
          f"{OUTPUT_DIR}/ANSWER_KEY_SECRET.json — do NOT share this file.")


if __name__ == "__main__":
    main()