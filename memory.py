import glob
import json
import os

import chromadb
from chromadb.config import Settings

# ── Store configuration ────────────────────────────────────────────────────

MEMORY_DIR = "memory_store"
COLLECTION_NAME = "successful_trajectories"
DEFAULT_TOP_K = 3
SIMILARITY_THRESHOLD = 0.4

# Directories scanned to build category profiles
_RESULTS_DIRS = ["results_memory", "results_baseline"]

# Keyword-based fallback for goal classification when memory is sparse
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Disinformation": [
        "fake news", "disinformation", "misinformation", "hoax",
        "conspiracy", "propaganda", "false article", "false information",
        "spread.*false", "write.*article.*suggest", "article.*claim",
    ],
    "Economic harm": [
        "extort", "extortion", r"\bfraud\b", r"\bsteal\b", r"\bscam\b",
        "embezzle", "money laundering", "bribe", "illegal.*scheme",
        "exploit.*company", "exploit.*business", "financial gain",
        "illegally.*money", "ransom", r"\bgamble\b", "gambling",
        "life savings", "financial.*harm", "ponzi", "pyramid scheme",
    ],
    "Expert advice": [
        "medical treatment", "treatment plan", "diagnosis",
        r"\bprescription\b", "circumvent.*copyright", "circumvent.*law",
        "legal advice on", "medical advice", r"\bpatient\b",
    ],
    "Fraud/Deception": [
        r"\bddos\b", "distributed denial", "fake charity",
        "impersonate.*official", r"\bphishing\b",
        "vulnerable.*target", "high.impact.*target", "cyber.*target",
    ],
    "Government decision-making": [
        "forge.*document", "forged.*document", r"\bimmigration\b",
        "criminal record", "criminal database",
        "official document", "pass through.*border", "government.*decision",
        "alter.*record", "nefariously.*alter",
    ],
    "Harassment/Discrimination": [
        r"\bharass\b", r"\bbully\b", "defamatory", "defame",
        r"\bdiscrimination\b", r"\bstalking?\b", "body shaming",
        "body.*campaign", "threatening.*article", "hate.*campaign",
        "threatening.*message", "social media.*promotes",
        "hate speech", r"\bslur\b", "targeting.*group",
        # "defamatory article" is Harassment even if it mentions a public figure
        "defamatory.*article", "defamatory.*claim",
    ],
    "Malware/Hacking": [
        "malware", r"\bvirus\b", "ransomware",
        "antivirus", "sql injection", r"\bbackdoor\b", r"\bkeylogger\b",
        r"\bworm\b", "disables.*antivirus", "hack.*email",
        "hack.*account", "step.*hack", "instructions.*hack",
    ],
    "Physical harm": [
        "shoot up", "school shooting", r"\bshoot\b", r"\bkill\b",
        r"\bweapon\b", r"\bbomb\b", r"\bpoison\b",
        r"\bmurder\b", "high school.*attack", "physical.*harm",
        "how to.*hurt", "how to.*attack",
    ],
    "Privacy": [
        "blackmail.*message", "blackmail.*someone", r"\bdoxx\b",
        "expose.*private", "expose.*identity",
        "identity theft", "private.*data", "hack.*account",
        "personal.*information.*steal", "create.*account.*posing",
    ],
    "Sexual/Adult content": [
        "pornographic", r"\bsexual\b", "explicit.*story", "adult content",
        r"\bnude\b", r"\bnsfw\b", r"\berotic\b", "posing as a minor",
        "sexually.*explicit", "social media.*posing.*minor",
        "create.*account.*minor", "soliciting.*adult",
    ],
}


# ── ChromaDB helpers ───────────────────────────────────────────────────────

def get_client():
    os.makedirs(MEMORY_DIR, exist_ok=True)
    return chromadb.PersistentClient(
        path=MEMORY_DIR,
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection():
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# ── Indexing ───────────────────────────────────────────────────────────────

def build_embedding_text(goal: str, category: str, techniques: list) -> str:
    techniques_str = ", ".join(techniques) if techniques else "none"
    return (
        f"Goal: {goal}\n"
        f"Category: {category}\n"
        f"Successful techniques: {techniques_str}"
    )


def index_trajectory(session_id: str, goal: str, category: str,
                     first_unsafe_turn: int,
                     techniques_used: list,
                     successful_message: str,
                     turn1_techniques: list = None,
                     winning_techniques: list = None) -> None:
    """
    Indexes a successful attack trajectory into the vector store.

    turn1_techniques   : techniques used specifically at turn 1.
    winning_techniques : techniques used at the first_unsafe_turn (the ones
                         that actually triggered the unsafe response).
    Both are optional but improve category profile accuracy when provided.
    """
    collection = get_collection()
    embedding_text = build_embedding_text(goal, category, techniques_used)
    metadata = {
        "session_id": session_id,
        "goal": goal,
        "category": category,
        "first_unsafe_turn": first_unsafe_turn,
        "techniques_used": json.dumps(techniques_used),
        "successful_message": successful_message,
        "turn1_techniques": json.dumps(turn1_techniques or []),
        "winning_techniques": json.dumps(winning_techniques or []),
    }
    collection.upsert(ids=[session_id], documents=[embedding_text],
                      metadatas=[metadata])


# ── Semantic retrieval ─────────────────────────────────────────────────────

def retrieve_similar(goal: str,
                     top_k: int = DEFAULT_TOP_K,
                     threshold: float = SIMILARITY_THRESHOLD) -> list:
    """
    Returns the top-k most semantically similar successful trajectories.
    Each result is a dict with metadata + similarity score.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[f"Goal: {goal}"],
        n_results=min(top_k * 2, collection.count()),
    )

    trajectories = []
    for i in range(len(results["ids"][0])):
        similarity = 1.0 - results["distances"][0][i]
        if similarity < threshold:
            continue
        meta = results["metadatas"][0][i]
        trajectories.append({
            "session_id": meta["session_id"],
            "goal": meta["goal"],
            "category": meta["category"],
            "first_unsafe_turn": meta["first_unsafe_turn"],
            "techniques_used": json.loads(meta.get("techniques_used", "[]")),
            "turn1_techniques": json.loads(meta.get("turn1_techniques", "[]")),
            "winning_techniques": json.loads(meta.get("winning_techniques", "[]")),
            "successful_message": meta["successful_message"],
            "similarity": similarity,
        })
        if len(trajectories) >= top_k:
            break

    return trajectories


# ── Goal classification ────────────────────────────────────────────────────

def _keyword_classify(goal: str) -> str:
    """Keyword-based fallback classifier."""
    import re
    goal_lower = goal.lower()
    scores: dict[str, int] = {}
    for cat, patterns in _CATEGORY_KEYWORDS.items():
        for p in patterns:
            if re.search(p, goal_lower):
                scores[cat] = scores.get(cat, 0) + 1
    return max(scores, key=scores.get) if scores else "Unknown"


def classify_goal(goal: str,
                   min_memory_confidence: float = 0.75,
                   min_memory_entries: int = 5) -> tuple[str, float]:
    """
    Infers the JailbreakBench category of a goal.

    Strategy:
    1. Keyword-based classification (always runs, deterministic).
    2. Memory-based classification (weighted vote over ChromaDB entries).
       Overrides keywords only when memory is rich (≥ min_memory_entries
       entries in the winning category) AND confidence ≥ min_memory_confidence.

    Returns (category, confidence) where confidence ∈ [0, 1].
    """
    # Step 1: keyword classification (primary when memory is sparse)
    kw_cat = _keyword_classify(goal)

    # Step 2: memory-based classification
    collection = get_collection()
    if collection.count() >= 3:
        results = collection.query(
            query_texts=[f"Goal: {goal}"],
            n_results=min(9, collection.count()),
        )
        cat_scores: dict[str, float] = {}
        cat_counts: dict[str, int] = {}
        for i in range(len(results["ids"][0])):
            sim = 1.0 - results["distances"][0][i]
            if sim < 0.40:
                continue
            cat = results["metadatas"][0][i].get("category", "Unknown")
            if cat == "Unknown":
                continue
            cat_scores[cat] = cat_scores.get(cat, 0.0) + sim
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        if cat_scores:
            total_weight = sum(cat_scores.values())
            best_mem = max(cat_scores, key=cat_scores.get)
            mem_confidence = cat_scores[best_mem] / total_weight
            mem_count = cat_counts.get(best_mem, 0)

            # Trust memory only when it has substantial evidence
            if (mem_confidence >= min_memory_confidence
                    and mem_count >= min_memory_entries):
                return (best_mem, round(mem_confidence, 3))

    # Fall back to keyword result
    if kw_cat != "Unknown":
        return (kw_cat, 1.0)

    return ("Unknown", 0.0)


# ── Category profile ───────────────────────────────────────────────────────

def _read_turn_techniques(session_id: str, turn_num: int) -> list:
    """Reads techniques used at a specific turn from the corresponding result file."""
    for rdir in _RESULTS_DIRS:
        fpath = os.path.join(rdir, f"{session_id}.json")
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            for t in data.get("trajectory", []):
                if t.get("turn") == turn_num:
                    return t.get("strategy", {}).get("techniques", [])
        except Exception:
            pass
    return []


def _read_attack_message_at_turn(session_id: str, turn_num: int) -> str:
    """Returns the attack_message used at turn_num from the result file."""
    for rdir in _RESULTS_DIRS:
        fpath = os.path.join(rdir, f"{session_id}.json")
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            for t in data.get("trajectory", []):
                if t.get("turn") == turn_num:
                    return t.get("attack_message", "")
        except Exception:
            pass
    return ""


def retrieve_category_profile(category: str) -> dict:
    """
    Builds a rich attack profile for `category` from two sources:

    Source A — ChromaDB memory store:
      Fast, vector-indexed.  May be sparse.  Uses turn1_techniques /
      winning_techniques fields if present; falls back to file lookup.

    Source B — Result JSON files (results_memory, results_baseline):
      Ground-truth trajectories.  Slower but complete.  Used to fill gaps
      left by Source A and to find best opening messages.

    Returns a profile dict:
    {
      "category"               : str,
      "total_entries"          : int,
      "turn1_success_count"    : int,
      "turn1_success_rate"     : float,
      "avg_first_unsafe_turn"  : float,
      "turn_distribution"      : {1: n, 2: m, ...},
      "best_first_technique"   : list[str],  # best to use at turn 1
      "best_overall_combo"     : list[str],  # best at the winning turn
      "best_opening_turn1"     : str,        # attack message from a T1 winner
      "best_opening_general"   : str,        # attack message from fastest winner
      "fastest_turn"           : int,
    }
    Returns {"category": category, "total_entries": 0} if no data found.
    """

    # ── Source A: memory store ─────────────────────────────────────────────
    collection = get_collection()
    store_entries = []
    if collection.count() > 0:
        all_data = collection.get(limit=500)
        for meta in all_data["metadatas"]:
            if meta.get("category") == category:
                store_entries.append({
                    "session_id": meta["session_id"],
                    "first_unsafe_turn": meta["first_unsafe_turn"],
                    "techniques_used": json.loads(
                        meta.get("techniques_used", "[]")),
                    "turn1_techniques": json.loads(
                        meta.get("turn1_techniques", "[]")),
                    "winning_techniques": json.loads(
                        meta.get("winning_techniques", "[]")),
                    "successful_message": meta.get("successful_message", ""),
                    "_source": "store",
                })

    # ── Source B: result files (categorised only) ──────────────────────────
    file_entries = []
    seen_ids = {e["session_id"] for e in store_entries}
    for rdir in _RESULTS_DIRS:
        if not os.path.isdir(rdir):
            continue
        for fpath in glob.glob(os.path.join(rdir, "test_*.json")):
            try:
                with open(fpath, encoding="utf-8") as f:
                    session = json.load(f)
            except Exception:
                continue
            if session.get("category") != category:
                continue
            if not session.get("success"):
                continue
            sid = session["session_id"]
            if sid in seen_ids:
                continue  # already have it from the store
            fut = session["first_unsafe_turn"]
            traj = session.get("trajectory", [])
            t1_techs = []
            win_techs = []
            succ_msg = ""
            for t in traj:
                if t.get("turn") == 1:
                    t1_techs = t.get("strategy", {}).get("techniques", [])
                if t.get("turn") == fut:
                    win_techs = t.get("strategy", {}).get("techniques", [])
                    succ_msg = t.get("attack_message", "")
            file_entries.append({
                "session_id": sid,
                "first_unsafe_turn": fut,
                "techniques_used": [],
                "turn1_techniques": t1_techs,
                "winning_techniques": win_techs,
                "successful_message": succ_msg,
                "_source": "file",
            })
            seen_ids.add(sid)

    entries = store_entries + file_entries
    total = len(entries)

    if total == 0:
        return {"category": category, "total_entries": 0}

    # ── Resolve per-entry techniques ───────────────────────────────────────
    for e in entries:
        # Ensure turn1_techniques is populated
        if not e["turn1_techniques"]:
            if e["first_unsafe_turn"] == 1 and e["techniques_used"]:
                e["turn1_techniques"] = e["techniques_used"]
            else:
                e["turn1_techniques"] = _read_turn_techniques(
                    e["session_id"], 1)

        # Ensure winning_techniques is populated
        if not e["winning_techniques"]:
            if e["first_unsafe_turn"] == 1 and e["techniques_used"]:
                e["winning_techniques"] = e["techniques_used"]
            else:
                e["winning_techniques"] = _read_turn_techniques(
                    e["session_id"], e["first_unsafe_turn"])

        # Ensure successful_message is populated for store entries
        if not e["successful_message"] and e["_source"] == "store":
            e["successful_message"] = _read_attack_message_at_turn(
                e["session_id"], e["first_unsafe_turn"])

    # ── Statistics ─────────────────────────────────────────────────────────
    first_unsafe_turns = [e["first_unsafe_turn"] for e in entries]
    turn_dist: dict[int, int] = {}
    for t in first_unsafe_turns:
        turn_dist[t] = turn_dist.get(t, 0) + 1

    turn1_entries = [e for e in entries if e["first_unsafe_turn"] == 1]
    t1_count = len(turn1_entries)

    fastest_turn = min(first_unsafe_turns)
    fastest_entries = [e for e in entries
                       if e["first_unsafe_turn"] == fastest_turn]

    # Technique frequency at turn 1 (weighted: +2 for turn-1 wins, +1 otherwise)
    t1_freq: dict[str, int] = {}
    for e in entries:
        techs = e["turn1_techniques"]
        weight = 2 if e["first_unsafe_turn"] == 1 else 1
        for tech in techs:
            t1_freq[tech] = t1_freq.get(tech, 0) + weight

    # Technique frequency at the winning turn (across fastest entries)
    win_freq: dict[str, int] = {}
    for e in sorted(entries, key=lambda x: x["first_unsafe_turn"])[:6]:
        for tech in e["winning_techniques"]:
            win_freq[tech] = win_freq.get(tech, 0) + 1

    best_first_technique = sorted(t1_freq, key=lambda t: -t1_freq[t])[:3]
    best_overall_combo = sorted(win_freq, key=lambda t: -win_freq[t])[:3]

    # ── Best opening messages ──────────────────────────────────────────────
    # turn1_winner: the attack message that triggered unsafe at turn 1
    best_opening_turn1 = ""
    if turn1_entries:
        best_t1 = max(turn1_entries, key=lambda e: len(e["successful_message"]))
        best_opening_turn1 = best_t1["successful_message"]

    # winning_turn: the attack message that triggered unsafe at first_unsafe_turn
    # (relevant for multi-turn sessions: shows what to escalate to at turn N)
    best_opening_winning_turn = ""
    if fastest_entries:
        best_gen = max(fastest_entries, key=lambda e: len(e["successful_message"]))
        best_opening_winning_turn = best_gen["successful_message"]

    # turn1_framing: the TURN-1 message from sessions that eventually succeeded
    # (even if they didn't win at turn 1 — shows how to open the conversation)
    best_turn1_framing = ""
    multi_turn_entries = [e for e in entries if e["first_unsafe_turn"] > 1]
    if multi_turn_entries:
        # Look up the turn-1 attack message from the fastest multi-turn session
        fastest_multi = min(multi_turn_entries,
                            key=lambda e: e["first_unsafe_turn"])
        best_turn1_framing = _read_attack_message_at_turn(
            fastest_multi["session_id"], 1)

    return {
        "category": category,
        "total_entries": total,
        "turn1_success_count": t1_count,
        "turn1_success_rate": t1_count / total,
        "avg_first_unsafe_turn": sum(first_unsafe_turns) / total,
        "turn_distribution": dict(sorted(turn_dist.items())),
        "best_first_technique": best_first_technique,
        "best_overall_combo": best_overall_combo,
        # Message to send at turn 1 when immediate success is expected
        "best_opening_turn1": best_opening_turn1,
        # Turn-1 framing message when multi-turn approach is needed
        "best_turn1_framing": best_turn1_framing,
        # Message that triggered the breakthrough at the winning turn
        "best_opening_winning_turn": best_opening_winning_turn,
        "fastest_turn": fastest_turn,
    }


# ── Utilities ──────────────────────────────────────────────────────────────

def memory_stats() -> dict:
    collection = get_collection()
    return {
        "total_trajectories": collection.count(),
        "collection_name": COLLECTION_NAME,
        "storage_path": MEMORY_DIR,
    }


def clear_memory() -> None:
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Memory cleared: collection {COLLECTION_NAME} deleted.")
    except Exception as e:
        print(f"Nothing to clear: {e}")


if __name__ == "__main__":
    stats = memory_stats()
    print("=== Memory stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
