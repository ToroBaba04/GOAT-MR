import os, sys, time
import config
import run  # the run.py module itself
from run import run_goat_session
from jbb_loader import load_jailbreakbench_behaviors

K = 10  # independent rounds per behavior


def freeze_indexation():
    """
    Disables memory/reflection indexation for the duration of the ASR@k run,
    so that all K rounds for a given behavior are evaluated under the exact
    same (frozen) memory state, isolating pure stochastic variability.
    Retrieval (reading memory/reflection) stays fully active.
    """
    run.index_trajectory = lambda *a, **k: None
    run.index_reflection = lambda *a, **k: None
    print("[ASR@K] Indexation frozen: memory/reflection state will not "
          "change during this run.")


def get_asrk_dir() -> str:
    return f"{config.RESULTS_DIR}_asrk"


def format_eta(elapsed_s: float, done: int, total: int) -> str:
    if done == 0:
        return "unknown"
    rate = elapsed_s / done
    remaining_s = rate * (total - done)
    hours = remaining_s / 3600
    return f"{hours:.1f}h remaining (avg {rate:.1f}s/conversation)"


def run_asr_k():
    results_dir = get_asrk_dir()
    os.makedirs(results_dir, exist_ok=True)

    lock_file = f".run_lock_{config.ACTIVE_CONFIG}_asrk"
    if os.path.exists(lock_file):
        print(f"[ERROR] An ASR@{K} run for '{config.ACTIVE_CONFIG}' is "
              f"already in progress (lock file exists).")
        with open(lock_file) as f:
            print(f"[ERROR] Lock held by PID: {f.read().strip()}")
        sys.exit(1)
    with open(lock_file, "w") as f:
        f.write(str(os.getpid()))

    freeze_indexation()
    start_time = time.time()

    try:
        behaviors = load_jailbreakbench_behaviors()  # full 100
        n_total = len(behaviors) * K
        print(f"\n=== ASR@{K} run — config: {config.ACTIVE_CONFIG} ===")
        print(f"Results dir : {results_dir}")
        print(f"Behaviors   : {len(behaviors)}")
        print(f"Rounds/beh. : {K}")
        print(f"Total convos: {n_total}\n")

        failed = []
        done_count = 0
        skipped_count = 0

        for b_idx, b in enumerate(behaviors, 1):
            for round_num in range(1, K + 1):
                session_id = f"asrk_b{b_idx:03d}_r{round_num:02d}"
                out_path = os.path.join(results_dir, f"{session_id}.json")

                if os.path.exists(out_path):
                    done_count += 1
                    skipped_count += 1
                    continue  # resumable: already done, skip silently

                print(f"\n>>> Behavior {b_idx}/{len(behaviors)} "
                      f"({b['category']}) — round {round_num}/{K} "
                      f"[{done_count}/{n_total} total done]")

                original_results_dir = config.RESULTS_DIR
                config.RESULTS_DIR = results_dir
                try:
                    run_goat_session(
                        goal=b["goal"],
                        category=b["category"],
                        session_id=session_id,
                    )
                    done_count += 1
                except Exception as e:
                    print(f"[BATCH ERROR] {session_id} failed: {e}")
                    failed.append({"session_id": session_id, "error": str(e)})
                finally:
                    config.RESULTS_DIR = original_results_dir

                if done_count % 25 == 0:
                    elapsed = time.time() - start_time
                    print(f"\n[PROGRESS] {done_count}/{n_total} "
                          f"({100*done_count/n_total:.1f}%) — "
                          f"{format_eta(elapsed, done_count - skipped_count, n_total - skipped_count)}\n")

        print(f"\n{'='*60}")
        print(f"Done: {done_count}/{n_total} (skipped {skipped_count} already-complete rounds)")
        if failed:
            print(f"{len(failed)} round(s) failed:")
            for f_ in failed:
                print(f"  {f_['session_id']}: {f_['error'][:100]}")
        print(f"{'='*60}")

    finally:
        if os.path.exists(lock_file):
            os.remove(lock_file)


if __name__ == "__main__":
    run_asr_k()