# GOAT-MR — GOAT with Memory and Reflection

GOAT-MR is a research framework for **automated multi-turn red-teaming of large
language models**. It extends GOAT (*Generative Offensive Agent Tester*, Meta,
2024), where an LLM "attacker" converses with a target model over several turns
and adapts its jailbreak strategy at each step. GOAT-MR adds two cumulative
learning mechanisms so the attacker improves across sessions instead of starting
from scratch each time:

- **Memory module**: successful attack trajectories are stored in a persistent
  vector database (ChromaDB). For a new objective, the attacker retrieves the
  most semantically similar past successes, plus a per-category profile
  (which techniques work best, at which turn, and how often).
- **Reflection module**: when a session fails, the attacker model analyses the
  conversation and stores a structured lesson (failure summary, point of
  strongest resistance, techniques that did not work, recommendations). Lessons
  from similar past failures are injected into later attacks.

The project is built as an ablation study: the four configurations below are
compared on the same benchmark, with the same attacker, target and judge.

| Configuration | Memory | Reflection |
|---------------|:------:|:----------:|
| `baseline`    |   no   |     no     |
| `memory`      |  yes   |     no     |
| `reflection`  |   no   |    yes     |
| `goat_mr`     |  yes   |    yes     |

## How it works

1. **Objectives**: the 100 harmful behaviours of the
   [JailbreakBench](https://jailbreakbench.github.io/) `JBB-Behaviors` dataset
   (10 categories), loaded through `jbb_loader.py`.
2. **Attacker** (`attacker.py`): at each of up to 5 turns, produces a strategy
   (reasoning plus chosen techniques) and the next attack message. It picks from
   a catalogue of 7 techniques (`techniques.py`): refusal suppression, dual
   response, response priming, hypothetical scenario, persona modification,
   topic splitting and opposite intent. In memory-enabled configurations its
   prompt also contains retrieved trajectories, the category profile and
   technique-efficiency statistics.
3. **Target** (`target.py`): the model under test, queried with the running
   conversation.
4. **Judge** (`judge.py`): a JailbreakBench-style `safe`/`unsafe` classifier,
   hardened against prompt injection from the evaluated response. It
   labels every turn, and a session succeeds if any turn is `unsafe`.
5. **Learning** (`memory.py`, `reflection.py`): after each session, successful
   trajectories are indexed in memory and failed ones go through reflection.
   Retrieval uses leave-one-out exclusion, so an objective never retrieves its
   own past answer.

Each session is saved as a JSON file containing the full trajectory, the
per-turn verdicts, the first unsafe turn and the techniques used.

## Metrics

- **ASR@1** (`compute_asr.py`): one conversation per behaviour. Reports global
  and per-category attack success rate, plus a *conditional ASR* that excludes
  sessions where the attacker only refused to cooperate and never made a
  genuine attempt.
- **ASR@10** (`run_asr_k.py`, `compute_asr_k.py`): 10 independent rounds per
  behaviour (1000 conversations per configuration) with memory and reflection
  indexing frozen, so variability comes only from stochasticity. Reports ASR@k
  for k = 1..10, the average success round and per-category results.
- **McNemar test** (`mcnemar_test.py`): paired exact test on per-behaviour
  ASR@10 outcomes between two configurations.
- **Human validation** (`export_for_validators.py`): exports anonymised,
  shuffled conversations (with an overlap between validators to measure
  agreement) as CSVs, to check the automatic judge against human labels.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install groq together openai chromadb sentence-transformers datasets
cp config.example.py config.py
# Edit config.py: uncomment and fill in the API keys of the providers you use
```

Python 3.10+ is required. `config.py` is git-ignored so your keys are never
committed.

## Configuration

Everything is set in `config.py`:

- `ATTACKER_*`, `TARGET_*`, `JUDGE_*`: provider and model for each role.
  Supported providers: Groq, Together, OpenAI, Cerebras, SambaNova, OpenRouter,
  DeepInfra (`llm_client.py`).
- `ATTACKER_FALLBACKS`: ordered providers used automatically when the attacker
  hits a rate limit. Target and judge fallbacks are documented but never
  triggered automatically, so their identity stays constant across the study.
- `ACTIVE_CONFIG`: `baseline`, `memory`, `reflection` or `goat_mr`. It selects
  the module flags and the result folders.
- `MAX_TURNS`, `TEMPERATURE`, `MAX_TOKENS`: generation parameters.

## Usage

```bash
# ASR@1: one run over the remaining JBB behaviours (resumable)
python3 run.py

# ASR@10: 10 rounds per behaviour, indexation frozen (resumable)
python3 run_asr_k.py

# Metrics (pass a configuration name)
python3 compute_asr.py memory
python3 compute_asr_k.py memory
python3 mcnemar_test.py baseline goat_mr

# Seed memory/reflection from the baseline ASR@10 runs
python3 bootstrap_baseline10_memory.py

# Export conversations for human validators
python3 export_for_validators.py
```

Set `ACTIVE_CONFIG` in `config.py` before running `run.py` or `run_asr_k.py`.
A lock file prevents two runs of the same configuration from overlapping.

## Project structure

```
goat-mr/
├── config.example.py               # Configuration template
├── llm_client.py                   # Unified multi-provider LLM client
├── techniques.py                   # Catalogue of the 7 attack techniques
├── attacker.py                     # Attacker agent and prompt construction
├── target.py                       # Target model
├── judge.py                        # Per-turn safe/unsafe judge
├── memory.py                       # Trajectory memory (ChromaDB), category profiles
├── reflection.py                   # Failure analysis and lesson store
├── jbb_loader.py                   # JailbreakBench behaviour loading
├── run.py                          # ASR@1 session runner
├── run_asr_k.py                    # ASR@10 batch runner
├── compute_asr.py                  # ASR@1 metrics
├── compute_asr_k.py                # ASR@10 metrics
├── mcnemar_test.py                 # Paired significance test
├── bootstrap_memory.py             # Seed memory from baseline ASR@1 results
├── bootstrap_baseline10_memory.py  # Seed memory/reflection from baseline ASR@10 results
└── export_for_validators.py        # Human-validation export
```

Git-ignored, generated locally: `results_asr1/` and `results_asrk/` (one
sub-folder of JSON sessions per configuration), `memory_store/` (ChromaDB),
`logs/`, `validator_export/` and `config.py`.

## Responsible use

This tool elicits harmful content from language models and is intended solely
for AI-safety research and defensive evaluation of models you are authorised to
test. Do not use it against systems without permission.

## References

- Pavlova et al., "Automated Red Teaming with GOAT: the Generative Offensive
  Agent Tester", Meta, 2024. arXiv:2410.01606
- Chao et al., "JailbreakBench: An Open Robustness Benchmark for Jailbreaking
  Large Language Models", 2024. arXiv:2404.01318
