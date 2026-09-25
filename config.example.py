# === Clés API ===
# Add your own API keys below (never commit real keys).
# Only the providers you actually use are required.
# GROQ_API_KEY       = ""   # add your Groq API key here
# TOGETHER_API_KEY   = ""   # add your Together AI API key here
# OPENROUTER_API_KEY = ""   # add your OpenRouter API key here
# SAMBANOVA_API_KEY  = ""   # add your SambaNova API key here
# CEREBRAS_API_KEY   = ""   # add your Cerebras API key here
# OPENAI_API_KEY     = ""   # add your OpenAI API key here
# DEEPINFRA_API_KEY  = ""   # add your DeepInfra API key here (default provider)


ATTACKER_PROVIDER, ATTACKER_MODEL = ("deepinfra", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
TARGET_PROVIDER,   TARGET_MODEL   = ("deepinfra", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
JUDGE_PROVIDER,    JUDGE_MODEL    = ("deepinfra", "meta-llama/Llama-3.3-70B-Instruct-Turbo")

# === Fallback automatique en cas de rate limit sur le fournisseur primaire ===
# Liste ordonnée de fournisseurs à essayer si l'attaquant atteint sa limite.
# Tous doivent héberger un Llama-3.3-70B équivalent.
ATTACKER_FALLBACKS = [
    ("deepinfra",  "Meta-Llama-3.3-70B-Instruct"),
    ("sambanova",  "Meta-Llama-3.3-70B-Instruct"),
    ("together",   "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
]


# === Target & judge fallback: defined for manual/documented use only ===
# NOT auto-triggered by default (enable_fallback=False in target.py and
# judge.py). Switching mid-study must be a deliberate, dated decision
# reported in chapter 4, since target/judge identity must stay constant
# across all four compared configurations (baseline, memory, reflection,
# goat_mr) for the ablation to remain valid.
TARGET_FALLBACKS = [
    ("openrouter", "meta-llama/llama-3.1-8b-instruct"),
    ("together",   "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
]

JUDGE_FALLBACKS = [
    ("openrouter", "meta-llama/llama-3.1-8b-instruct"),
    ("together",   "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
]


# === Active Configuration (determines where results are stored) ===
# Possible values : "baseline" | "memory" | "reflection" | "goat_mr"
ACTIVE_CONFIG = "goat_mr"  # change this to switch between configurations

ASR1_ROOT = "results_asr1"
ASRK_ROOT = "results_asrk"

RESULTS_DIRS = {
    "baseline":   f"{ASR1_ROOT}/results_baseline",
    "memory":     f"{ASR1_ROOT}/results_memory",
    "reflection": f"{ASR1_ROOT}/results_reflection",
    "goat_mr":    f"{ASR1_ROOT}/results_goat_mr",
}

ASRK_RESULTS_DIRS = {
    "baseline":   f"{ASRK_ROOT}/results_baseline_asrk",
    "memory":     f"{ASRK_ROOT}/results_memory_asrk",
    "reflection": f"{ASRK_ROOT}/results_reflection_asrk",
    "goat_mr":    f"{ASRK_ROOT}/results_goat_mr_asrk",
}

RESULTS_DIR = RESULTS_DIRS[ACTIVE_CONFIG]

# Module activation flags
USE_MEMORY     = ACTIVE_CONFIG in ("memory", "goat_mr")
USE_REFLECTION = ACTIVE_CONFIG in ("reflection", "goat_mr")

MAX_TURNS   = 5
TEMPERATURE = 0.7
MAX_TOKENS  = 512