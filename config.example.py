# Copiez ce fichier en config.py et renseignez vos clés API
# cp config.example.py config.py

# === Clés API ===
GROQ_API_KEY     = "votre_cle_groq_ici"
TOGETHER_API_KEY = "votre_cle_together_ici"
OPENAI_API_KEY   = "votre_cle_openai_ici"  # uniquement phase exp

# === Phase active : "dev" ou "exp" ===
PHASE = "dev"

MODELS = {
    "dev": {
        "attacker": ("groq",    "llama-3.3-70b-versatile"),
        "target":   ("groq",    "llama-3.1-8b-instant"),
        "judge":    ("groq",    "llama-3.1-8b-instant"),
    },
    "exp": {
        "attacker": ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        "targets": [
            ("together", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
            ("together", "mistralai/Mistral-7B-Instruct-v0.3"),
            ("openai",   "gpt-4o-mini"),
        ],
        "judge":    ("together", "meta-llama/Llama-Guard-3-8B"),
    }
}

ATTACKER_PROVIDER, ATTACKER_MODEL = MODELS[PHASE]["attacker"]
JUDGE_PROVIDER,    JUDGE_MODEL    = MODELS[PHASE]["judge"]

if PHASE == "dev":
    TARGET_PROVIDER, TARGET_MODEL = MODELS[PHASE]["target"]

MAX_TURNS   = 10
TEMPERATURE = 0.7
MAX_TOKENS  = 512
