# === Clés API ===
GROQ_API_KEY       = "votre_cle_groq"
TOGETHER_API_KEY   = "votre_cle_together"
OPENAI_API_KEY     = "votre_cle_openai"        # phase exp uniquement
CEREBRAS_API_KEY   = "votre_cle_cerebras"      # optionnel selon accès
SAMBANOVA_API_KEY  = "votre_cle_sambanova"
OPENROUTER_API_KEY = "votre_cle_openrouter"

# === Phase active : "dev" ou "exp" ===
PHASE = "dev"

# Modèles disponibles par fournisseur (pour référence)
# - groq        : llama-3.3-70b-versatile, llama-3.1-8b-instant
# - sambanova   : Meta-Llama-3.3-70B-Instruct
# - together    : meta-llama/Llama-3.3-70B-Instruct-Turbo
# - openrouter  : meta-llama/llama-3.3-70b-instruct
# - cerebras    : llama-3.3-70b (accès restreint sur le free tier)

MODELS = {
    "dev": {
        # Attaquant principal sur SambaNova (quota gratuit généreux, Llama-3.3-70B)
        "attacker": ("sambanova", "Meta-Llama-3.3-70B-Instruct"),
        # Cible et juge sur Groq (modèle 8B, quota TPD suffisant)
        "target":   ("groq", "llama-3.1-8b-instant"),
        "judge":    ("groq", "llama-3.1-8b-instant"),
    },
    "exp": {
        "attacker": ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        "targets": [
            ("together", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
            ("together", "mistralai/Mistral-7B-Instruct-v0.3"),
            ("openai",   "gpt-4o-mini"),
        ],
        "judge":    ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    },
}

ATTACKER_PROVIDER, ATTACKER_MODEL = MODELS[PHASE]["attacker"]
JUDGE_PROVIDER,    JUDGE_MODEL    = MODELS[PHASE]["judge"]

if PHASE == "dev":
    TARGET_PROVIDER, TARGET_MODEL = MODELS[PHASE]["target"]

# === Fallback automatique en cas de rate limit sur le fournisseur primaire ===
# Liste ordonnée de fournisseurs à essayer si l'attaquant atteint sa limite.
# Tous doivent héberger un Llama-3.3-70B équivalent.
ATTACKER_FALLBACKS = [
    ("sambanova",  "Meta-Llama-3.3-70B-Instruct"),
    ("groq",       "llama-3.3-70b-versatile"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    ("together",   "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
]

# === Paramètres de génération (conformes au papier GOAT) ===
MAX_TURNS   = 5
TEMPERATURE = 0.7
MAX_TOKENS  = 512