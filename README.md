# GOAT-MR — GOAT with Memory and Reflection

Extension du système de red-teaming GOAT (Meta, 2024) avec deux mécanismes
d'apprentissage cumulatif :
- **Module Mémoire** : capitalisation des trajectoires d'attaque réussies
- **Module Réflexion** : analyse et exploitation des échecs passés

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install groq together openai chromadb sentence-transformers
cp config.example.py config.py
# Renseignez vos clés API dans config.py
```

## Utilisation

```bash
python3 run.py
```

## Structure
goat_mr/
├── config.example.py   # Template de configuration
├── llm_client.py       # Client unifié multi-fournisseurs
├── techniques.py       # Catalogue des 7 techniques d'attaque
├── attacker.py         # Agent attaquant
├── target.py           # Modèle cible
├── judge.py            # Juge d'évaluation
└── run.py              # Point d'entrée

## Référence

Ahmad et al., "GOAT: Who is Sleepy Now?", Meta, 2024. arXiv:2410.01606
