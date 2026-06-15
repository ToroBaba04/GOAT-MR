from llm_client import get_completion
import config

def query_target(conversation_history: list) -> str:
    return get_completion(
        provider=config.TARGET_PROVIDER,
        model=config.TARGET_MODEL,
        messages=conversation_history,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        enable_fallback=False,   # cible figée pour l'évaluation
    )