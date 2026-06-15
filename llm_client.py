import time
from groq import Groq
from together import Together
from openai import OpenAI
import config


def _make_client(provider: str):
    """Crée et retourne un client API pour le fournisseur donné."""
    if provider == "groq":
        return Groq(api_key=config.GROQ_API_KEY)

    if provider == "together":
        return Together(api_key=config.TOGETHER_API_KEY)

    if provider == "openai":
        return OpenAI(api_key=config.OPENAI_API_KEY)

    if provider == "cerebras":
        # API Cerebras Cloud compatible OpenAI
        return OpenAI(
            api_key=config.CEREBRAS_API_KEY,
            base_url="https://api.cerebras.ai/v1",
        )

    if provider == "sambanova":
        # API SambaNova Cloud compatible OpenAI
        return OpenAI(
            api_key=config.SAMBANOVA_API_KEY,
            base_url="https://api.sambanova.ai/v1",
        )

    if provider == "openrouter":
        # OpenRouter, compatible OpenAI
        return OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )

    raise ValueError(f"Fournisseur inconnu : {provider}")


def _is_rate_limit_error(exc: Exception) -> bool:
    """Détecte si l'exception est due à une limite de quota (HTTP 429)."""
    msg = str(exc).lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "quota" in msg
        or "too many requests" in msg
        or "insufficient" in msg
    )


def _single_call(provider: str, model: str, messages: list,
                  temperature: float, max_tokens: int) -> str:
    """Appel unique à un fournisseur sans retry interne."""
    client = _make_client(provider)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def get_completion(provider: str, model: str,
                    messages: list, temperature: float = 0.7,
                    max_tokens: int = 512,
                    retries: int = 3,
                    enable_fallback: bool = True) -> str:
    """
    Appel unifié avec :
    - retry exponentiel sur erreurs réseau transitoires
    - fallback automatique sur fournisseurs alternatifs si rate limit atteint
      (uniquement pour l'attaquant, contrôlé par enable_fallback)

    `enable_fallback` est activé par défaut, ce qui permet à l'attaquant
    de basculer entre fournisseurs lorsque l'un atteint sa limite quotidienne.
    Pour la cible ou le juge, appelez avec enable_fallback=False afin de
    conserver la fidélité du modèle évalué.
    """
    # Construit la liste des (provider, model) à essayer
    candidates = [(provider, model)]
    if enable_fallback:
        for fb_provider, fb_model in config.ATTACKER_FALLBACKS:
            if (fb_provider, fb_model) not in candidates:
                candidates.append((fb_provider, fb_model))

    last_error = None

    for prov, mod in candidates:
        for attempt in range(1, retries + 1):
            try:
                return _single_call(prov, mod, messages,
                                     temperature, max_tokens)

            except Exception as e:
                last_error = e
                is_rate_limit = _is_rate_limit_error(e)

                if is_rate_limit and enable_fallback:
                    # On abandonne ce fournisseur immédiatement
                    print(f"[RATE LIMIT] {prov}/{mod} épuisé, "
                           f"bascule vers le fournisseur suivant.")
                    break

                if attempt < retries:
                    wait = 2 ** attempt
                    print(f"[ERREUR réseau] {prov}/{mod} tentative "
                           f"{attempt}/{retries} : {str(e)[:120]}. "
                           f"Nouvelle tentative dans {wait}s...")
                    time.sleep(wait)
                else:
                    # Toutes les tentatives échouées pour ce fournisseur
                    if enable_fallback:
                        print(f"[ECHEC] {prov}/{mod} après {retries} "
                               f"tentatives, bascule vers le suivant.")
                    break

    raise RuntimeError(
        f"Echec sur tous les fournisseurs. Dernière erreur : {last_error}"
    )