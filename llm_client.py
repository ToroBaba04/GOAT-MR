import time
from groq import Groq
from together import Together
import config

def get_completion(provider: str, model: str,
                   messages: list, temperature: float = 0.7,
                   max_tokens: int = 512,
                   retries: int = 3) -> str:
    """
    Appel unifié avec retry automatique en cas d'erreur réseau.
    """
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            if provider == "groq":
                client = Groq(api_key=config.GROQ_API_KEY)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

            elif provider == "together":
                client = Together(api_key=config.TOGETHER_API_KEY)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

            elif provider == "openai":
                from openai import OpenAI
                client = OpenAI(api_key=config.OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

            else:
                raise ValueError(f"Fournisseur inconnu : {provider}")

        except Exception as e:
            last_error = e
            wait = 2 ** attempt  # backoff exponentiel : 2s, 4s, 8s
            print(f"[ERREUR réseau] Tentative {attempt}/{retries} "
                  f"échouée : {e}. Nouvelle tentative dans {wait}s...")
            time.sleep(wait)

    raise RuntimeError(
        f"Échec après {retries} tentatives. "
        f"Dernière erreur : {last_error}"
    )