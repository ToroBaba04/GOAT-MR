from llm_client import _single_call

providers_to_test = [
    ("cerebras",   "llama-3.3-70b"),
    ("sambanova",  "Meta-Llama-3.3-70B-Instruct"),
    ("openrouter", "meta-llama/llama-3.3-70b-instruct"),
    ("together",   "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
]

messages = [{"role": "user", "content": "Reply with exactly: OK"}]

for prov, model in providers_to_test:
    try:
        out = _single_call(prov, model, messages,
                            temperature=0.0, max_tokens=20)
        print(f"[OK]   {prov:12s} → {out.strip()[:60]}")
    except Exception as e:
        print(f"[FAIL] {prov:12s} → {str(e)[:120]}")