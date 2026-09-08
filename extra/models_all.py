import re
from mlx_lm import load, generate

# MLX-format, 4-bit quantized checkpoint --------------
MODEL_REGISTRY = {
    "qwen3-4b": "mlx-community/Qwen3-4B-4bit",
    "qwen3-8b": "mlx-community/Qwen3-8B-4bit",
    "gemma4-12b": "mlx-community/gemma-4-12B-it-4bit",
    "aya-expanse-8b": "mlx-community/aya-expanse-8b-4bit",
    "aya-expanse-32b": "mlx-community/aya-expanse-32b-8bit",
}

MODEL_LANGUAGES = {
    "qwen3": "119 languages and dialects, incl. Hindi",
    "gemma4": "140+ languages pretrained; Hindi at broad (not top-tier tuned) level",
    "aya-expanse": "23 languages incl. Hindi, specifically tuned for multilingual quality",
}

_CACHE = {}

# Some models (Qwen3 in particular) emit a <think>...</think> reasoning
# block by default before the real answer. We try to switch this off via
# the chat template (enable_thinking=False); as a safety net we also strip
# any such block before parsing, in case it slips through anyway.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _load(model_key):
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_key}'. Options: {list(MODEL_REGISTRY)}")
    if model_key in _CACHE:
        return _CACHE[model_key]
    repo_id = MODEL_REGISTRY[model_key]
    print(f"Loading {model_key} ({repo_id}) via MLX -- this only happens once per run...")
    model, tokenizer = load(repo_id)
    _CACHE[model_key] = (model, tokenizer)
    return model, tokenizer


def _apply_chat_template(tokenizer, messages):
    """Apply the chat template, disabling extended 'thinking' mode when the
    tokenizer supports it (Qwen3 and similar). Falls back gracefully for
    tokenizers that don't accept that kwarg (Gemma, Aya, etc.)."""
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def _clean(full_response):
    """Strip any <think> reasoning block the model produced anyway."""
    return _THINK_BLOCK.sub("", full_response).strip()


def _parse(full_response):
    """
    Strictly extract a single 0/1 response and its justification.

    Returns (score, justification). `score` is None -- not coerced to "0"
    or "1" -- whenever the response line is missing, empty, or doesn't
    reduce to EXACTLY '0' or '1' once markdown/bracket decoration is
    stripped. This deliberately rejects malformed outputs like "0/1",
    "0 or 1", "01", "**1**." etc. instead of silently guessing.
    """
    score, justification = None, ""

    m = re.search(r"response\s*:\s*(.+)", full_response, re.IGNORECASE)
    if m:
        raw = m.group(1).splitlines()[0]
        cleaned = re.sub(r"[\*\[\]`\s\.]", "", raw)  # strip **, [], backticks, spaces, periods
        if cleaned in ("0", "1"):
            score = cleaned

    m = re.search(r"justification\s*:\s*(.*)", full_response, re.IGNORECASE | re.DOTALL)
    if m:
        justification = m.group(1).strip()

    return score, justification


def query_model(model_key, prompt, max_new_tokens=220, temperature=0.2, top_p=0.9,
                 max_retries=2):
    """
    Run any model in MODEL_REGISTRY via MLX. Loads weights once (cached in
    _CACHE) and reuses them for every subsequent call in the same process.

    If the output doesn't contain a clean, parseable 0/1 response (e.g. a
    leftover reasoning block or truncated generation ate the token budget),
    retries up to `max_retries` times with a larger budget each time, rather
    than silently returning a wrong or blank answer.
    """
    model, tokenizer = _load(model_key)
    messages = [{"role": "user", "content": prompt}]
    text = _apply_chat_template(tokenizer, messages)

    budget = max_new_tokens
    full_response, score, justification = "", None, ""
    for _ in range(max_retries + 1):
        raw_response = generate(
            model, tokenizer, prompt=text, max_tokens=budget, verbose=False,
        ).strip()
        full_response = _clean(raw_response)
        score, justification = _parse(full_response)
        if score is not None:
            return full_response, score, justification
        budget = int(budget * 1.5)  

    return full_response, score, justification