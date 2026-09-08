import re
from mlx_lm import load, generate

MODEL_ID = "mlx-community/Qwen3-8B-4bit"

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)

_model, _tokenizer = load(MODEL_ID)


def _parse(full_response):
    """Extract a strict 0/1 response + justification, or None if malformed."""
    score, justification = None, ""

    m = re.search(r"response\s*:\s*(.+)", full_response, re.IGNORECASE)
    if m:
        raw = m.group(1).splitlines()[0]
        cleaned = re.sub(r"[\*\[\]`\s\.]", "", raw)
        if cleaned in ("0", "1"):
            score = cleaned

    m = re.search(r"justification\s*:\s*(.*)", full_response, re.IGNORECASE | re.DOTALL)
    if m:
        justification = m.group(1).strip()

    return score, justification


def query_model(prompt, max_new_tokens=220):
    messages = [{"role": "user", "content": prompt}]
    text = _tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False
    )

    raw = generate(_model, _tokenizer, prompt=text, max_tokens=max_new_tokens, verbose=False).strip()
    full_response = _THINK_BLOCK.sub("", raw).strip()
    score, justification = _parse(full_response)

    return full_response, score, justification