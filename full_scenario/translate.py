import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "facebook/nllb-200-1.3B"

# Load model
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpusource venv/bin/activate"
print(f"Using device: {device}")

model = model.to(device)
model.eval()
print("Model ready!")

# Remove Reddit tags
def strip_forum_tags(text):
    return re.sub(
        r"^(AITA|WIBTA|Aitah?)\b[:\-\|]?\s*",
        "",
        str(text),
        flags=re.IGNORECASE
    ).strip()

# Split long scenarios
def split_text(text, max_tokens=400):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sentence in sentences:
        test = (current + " " + sentence).strip()
        if len(tokenizer.encode(test)) <= max_tokens:
            current = test
        else:
            if current:
                chunks.append(current)
            # Handle very long single sentences
            words = sentence.split()
            current = ""
            for word in words:
                test = (current + " " + word).strip()
                if len(tokenizer.encode(test)) <= max_tokens:
                    current = test
                else:
                    if current:
                        chunks.append(current)
                    current = word
    if current:
        chunks.append(current)
    return chunks


# Translate a batch
def translate_batch(batch, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=False
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=250,
            num_beams=1
        )
    return tokenizer.batch_decode(
        output,
        skip_special_tokens=True
    )

# Translate all scenarios to MULTIPLE target languages in one pass, so
# each row is only split into chunks once, no matter how many target
# languages it's translated into.
def translate_multi(texts, src_lang, tgt_langs, batch_size=8):
    """
    tgt_langs: dict of {output_key: nllb_lang_code}, e.g.
        {"hi_text": "hin_Deva", "ne_text": "npi_Deva"}

    Returns: dict of {output_key: [translated_text, ...]} aligned to `texts`.
    """
    results = {key: [] for key in tgt_langs}

    for index, text in enumerate(texts):
        print(f"Translating {index + 1}/{len(texts)}...", flush=True)
        text = str(text).strip()
        chunks = split_text(text, max_tokens=400)  # done once per row

        for key, tgt_lang in tgt_langs.items():
            translated_chunks = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                translated_chunks.extend(
                    translate_batch(batch, src_lang, tgt_lang)
                )
            results[key].append(" ".join(translated_chunks))

    return results


# Main
if __name__ == "__main__":
    print("Reading input CSV...")
    df = pd.read_csv("result/ethics_pilot.csv")
    if "input" in df.columns:
        df = df.rename(columns={"input": "en_text"})
    df["en_text"] = df["en_text"].apply(strip_forum_tags)
    print(f"Loaded {len(df)} rows.")

    print("Translating...")
    translations = translate_multi(
        df["en_text"].tolist(),
        src_lang="eng_Latn",
        tgt_langs={"hi_text": "hin_Deva", "ne_text": "npi_Deva"},
        batch_size=8
    )
    df["hi_text"] = translations["hi_text"]
    df["ne_text"] = translations["ne_text"]

    df.to_csv("result/ethics_translated.csv", index=False)
    print("Saved to result/ethics_translated.csv")