import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL = "facebook/nllb-200-1.3B"

print("Loading model (This can take a minute the first time)")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU
else:
    device = "cpu"
print(f"Using device: {device}")
model = model.to(device)


def strip_forum_tags(text):
    """Remove AITA/WIBTA/etc. tags from the start of scenario text before translation.
    These are Reddit forum labels, not real words, NLLB just passes them through
    untranslated, which leaves ugly English fragments stuck in the hindi output."""
    return re.sub(r"^(AITA|WIBTA|Aitah?)\b[:\-\|]?\s*", "", text, flags=re.IGNORECASE).strip()


def translate(texts, src_lang, tgt_lang, batch_size=16, max_length=200):
    """Translate a list of texts in batches using NLLB directly."""
    tokenizer.src_lang = src_lang
    output = []

    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        print(f"  batch {batch_num}/{total_batches}...")
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        # NLLB needs to know the TARGET language id to force generation into it
        tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)

        generated = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=max_length,
        )

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        output.extend(decoded)

    return output


if __name__ == "__main__":
    df = pd.read_csv("ethics_pilot.csv")
    df = df.rename(columns={"input": "en_text"})

    print("Stripping forum tags (AITA/WIBTA)...")
    df["en_text"] = df["en_text"].apply(strip_forum_tags)

    print("Translating to Hindi...")
    df["hi_text"] = translate(
        df["en_text"].tolist(),
        "eng_Latn",
        "hin_Deva",        
    )

    df.to_csv("ethics_translated.csv", index=False)
    print(f"Saved {len(df)} translations")

#MAD-400