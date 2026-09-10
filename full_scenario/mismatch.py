import re
import pandas as pd

def get_prediction(text):
    m = re.search(r"response:\s*([01])", str(text), re.I)
    return int(m.group(1)) if m else None

def load(path, language):
    df = pd.read_csv(path)
    df["prediction"] = df["output"].apply(get_prediction)
    df = df.dropna(subset=["prediction"])
    return df.assign(language=language, item_id=df["input_id"])[
        ["language", "item_id", "prediction", "output"]
    ]

files = [
    ("result/eval_en.csv", "English"),
    ("result/eval_hi.csv", "Hindi"),
    ("result/eval_ne.csv", "Nepali"),
]
df = pd.concat([load(path, lang) for path, lang in files])

english = df[df.language == "English"].set_index("item_id")

def mismatches_for(language):
    lang_df = df[df.language == language].set_index("item_id")
    joined = lang_df.join(english, lsuffix="_lang", rsuffix="_en", how="inner")
    return joined[joined.prediction_lang != joined.prediction_en][
        ["prediction_en", "prediction_lang", "output_en", "output_lang"]
    ]

hi_mismatches = mismatches_for("Hindi")
ne_mismatches = mismatches_for("Nepali")

hi_ids = hi_mismatches.index.tolist()
ne_ids = ne_mismatches.index.tolist()

print("Hindi mismatch item_ids:", hi_ids)
print("Nepali mismatch item_ids:", ne_ids)