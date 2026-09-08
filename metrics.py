import re
import pandas as pd
from sklearn.metrics import accuracy_score

def get_prediction(text):
    m = re.search(r"response:\s*([01])", str(text), re.I)
    return int(m.group(1)) if m else None


def load(path, language):
    df = pd.read_csv(path)
    df["prediction"] = df["output"].apply(get_prediction)
    df = df.dropna(subset=["prediction"])

    return df.assign(
        language=language,
        item_id=df["input_id"]
    )[["language", "item_id", "prediction"]]

files = [
    ("result/eval_en.csv", "English"),
    ("result/eval_hi.csv", "Hindi"),
    ("result/eval_ne.csv", "Nepali"),
]

df = pd.concat([load(path, lang) for path, lang in files])

english = df[df.language == "English"]
reference = dict(zip(english.item_id, english.prediction))

df = df[df.language != "English"].copy()
df["label"] = df.item_id.map(reference)

result = (
    df.dropna(subset=["label"])
      .groupby("language")
      .apply(lambda x: accuracy_score(x.label, x.prediction))
)
print(result)