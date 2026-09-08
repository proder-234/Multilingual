"""
metrics.py — multilingual reasoning-consistency evaluation

Reads eval_{lang}.csv files (input_id, output), reshapes into a
language/item_id/prediction table, and compares every language against
BASE_LANGUAGE ("english") on the same items. There's no ground-truth
label — the base language's prediction per item stands in for one, so
"accuracy"/"F1" below mean "agreement with English," not correctness.
A language scoring lower isn't wrong, it's diverging more from English,
which is the training/cultural gap this study is measuring.

  (1) Classification performance (vs. base language)
        precision / recall / F1 / confusion matrix / accuracy, pooled
        and per-language
  (2) Multilingual robustness
        translation invariance, per-language gap vs. base, pairwise
        agreement, F1 variance across languages
  (3) Fleiss' kappa across languages (supplementary agreement measure,
      not a classical reliability statistic)
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

BASE_LANGUAGE = "english"

_RESPONSE_RE = re.compile(r"response:\s*(\S+)", re.IGNORECASE)


def _parse_prediction(output_text):
    m = _RESPONSE_RE.search(str(output_text))
    if not m:
        return None
    val = m.group(1).strip()
    return int(val) if val in ("0", "1") else None


def load_eval_csv(path, language):
    df = pd.read_csv(path)
    df["prediction"] = df["output"].apply(_parse_prediction)
    n_bad = df["prediction"].isna().sum()
    if n_bad:
        print(f"[{path}] dropping {n_bad} unparseable predictions of {len(df)}")
    df = df.dropna(subset=["prediction"]).copy()
    df["prediction"] = df["prediction"].astype(int)
    df["language"] = language
    return df.rename(columns={"input_id": "item_id"})[["language", "item_id", "prediction"]]


def build_results(eval_specs):
    """eval_specs: [{"path": "result/eval_en.csv", "language": "English"},  {"path": "result/eval_ne.csv", "language": "Nepali"},]"""
    return pd.concat([load_eval_csv(s["path"], s["language"]) for s in eval_specs], ignore_index=True)


def with_base_label(df, base_language=BASE_LANGUAGE):
    """Attach label = base language's prediction for that item_id; drop
    the base language's own rows and items missing a base prediction."""
    base = df[df["language"].str.lower() == base_language.lower()]
    if base.empty:
        raise ValueError(f"No rows for base language '{base_language}'")
    ref = dict(zip(base["item_id"], base["prediction"]))

    out = df[df["language"].str.lower() != base_language.lower()].copy()
    out["label"] = out["item_id"].map(ref)
    n_missing = out["label"].isna().sum()
    if n_missing:
        print(f"Dropping {n_missing} rows with no '{base_language}' prediction for that item")
    return out.dropna(subset=["label"]).astype({"label": int})


# ---------------------------------------------------------------------------
# (1) Classification performance vs. base language
# ---------------------------------------------------------------------------

def overall_metrics(df, base_language=BASE_LANGUAGE):
    labeled = with_base_label(df, base_language)
    p, r, f1, _ = precision_recall_fscore_support(
        labeled["label"], labeled["prediction"], average="binary", zero_division=0
    )
    cm = confusion_matrix(labeled["label"], labeled["prediction"]).tolist()
    return {"precision": p, "recall": r, "f1": f1, "confusion_matrix": cm}


def aggregate_accuracy(df, base_language=BASE_LANGUAGE):
    labeled = with_base_label(df, base_language)
    return accuracy_score(labeled["label"], labeled["prediction"])


def per_language_metrics(df, base_language=BASE_LANGUAGE):
    labeled = with_base_label(df, base_language)
    rows = []
    for language, g in labeled.groupby("language"):
        p, r, f1, _ = precision_recall_fscore_support(
            g["label"], g["prediction"], average="binary", zero_division=0
        )
        rows.append({
            "language": language,
            "n": len(g),
            "accuracy": accuracy_score(g["label"], g["prediction"]),
            "precision": p, "recall": r, "f1": f1,
        })
    return pd.DataFrame(rows).sort_values("f1")


# ---------------------------------------------------------------------------
# (2) Multilingual robustness
# ---------------------------------------------------------------------------

def translation_invariance(df):
    """Fraction of items where ALL languages predict the same thing."""
    pivot = df.pivot_table(index="item_id", columns="language", values="prediction").dropna()
    return (pivot.nunique(axis=1) == 1).mean() if len(pivot) else float("nan")


def language_gap(per_lang_df):
    """Each language's F1/accuracy gap from the base language (=1.0 by definition)."""
    out = per_lang_df.copy()
    out["f1_gap"] = 1.0 - out["f1"]
    out["accuracy_gap"] = 1.0 - out["accuracy"]
    return out[["language", "n", "f1", "f1_gap", "accuracy", "accuracy_gap"]].sort_values(
        "f1_gap", ascending=False
    )


def pairwise_language_agreement(df):
    pivot = df.pivot_table(index="item_id", columns="language", values="prediction")
    langs = pivot.columns.tolist()
    rows = []
    for i in range(len(langs)):
        for j in range(i + 1, len(langs)):
            sub = pivot[[langs[i], langs[j]]].dropna()
            agreement = (sub[langs[i]] == sub[langs[j]]).mean() if len(sub) else float("nan")
            rows.append({"language_1": langs[i], "language_2": langs[j], "n": len(sub), "agreement": agreement})
    return pd.DataFrame(rows).sort_values("agreement")


def per_language_variance(per_lang_df):
    return per_lang_df["f1"].agg(["mean", "std", "min", "max"]).rename({
        "mean": "f1_mean", "std": "f1_std", "min": "f1_min", "max": "f1_max"
    })


# ---------------------------------------------------------------------------
# (3) Fleiss' kappa (supplementary, symmetric — no base language)
# ---------------------------------------------------------------------------

def fleiss_kappa(table):
    n_items, n_cat = table.shape
    n_raters = table.sum(axis=1)[0]
    p_j = table.sum(axis=0) / (n_items * n_raters)
    P_i = ((table ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()
    return 1.0 if P_e == 1 else (P_bar - P_e) / (1 - P_e)


def fleiss_kappa_overall(df, categories=(0, 1)):
    pivot = df.pivot_table(index="item_id", columns="language", values="prediction").dropna()
    table = np.zeros((len(pivot), len(categories)))
    for i, (_, row) in enumerate(pivot.iterrows()):
        for c_idx, c in enumerate(categories):
            table[i, c_idx] = (row == c).sum()
    return fleiss_kappa(table) if len(pivot) else float("nan")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def run_full_report(df, base_language=BASE_LANGUAGE):
    per_lang = per_language_metrics(df, base_language)
    return {
        "classification_performance": {
            "overall_vs_base": overall_metrics(df, base_language),
            "aggregate_accuracy_vs_base": aggregate_accuracy(df, base_language),
            "per_language_vs_base": per_lang,
        },
        "multilingual_robustness": {
            "translation_invariance": translation_invariance(df),
            "language_gap": language_gap(per_lang),
            "pairwise_agreement": pairwise_language_agreement(df),
            "per_language_variance": per_language_variance(per_lang),
        },
        "agreement": {"fleiss_kappa_cross_language": fleiss_kappa_overall(df)},
    }


if __name__ == "__main__":
    EVAL_SPECS = [
        {"path": "result/eval_en.csv", "language": "English"},
        {"path": "result/eval_hi.csv", "language": "Hindi"},
        {"path": "result/eval_ne.csv", "language": "Nepali"},
    ]
    df = build_results(EVAL_SPECS)
    report = run_full_report(df)

    print("\n=== Overall vs. English ===")
    print(report["classification_performance"]["overall_vs_base"])
    print("\n=== Aggregate accuracy vs. English ===")
    print(report["classification_performance"]["aggregate_accuracy_vs_base"])
    print("\n=== Per-language metrics vs. English ===")
    print(report["classification_performance"]["per_language_vs_base"])

    print("\n=== Translation invariance ===")
    print(report["multilingual_robustness"]["translation_invariance"])
    print("\n=== Language gap vs. English ===")
    print(report["multilingual_robustness"]["language_gap"])
    print("\n=== Pairwise language agreement ===")
    print(report["multilingual_robustness"]["pairwise_agreement"])
    print("\n=== Per-language F1 variance ===")
    print(report["multilingual_robustness"]["per_language_variance"])

    print("\n=== Fleiss' kappa across languages ===")
    print(report["agreement"]["fleiss_kappa_cross_language"])