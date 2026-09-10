import pandas as pd

def load_ethics_commonsense(n=None, seed=42):
    url = "https://huggingface.co/datasets/hendrycks/ethics/resolve/refs%2Fconvert%2Fparquet/commonsense/test/0000.parquet"
    df = pd.read_parquet(url)

    df = df[["input"]].dropna()

    if n is not None:
        df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df.insert(0, "input_id", range(len(df)))
    return df

if __name__ == "__main__":
    df = load_ethics_commonsense(n=None)
    df.to_csv("result/ethics_pilot.csv", index=False)

    print(f"Saved {len(df)} scenarios")