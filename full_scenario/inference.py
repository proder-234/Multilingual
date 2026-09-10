import argparse
import csv

from prompt import LANG_COL, generate_prompt
from models import query_model

LANG_NAME = {"en": "English", "hi": "Hindi", "ne": "Nepali"}


def run(input_csv, output_csv, lang):
    text_col = LANG_COL[lang]
    target_language = LANG_NAME[lang]

    with open(input_csv, newline="", encoding="utf-8") as f_in:
        rows = list(csv.DictReader(f_in))

    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["input_id", "output"])
        writer.writeheader()

        for i, row in enumerate(rows, 1):
            scenario = row[text_col]
            prompt = generate_prompt(scenario, target_language)

            print(f"[{i}/{len(rows)}] generating...", end=" ", flush=True)
            full_response, score, justification = query_model(prompt)

            writer.writerow({
                "input_id": row["input_id"],
                "output": f"input: {scenario}\nresponse: {score}\njustification: {justification}",
            })
            f_out.flush()

            print(f"response={score}" if score is not None else "[!] could not parse a clean 0/1 response")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="result/ethics_translated.csv")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--lang", default="hi", choices=list(LANG_COL.keys()))
    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = f"result/eval_{args.lang}.csv"

    run(args.input_csv, args.output_csv, args.lang)
    print("Done.")

    #caffeinate -i python3 inference.py --lang en