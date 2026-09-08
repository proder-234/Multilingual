import argparse
import csv
import os

from prompt import LANG_COL, generate_prompt
from models import query_model, MODEL_REGISTRY


# Language names used in the prompt
LANG_NAME = {
    "en": "English",
    "hi": "Hindi",
}


def _load_existing(output_csv):
    """
    Load a previous run's output.

    This allows --retry_failed to keep already-good rows
    and only regenerate rows that do not have an output.
    """
    if not os.path.exists(output_csv):
        return {}

    with open(output_csv, newline="", encoding="utf-8") as f:
        return {
            row["input_id"]: row
            for row in csv.DictReader(f)
        }


def run(input_csv, output_csv, model, lang, retry_failed=False):

    # Get the correct translated-text column
    text_col = LANG_COL[lang]
    target_language = LANG_NAME[lang]

    # Load input CSV
    with open(input_csv, newline="", encoding="utf-8") as f_in:
        rows = list(csv.DictReader(f_in))

    # Load previous results if retrying failed rows
    existing = _load_existing(output_csv) if retry_failed else {}

    # Output now has only input_id + formatted output
    fieldnames = ["input_id", "output"]

    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:

        writer = csv.DictWriter(
            f_out,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for i, row in enumerate(rows, 1):

            input_id = row["input_id"]

            # ---------------------------------------------------------
            # Keep an already-successful result when using --retry_failed
            # ---------------------------------------------------------
            prior = existing.get(input_id)

            if prior and prior.get("output", "").strip():
                writer.writerow({
                    "input_id": input_id,
                    "output": prior["output"],
                })

                print(
                    f"[{i}/{len(rows)}] "
                    f"kept existing response"
                )

                f_out.flush()
                continue

            # ---------------------------------------------------------
            # Get scenario in selected language
            # ---------------------------------------------------------
            scenario = row[text_col]

            # ---------------------------------------------------------
            # Generate prompt
            # ---------------------------------------------------------
            prompt = generate_prompt(
                scenario,
                target_language
            )

            print(
                f"[{i}/{len(rows)}] generating...",
                end=" ",
                flush=True
            )

            # ---------------------------------------------------------
            # Run model
            # ---------------------------------------------------------
            full_response, score, justification = query_model(
                model,
                prompt
            )

            # ---------------------------------------------------------
            # Format everything into one CSV cell
            # ---------------------------------------------------------
            formatted_output = (
                f"input: {scenario}\n"
                f"response: {score}\n"
                f"justification: {justification}"
            )

            # ---------------------------------------------------------
            # Save result
            # ---------------------------------------------------------
            writer.writerow({
                "input_id": input_id,
                "output": formatted_output,
            })

            # Save immediately so progress isn't lost
            f_out.flush()

            # ---------------------------------------------------------
            # Print status
            # ---------------------------------------------------------
            if score is None:
                print(
                    "[!] could not parse a clean 0/1 response "
                    "after retries"
                )
            else:
                print(f"response={score}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        default="ethics_translated.csv"
    )

    parser.add_argument(
        "--output_csv",
        default=None,
        help="Defaults to eval_{lang}.csv if not specified."
    )

    parser.add_argument(
        "--model",
        default="qwen3-4b",
        choices=list(MODEL_REGISTRY.keys())
    )

    parser.add_argument(
        "--lang",
        default="hi",
        choices=list(LANG_COL.keys())
    )

    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help=(
            "Reuse an existing output CSV, keep rows that already "
            "have an output, and regenerate only missing rows."
        ),
    )

    args = parser.parse_args()

    if args.output_csv is None:
        args.output_csv = f"eval_{args.lang}.csv"

    print(
        f"Loading model '{args.model}' "
        "(this can take a while the first time)..."
    )

    run(
        args.input_csv,
        args.output_csv,
        args.model,
        args.lang,
        retry_failed=args.retry_failed,
    )

    print("Done.")