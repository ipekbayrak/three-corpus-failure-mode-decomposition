#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

PLACEHOLDER_RULES = {
    "author name": "Author Name",
    "corresponding email": "corresponding.author@example.com",
    "affiliation": "Affiliation Placeholder",
    "affiliation city": "city={City}",
    "affiliation country": "country={Country}",
    "funding statement": "Funding information will be added at submission.",
    "data-availability statement": (
        "A formal data-availability statement will be added at submission."
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail explicitly when the journal manuscript still contains submission "
            "placeholders."
        )
    )
    parser.add_argument(
        "--manuscript",
        default="paper/manuscript.tex",
        help="Path to the journal manuscript TeX file.",
    )
    args = parser.parse_args()

    manuscript_path = Path(args.manuscript)
    text = manuscript_path.read_text(encoding="utf-8")
    missing_fields = [
        field_name
        for field_name, placeholder in PLACEHOLDER_RULES.items()
        if placeholder in text
    ]

    if missing_fields:
        print(
            "Submission metadata is incomplete. Replace these placeholders before calling "
            "the journal manuscript submission-ready:"
        )
        for field_name in missing_fields:
            print(f"- {field_name}")
        return 1

    print("Submission metadata placeholders are cleared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
