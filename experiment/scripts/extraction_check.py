#!/usr/bin/env python3
"""Cross-check the annotator's answer against a regex, and surface the fights.

    python -m scripts.extraction_check              # summary + disagreements
    python -m scripts.extraction_check --full       # with the response text
    python -m scripts.extraction_check --strict     # exit 1 if any disagree

WHY THIS EXISTS. Only 21 of the pilot's 40 responses ended with a bare answer
word, and non-compliance is not random: in `default_keep` the conditional KEEP
rate is 0.231 among compliant responses and 0.667 among non-compliant ones.
Extraction is therefore not tidying — parse the hard cases badly and the
headline moves.

Neither reader dominates the other, which is the point. On the pilot the
annotator was right once, the regex twice, and both were wrong once. So this
does not pick a winner: it reports the method mix, counts the disagreements,
and prints them for a human. That is PLAN.md §8.4's hand-coding step aimed at
the cases that actually need it instead of a random 60.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from src import contracts as c
from src import export

METHODS = ("final_line", "terminal", "last_caps", "last_mention", "none")


def compare(rows: list[dict]) -> tuple[list[dict], collections.Counter, collections.Counter]:
    """(disagreements, n by method, agreements by method)."""
    disagreements, totals, agrees = [], collections.Counter(), collections.Counter()
    for row in rows:
        answer, method = c.extract_answer(row.get("response_text") or "")
        totals[method] += 1
        if answer == row.get("answer"):
            agrees[method] += 1
        else:
            disagreements.append({**row, "regex_answer": answer, "method": method})
    return disagreements, totals, agrees


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", default=str(c.RAW_PATH))
    parser.add_argument("--annotated", default=str(c.ANNOTATED_PATH))
    parser.add_argument("--full", action="store_true", help="print the response text")
    parser.add_argument("--strict", action="store_true", help="exit 1 on any disagreement")
    args = parser.parse_args(argv)

    rows = [r for r in export.build_rows(Path(args.raw), Path(args.annotated))
            if r["mode"] == "cot" and r.get("answer")]
    if not rows:
        print("no annotated cot records — run the runner and annotate first")
        return 0

    disagreements, totals, agrees = compare(rows)

    print(f"{len(rows)} annotated cot records\n")
    print(f"{'method':14s} {'n':>4s} {'agree':>7s}   what it means")
    print("-" * 78)
    blurb = {
        "final_line": "the last line is a bare answer word — no inference",
        "terminal": "a closing declaration: 'the answer is KEEP'",
        "last_caps": "last ALL-CAPS mention; models shout conclusions",
        "last_mention": "last mention outside parentheses — weakest",
        "none": "no answer word found at all",
    }
    for method in METHODS:
        if totals[method]:
            print(f"{method:14s} {totals[method]:>4d} {agrees[method]:>4d}/{totals[method]:<3d}"
                  f"  {blurb[method]}")
    compliant = totals["final_line"]
    print(f"\nformat compliance: {compliant}/{len(rows)} "
          f"({compliant / len(rows):.0%}) ended with a bare answer word")
    print(f"agreement: {sum(agrees.values())}/{len(rows)} "
          f"({sum(agrees.values()) / len(rows):.0%})")

    # The number that matters for the headline: does non-compliance track the
    # answer? If it does, mis-parsing the hard cases biases the flip rate.
    print("\nconditional KEEP rate by scenario, split on compliance")
    print("(a gap here means extraction errors do not cancel out)")
    for scenario in sorted({r["scenario"] for r in rows}):
        for label, want in (("compliant", True), ("non-compliant", False)):
            group = [r for r in rows if r["scenario"] == scenario
                     and (c.extract_answer(r["response_text"] or "")[1] == "final_line") == want]
            keep = sum(1 for r in group if r["answer"] == "KEEP")
            switch = sum(1 for r in group if r["answer"] == "SWITCH")
            rate = f"{keep / (keep + switch):.3f}" if keep + switch else "n/a"
            print(f"   {scenario:16s} {label:14s} n={len(group):>3d}  keep={rate}")

    if not disagreements:
        print("\nOK: the regex and the annotator agree on every record.")
        return 0

    print(f"\n!! {len(disagreements)} disagreement(s) — hand-adjudicate these:\n")
    for row in disagreements:
        lines = [l.strip() for l in (row["response_text"] or "").strip().splitlines() if l.strip()]
        tail = lines[-1] if lines else "(empty)"
        print(f"  {row['run_id'][:12]}  {row['model'].split('/')[-1]:22s} {row['scenario']:15s}")
        print(f"     regex={str(row['regex_answer']):7s}({row['method']:12s})  "
              f"annotator={row['answer']:7s}  depth={row['depth']}")
        print(f"     ends: {tail[:96]!r}")
        if args.full:
            print("     ---")
            for line in (row["response_text"] or "").strip().splitlines():
                print(f"     {line}")
        print()
    print("Neither reader dominates: on the pilot the annotator was right once,")
    print("the regex twice, and both were wrong once. Read the text, not the labels.")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(_main())
