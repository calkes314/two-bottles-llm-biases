#!/usr/bin/env python3
"""Print the contamination check as a model x scenario table.

    python -m scripts.contamination_report

PLAN.md §2 asks 15 calls (5 models x 3 scenarios) and then says "Read the 15 by
hand: the result is whether the bottle scenarios get named as *A Study in Pink*
while the isomorph does not." CONTRACTS.md §9 puts those records in
`data/raw.jsonl` under `mode: "contamination"` and says to filter on mode.

Nothing in `analyze.py` touches them — deliberately, they are not part of the
factorial grid — so this reads them and lays them out. No API calls; run
`python -m src.runner --mode contamination --yes` first.

The recognition heuristic below is a *reading aid*, not a result. It flags
titles worth looking at; the verdict is the operator's, from the full text that
`--full` prints.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from src import contracts as c

#: Titles the bottle scenario is expected to evoke. PLAN.md §2's split is
#: whether the *bottle* scenarios get named while the *isomorph* does not, so
#: this list only has to be good enough to sort responses into "named
#: something" and "said NONE".
KNOWN_TITLES = (
    "a study in pink", "study in pink", "sherlock",
    "the princess bride", "princess bride",
    "battle of wits", "vizzini",
    "hamlet", "the westing game", "death note",
    "iocaine", "russian roulette",
)

NONE_MARKERS = ("none", "no,", "does not resemble", "doesn't resemble",
                "not resemble", "i don't recognise", "i do not recognise",
                "i'm not aware", "i am not aware", "no specific")


def load(raw_path: Path) -> list[dict]:
    if not raw_path.exists():
        sys.exit(f"{raw_path} does not exist. Run the contamination check first:\n"
                 f"  python -m src.runner --mode contamination --yes")
    out = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("mode") == c.CONTAMINATION_MODE:
            out.append(rec)
    return out


def text_of(rec: dict) -> str:
    choices = (rec.get("response") or {}).get("choices") or [{}]
    message = choices[0].get("message") or {}
    return (message.get("content") or "").strip()


def _truncated(rec: dict) -> bool:
    """`finish_reason: length` — the verdict may have been cut off mid-sentence.
    Six of the real 15 were, at `contracts.CONTAMINATION_MAX_TOKENS` = 200:
    the models that confabulate also ramble, and the answer comes last."""
    choices = (rec.get("response") or {}).get("choices") or [{}]
    return choices[0].get("finish_reason") == "length"


def verdict(text: str) -> tuple[str, str]:
    """(label, detail). `label` is NAMED / NONE / UNCLEAR.

    Measured 2026-09-03 against the real 15 responses. Three of the five models
    do not recognise anything; they *confabulate*, thrashing through a list of
    candidates and often rejecting each one — "The Hardest Logic Puzzle Ever …
    no, it is more like … no, I mean … NONE". An earlier version of this
    function counted any quoted phrase as a recognition, which scored every one
    of those as NAMED, including two that ended in an explicit NONE. That is
    the wrong way to be wrong for a check whose whole output is a yes/no.

    So: a trailing NONE is authoritative however much rambling precedes it, a
    known title is a real recognition, and a bare quoted phrase is UNCLEAR —
    "read this one" — never NAMED.
    """
    low = text.lower()

    # An explicit NONE VERDICT, wherever it sits. A fixed tail window was wrong:
    # gemma answers "**NONE**" on its own line and then adds a parenthetical
    # noting that films like *The Princess Bride* are NOT the match — so the
    # verdict is early, the title is late, and a tail-only check scored the
    # response NAMED off a counter-example.
    verdict_none = any(
        re.fullmatch(r"[\s*_`#>\-]*(none)[\s*_`.!:;,)]*", line, re.I)
        for line in text.splitlines()
    ) or bool(re.search(r"(answer|conclude|verdict)[^.\n]{0,24}\bnone\b", low))

    trailing_none = verdict_none or any(m in low[-260:] for m in NONE_MARKERS)
    hits = [t for t in KNOWN_TITLES if t in low]

    # Both signals present is genuinely ambiguous and must read as ambiguous.
    # gpt-4o-mini's isomorph response names *The Princess Bride*, explains the
    # Vizzini goblet mapping, and then hedges — 'If neither of these resonates
    # strongly, I'll conclude with "NONE," but "The Princess Bride" is a notable
    # reference'. Scoring the trailing NONE as the verdict there is a false
    # negative on the single most important cell in the table; scoring the title
    # as the verdict is a false positive on the three models that list
    # candidates only to dismiss them. There is no rule that gets both right,
    # which is the actual finding: on these five models this check cannot be
    # automated. UNCLEAR means read it.
    if hits and trailing_none:
        return "UNCLEAR", f"names {hits[0]} AND says NONE — read it"
    if trailing_none:
        return "NONE", ""
    if hits:
        return "NAMED", hits[0]

    if any(low.startswith(m) or f" {m}" in low[:200] for m in NONE_MARKERS):
        return "NONE", ""

    # Named something we do not recognise. Could be a real title this list is
    # missing, or a fabrication. Not decidable here — flag it for reading.
    quoted = re.findall(r'[""\"“]([^""\"”]{3,60})[""\"”]', text)
    if quoted:
        return "UNCLEAR", f"names: {quoted[0]}"
    return "UNCLEAR", ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=c.RAW_PATH)
    ap.add_argument("--full", action="store_true",
                    help="print every response in full underneath the table")
    ap.add_argument("--width", type=int, default=72,
                    help="characters of response to show in the table")
    args = ap.parse_args(argv)

    recs = load(args.raw)
    if not recs:
        sys.exit(f"no mode={c.CONTAMINATION_MODE!r} records in {args.raw}. Run:\n"
                 f"  python -m src.runner --mode contamination --yes")

    by_cell: dict[tuple[str, str], dict] = {}
    for rec in recs:
        by_cell[(rec["model"], rec["scenario"])] = rec

    models = [m for m in dict.fromkeys(r["model"] for r in recs)]
    scenarios = [s for s in c.SCENARIOS if any(s == r["scenario"] for r in recs)]

    print(f"CONTAMINATION CHECK — {len(recs)} record(s) from {args.raw}")
    print(f"{len(models)} model(s) x {len(scenarios)} scenario(s)")
    print()
    print("PLAN.md §2: the result you want is the bottle scenarios being named")
    print("while the isomorph is not. NAMED/NONE below is a reading aid; the")
    print("verdict is yours from the text.")
    print()

    mw = max(len("model"), max(len(m) for m in models))
    for scenario in scenarios:
        print(f"### {scenario}")
        print(f"  {'model'.ljust(mw)}  {'verdict'.ljust(8)}  what it named / said")
        print(f"  {'-' * mw}  {'-' * 8}  {'-' * args.width}")
        for model in models:
            rec = by_cell.get((model, scenario))
            if rec is None:
                print(f"  {model.ljust(mw)}  {'MISSING'.ljust(8)}  —")
                continue
            text = text_of(rec)
            label, detail = verdict(text)
            shown = detail if detail else " ".join(text.split())[:args.width]
            cut = "" if not _truncated(rec) else "  [CUT OFF]"
            print(f"  {model.ljust(mw)}  {label.ljust(8)}  {shown}{cut}")
        print()

    named = {s: sum(1 for m in models
                    if (rec := by_cell.get((m, s))) and verdict(text_of(rec))[0] == "NAMED")
             for s in scenarios}
    n_cut = sum(1 for r in recs if _truncated(r))
    if n_cut:
        print(f"** {n_cut} of {len(recs)} responses were CUT OFF at "
              f"max_tokens={c.CONTAMINATION_MAX_TOKENS}. **")
        print("   These models list candidates before concluding, so a cut-off")
        print("   response may have lost its verdict. Read those with --full and")
        print("   raise contracts.CONTAMINATION_MAX_TOKENS if the text is needed")
        print("   (that changes no run_id, so move the records aside first —")
        print("   CONTRACTS.md §2).")
        print()
    print("RECOGNITION COUNT (NAMED / total), per scenario")
    for scenario in scenarios:
        print(f"  {scenario.ljust(16)} {named[scenario]}/{len(models)}")
    print()
    bottles = [s for s in scenarios if s in ("default_keep", "default_switch")]
    if "isomorph" in scenarios and bottles:
        b = sum(named[s] for s in bottles)
        bt = len(bottles) * len(models)
        print(f"  bottle scenarios: {b}/{bt} named")
        print(f"  isomorph        : {named['isomorph']}/{len(models)} named")
        unclear = {s: sum(1 for m in models
                          if (rec := by_cell.get((m, s)))
                          and verdict(text_of(rec))[0] == "UNCLEAR")
                   for s in scenarios}
        print(f"  unclear         : bottles {sum(unclear[s] for s in bottles)}, "
              f"isomorph {unclear.get('isomorph', 0)}")
        print()
        if named["isomorph"] > 0:
            print("  ** THE ISOMORPH WAS RECOGNISED. PLAN.md §2: rewrite it. **")
        elif unclear.get("isomorph", 0):
            print("  ** VERDICT NOT AUTOMATABLE for the isomorph: "
                  f"{unclear['isomorph']} response(s) are UNCLEAR. **")
            print("  Read them with --full before concluding anything. Do NOT take the")
            print("  count above as the answer — the labels above are a reading aid and")
            print("  an UNCLEAR isomorph cell is exactly the case they cannot decide.")
        elif b > 0:
            print("  -> recognition split as PLAN.md §2 predicts: bottles named, isomorph not.")
        else:
            print("  -> nothing recognised anywhere; no contamination signal either way.")

    if args.full:
        print()
        print("=" * 78)
        print("FULL RESPONSES")
        print("=" * 78)
        for scenario in scenarios:
            for model in models:
                rec = by_cell.get((model, scenario))
                if rec is None:
                    continue
                print()
                print(f"--- {model} / {scenario} "
                      f"(provider={rec.get('provider')}, "
                      f"returned={rec.get('model_returned')}) ---")
                print(text_of(rec) or "(empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
