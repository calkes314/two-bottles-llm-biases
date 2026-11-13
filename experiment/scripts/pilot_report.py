#!/usr/bin/env python3
"""Print CoT responses in full, each followed by its annotation, plus the
diagnostics PLAN.md §9 step 5 says to read the pilot for.

    python -m scripts.pilot_report --config config/pilot.yaml
    python -m scripts.pilot_report --diagnostics-only

PLAN.md §9: "Step 5 is the one not to skip. You will find that your scenario
wording lets models weasel out, or that every model says the same thing, or that
the interesting variance sits somewhere you did not predict."

That requires reading the responses, so this prints them whole and does not
summarise them. The diagnostics underneath are mechanical checks for the four
failure modes worth catching before the full grid is paid for:

  1. weaselling out of the binary — RANDOMISE / THIRD_OPTION / REFUSE rates
  2. the final-line instruction being ignored — is the last line exactly one of
     the two words PLAN.md §2 asks for?
  3. truncation — `finish_reason: length` destroys the final line and reads as
     disobedience in the data (CONTRACTS.md §8, the reason COT_MAX_TOKENS went
     900 -> 2000)
  4. annotator error on an obvious case — where the final line IS exactly KEEP
     or SWITCH, that is ground truth, and the annotator disagreeing with it is
     an annotator bug, not a model result

Check 4 is the cheap kappa: it needs no hand-coding and covers every response
whose answer is unambiguous. PLAN.md §8.4's 60-response hand-coding pass is
still what settles the *depth* codes.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src import contracts as c
from src import manifest as manifest_mod

BINARY = ("KEEP", "SWITCH")
NONBINARY = ("RANDOMISE", "THIRD_OPTION", "REFUSE", "UNPARSEABLE")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def response_text(rec: dict) -> str:
    choices = (rec.get("response") or {}).get("choices") or [{}]
    return ((choices[0].get("message") or {}).get("content") or "")


def reasoning_text(rec: dict) -> str:
    """Some models put their trace in `message.reasoning`, where the annotator
    never sees it (CONTRACTS.md §8). Worth surfacing when content is short."""
    choices = (rec.get("response") or {}).get("choices") or [{}]
    return ((choices[0].get("message") or {}).get("reasoning") or "")


def finish_reason(rec: dict) -> str | None:
    choices = (rec.get("response") or {}).get("choices") or [{}]
    return choices[0].get("finish_reason")


def final_line_answer(text: str) -> str | None:
    """The answer PLAN.md §2's instruction asks for: a final line containing
    exactly one word, KEEP or SWITCH.

    Returns the word if the last non-empty line is that and nothing else after
    stripping surrounding markdown/punctuation, else None. `None` is the signal
    that the *format* instruction was not followed, which is a different fact
    from the model having no answer — `annotate.py` recovers the answer from
    anywhere in the text, and it is right to.
    """
    for line in reversed([l.strip() for l in text.splitlines()]):
        if not line:
            continue
        bare = re.sub(r"^[\s*_`#>\-]+|[\s*_`.!:;,]+$", "", line).upper()
        return bare if bare in BINARY else None
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=c.CONFIG_PATH,
                    help="restrict to the run this config describes")
    ap.add_argument("--raw", type=Path, default=c.RAW_PATH)
    ap.add_argument("--annotated", type=Path, default=c.ANNOTATED_PATH)
    ap.add_argument("--diagnostics-only", action="store_true",
                    help="skip the full responses, print only the checks")
    ap.add_argument("--all", action="store_true",
                    help="every cot record in --raw, ignoring --config")
    args = ap.parse_args(argv)

    raw = read_jsonl(args.raw)
    ann = {a["run_id"]: a for a in read_jsonl(args.annotated) if "run_id" in a}

    cot = [r for r in raw if r.get("mode") == "cot"]
    if not args.all:
        wanted = {j["run_id"] for j in manifest_mod.build_jobs(
            manifest_mod.load_config(args.config)) if j["mode"] == "cot"}
        cot = [r for r in cot if r.get("run_id") in wanted]
        print(f"restricted to the {len(wanted)} cot job(s) in {args.config}")

    if not cot:
        print(f"no matching cot records in {args.raw}.", file=sys.stderr)
        return 1

    # Group by (model, scenario), in config order, then persona/temperature/replicate.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for rec in cot:
        groups[(rec["model"], rec["scenario"])].append(rec)
    for key in groups:
        groups[key].sort(key=lambda r: (r.get("persona", ""), r.get("temperature", 0),
                                        r.get("replicate", 0)))

    n_missing_ann = sum(1 for r in cot if r["run_id"] not in ann)
    print(f"records: {len(cot)} cot   annotations: {len(cot) - n_missing_ann}"
          f"   missing: {n_missing_ann}")
    if n_missing_ann:
        print(f"  -> run: python -m src.annotate --config {args.config}")
    print()

    # ------------------------------------------------------- full responses ----
    if not args.diagnostics_only:
        n = 0
        for (model, scenario), recs in groups.items():
            print("=" * 78)
            print(f"### {model}  /  {scenario}   ({len(recs)} responses)")
            print("=" * 78)
            for rec in recs:
                n += 1
                a = ann.get(rec["run_id"])
                text = response_text(rec)
                usage = rec.get("usage") or {}
                print()
                print("-" * 78)
                print(f"[{n}] persona={rec.get('persona')} T={rec.get('temperature')} "
                      f"replicate={rec.get('replicate')}")
                print(f"     run_id={rec.get('run_id')}")
                print(f"     provider={rec.get('provider')} "
                      f"returned={rec.get('model_returned')} "
                      f"finish={finish_reason(rec)} "
                      f"out_tok={usage.get('completion_tokens')} "
                      f"cost=${usage.get('cost')}")
                print("-" * 78)
                print(text if text.strip() else "(empty content)")
                reasoning = reasoning_text(rec)
                if reasoning.strip():
                    print()
                    print(f"     [message.reasoning, {len(reasoning)} chars — the "
                          f"annotator does NOT see this]")
                    print(reasoning)
                print()
                fl = final_line_answer(text)
                if a is None:
                    print("     ANNOTATION: (none)")
                else:
                    flags = []
                    if a.get("flag_no_basis"):
                        flags.append("no_basis")
                    if a.get("flag_rigged"):
                        flags.append("rigged")
                    print(f"     ANNOTATION: answer={a.get('answer')} "
                          f"depth={a.get('depth')} "
                          f"flags={','.join(flags) or 'none'} "
                          f"parse_ok={a.get('parse_ok')} "
                          f"annotator={a.get('annotator_model')}")
                    if not a.get("parse_ok", True):
                        print(f"       raw_annotator_output: "
                              f"{(a.get('raw_annotator_output') or '')[:200]!r}")
                print(f"     FINAL LINE: {fl if fl else 'NOT one bare word — format ignored'}")
                if a is not None and fl and a.get("answer") != fl:
                    print(f"     ** DISAGREEMENT: final line says {fl}, "
                          f"annotator said {a.get('answer')} **")
            print()

    # ----------------------------------------------------------- diagnostics ----
    print("=" * 78)
    print("DIAGNOSTICS")
    print("=" * 78)
    print()

    answers = Counter(ann[r["run_id"]]["answer"] for r in cot if r["run_id"] in ann)
    unparse = answers.get("UNPARSEABLE", 0)
    total_ann = sum(answers.values())
    print("1. UNPARSEABLE")
    print(f"   {unparse} of {total_ann} annotated "
          f"({100 * unparse / total_ann if total_ann else 0:.1f}%)  "
          f"— PLAN.md §4 target: under 2%")
    bad_json = sum(1 for r in cot if r["run_id"] in ann
                   and not ann[r["run_id"]].get("parse_ok", True))
    print(f"   of which the annotator returned invalid JSON rather than coding "
          f"UNPARSEABLE: {bad_json}")
    if unparse:
        print("   PLAN.md §4: hand-review every instance. run_ids:")
        for r in cot:
            if r["run_id"] in ann and ann[r["run_id"]]["answer"] == "UNPARSEABLE":
                print(f"     {r['run_id']}  {r['model']} / {r['scenario']} "
                      f"T={r.get('temperature')} rep={r.get('replicate')}")
    print()

    print("2. ANSWER DISTRIBUTION PER CELL")
    hdr = (f"   {'model':<34} {'scenario':<15} {'n':>3} "
           + " ".join(f"{a[:6]:>6}" for a in c.ANSWERS))
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    for (model, scenario), recs in groups.items():
        counts = Counter(ann[r["run_id"]]["answer"] for r in recs if r["run_id"] in ann)
        print(f"   {model:<34} {scenario:<15} {len(recs):>3} "
              + " ".join(f"{counts.get(a, 0):>6}" for a in c.ANSWERS))
    print()

    print("3. WEASELLING OUT OF THE BINARY")
    nb_total = sum(answers.get(a, 0) for a in NONBINARY)
    print(f"   {nb_total} of {total_ann} responses are not KEEP or SWITCH "
          f"({100 * nb_total / total_ann if total_ann else 0:.1f}%)")
    for a in NONBINARY:
        if answers.get(a):
            print(f"     {a:<13} {answers[a]}")
    if not nb_total:
        print("     none — the scenario wording held")
    print()

    print("4. FINAL-LINE FORMAT COMPLIANCE  (PLAN.md §2's instruction)")
    hdr = f"   {'model':<34} {'scenario':<15} {'complied':>9} {'ignored':>8}"
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    offenders = []
    for (model, scenario), recs in groups.items():
        ok = bad = 0
        for r in recs:
            if final_line_answer(response_text(r)):
                ok += 1
            else:
                bad += 1
                offenders.append(r)
        print(f"   {model:<34} {scenario:<15} {ok:>9} {bad:>8}")
    if offenders:
        print(f"   {len(offenders)} response(s) did not end in one bare word. The "
              f"annotator recovers the answer from anywhere in the text, so this")
        print("   costs no data — but it is the thing PLAN.md §2 warns is a confound "
              "if it differs between the two scenarios.")
    print()

    print("5. TRUNCATION AND OUTPUT LENGTH  (CONTRACTS.md §8)")
    hdr = (f"   {'model':<34} {'median':>7} {'min':>6} {'max':>6} "
           f"{'cap':>5} {'at cap':>7} {'no CoT?':>8}")
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    for model in dict.fromkeys(r["model"] for r in cot):
        recs = [r for r in cot if r["model"] == model]
        toks = [int((r.get("usage") or {}).get("completion_tokens") or 0) for r in recs]
        chars = [len(response_text(r)) for r in recs]
        at_cap = sum(1 for r in recs if finish_reason(r) == "length")
        cap = max((r.get("max_tokens") or 0) for r in recs)
        med = statistics.median(toks) if toks else 0
        note = ""
        if med < 50:
            note = "NO CoT"
        elif at_cap > len(recs) / 2:
            note = "TRUNCATED"
        print(f"   {model:<34} {med:>7.0f} {min(toks or [0]):>6} {max(toks or [0]):>6} "
              f"{cap:>5} {at_cap:>7} {note:>8}")
    print("   median output under ~50 tokens = the model is not doing chain of "
          "thought and cannot be depth-coded")
    print("   (contracts.NO_COT_MODELS); most calls at cap = truncated before the "
          "final line.")
    hidden = [r for r in cot if len(reasoning_text(r)) > 200
              and len(response_text(r)) < 200]
    if hidden:
        print(f"   ** {len(hidden)} response(s) put their reasoning in "
              f"message.reasoning, which the annotator never sees. **")
    print()

    print("6. ANNOTATOR AGREEMENT ON UNAMBIGUOUS CASES")
    checked = disagree = skipped = 0
    rows = []
    for r in cot:
        a = ann.get(r["run_id"])
        fl = final_line_answer(response_text(r))
        if a is None:
            continue
        if not fl:
            # No bare final line, so there is no mechanical ground truth and
            # this record cannot be checked here. Counted, not silently
            # dropped: on the 40-call pilot every one of the annotator's five
            # answer errors was in THIS stratum, and reporting only the checked
            # half printed "21/21 agree (100.0%)" — a perfect score computed on
            # the subset where the annotator cannot fail.
            skipped += 1
            continue
        checked += 1
        if a["answer"] != fl:
            disagree += 1
            rows.append((r, fl, a["answer"]))
    if not checked:
        print("   no response ended in one bare word, so there is no ground truth "
              "to check against.")
    else:
        print(f"   {checked - disagree}/{checked} agree "
              f"({100 * (checked - disagree) / checked:.1f}%) where the final line "
              f"is itself unambiguous")
        print(f"   {skipped} of {checked + skipped} response(s) NOT CHECKABLE here — "
              f"no bare final line, so no mechanical ground truth.")
        if skipped:
            print("   Read that pair together. This check cannot fail on the "
                  "checked half by construction, and errors concentrate in the "
                  "other half: run `python -m scripts.extraction_check` for a "
                  "second opinion on the unchecked ones.")
        for r, fl, got in rows:
            print(f"     {r['run_id'][:12]} {r['model']} / {r['scenario']} "
                  f"T={r.get('temperature')} rep={r.get('replicate')}: "
                  f"final line {fl}, annotator {got}")
        if disagree:
            print("   These are annotator errors on obvious cases, not model results.")
    print()

    print("7. FLIP DIRECTION  (PLAN.md §2's critical manipulation)")
    print("   Conditional KEEP rate per (model, scenario): n(KEEP)/(n(KEEP)+n(SWITCH)),")
    print("   the denominator CONTRACTS.md §8 requires. flip = keep(default_keep) - "
          "keep(default_switch).")
    for model in dict.fromkeys(r["model"] for r in cot):
        rates = {}
        for scenario in ("default_keep", "default_switch"):
            recs = groups.get((model, scenario), [])
            cnt = Counter(ann[r["run_id"]]["answer"] for r in recs if r["run_id"] in ann)
            denom = cnt.get("KEEP", 0) + cnt.get("SWITCH", 0)
            rates[scenario] = (cnt.get("KEEP", 0) / denom if denom else None, denom)
        k, ks = rates.get("default_keep", (None, 0))
        s, ss = rates.get("default_switch", (None, 0))
        if k is None or s is None:
            print(f"   {model:<34} incomplete (need both scenarios)")
        else:
            print(f"   {model:<34} keep={k:.3f} (n={ks})  switch={s:.3f} (n={ss})  "
                  f"flip={k - s:+.3f}")
    print("   A pilot cell is 10 draws. These are direction only — no interval here "
          "is worth quoting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
