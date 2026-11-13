#!/usr/bin/env python3
"""PLAN.md §8.3's empirical validation of the analytic temperature curve.

    python -m scripts.direct_validation --dry-run     # 200 jobs, no calls
    python -m scripts.direct_validation --yes         # run them
    python -m scripts.direct_validation --report      # compare, no calls

CONTRACTS.md §8 specifies this sub-run completely and then says it is "a one-off
script run after the direct grid lands, not part of `manifest.py`'s default
output". `manifest.py` therefore refuses to emit these jobs and
`contracts.MODES` excludes the mode so it cannot arrive through `answer_modes`.
This is that script.

    model        openai/gpt-4o-mini   (the only pin measured aligned and clean
                                       on every endpoint)
    scenario     default_keep
    persona      none
    temperature  0.4 and 1.2          (the two levels PLAN.md §8.3 names)
    replicate    0..99 at each        -> 200 calls
    prompt       byte-identical to the direct prompt
    max_tokens / logprobs / top_logprobs / provider_routing   as direct mode

**The trap CONTRACTS.md §8 names, avoided here.** A validation job carries a
REAL temperature and a REAL replicate index, because it is a sampling run. Reuse
direct mode's `temperature=1.0, replicate=0` and all 200 draws collapse onto two
`run_id`s, so resume writes the first of each and silently discards the other
198. `--dry-run` asserts 200 distinct ids before anything is spent.

What the test is: the observed KEEP/SWITCH proportions from 100 real draws at
each temperature, against what `curve.rescale` predicts from the SAME cell's
T=1.0 logprobs in the direct grid. PLAN.md §8.3 expects providers not to apply
temperature identically, and top-p truncation to interact with it.
**A disagreement is the reportable methods result, not a bug to fix.**
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

from src import contracts as c
from src import curve as curve_mod
from src import manifest as manifest_mod
from src import prices as prices_mod
from src import runner as runner_mod

# CONTRACTS.md §8's table, verbatim. Not configurable: the pin, the scenario and
# the persona are all part of what makes the comparison valid.
VALIDATION_MODEL = "openai/gpt-4o-mini"
VALIDATION_SCENARIO = "default_keep"
VALIDATION_PERSONA = "none"
VALIDATION_TEMPERATURES = (0.4, 1.2)
VALIDATION_REPLICATES = 100

MODE = c.DIRECT_VALIDATION_MODE


# ------------------------------------------------------------------ jobs ----

def build_validation_jobs(tier: str = "mid") -> list[dict]:
    """The 200 jobs. Same shape as a direct job (CONTRACTS.md §3, §8) except
    that `mode`, `temperature` and `replicate` are the validation ones."""
    jobs: list[dict] = []
    for temperature in VALIDATION_TEMPERATURES:
        for replicate in range(VALIDATION_REPLICATES):
            jobs.append({
                "run_id": c.run_id(VALIDATION_MODEL, VALIDATION_SCENARIO,
                                   VALIDATION_PERSONA, MODE, temperature, replicate),
                "experiment": None,          # filled from config below
                "model": VALIDATION_MODEL,
                "tier": tier,
                "scenario": VALIDATION_SCENARIO,
                "persona": VALIDATION_PERSONA,
                "mode": MODE,
                # REAL temperature, REAL replicate. This is the whole point.
                "temperature": float(temperature),
                "replicate": int(replicate),
                "messages": c.build_messages(VALIDATION_SCENARIO, VALIDATION_PERSONA,
                                             MODE, False),
                "max_tokens": c.DIRECT_MAX_TOKENS,
                "logprobs": True,
                "top_logprobs": c.DIRECT_TOP_LOGPROBS,
                "provider_routing": dict(c.DIRECT_PROVIDER_PIN[VALIDATION_MODEL]),
            })
    return jobs


def _check_distinct(jobs: list[dict]) -> None:
    ids = {j["run_id"] for j in jobs}
    expected = len(VALIDATION_TEMPERATURES) * VALIDATION_REPLICATES
    if len(jobs) != expected or len(ids) != expected:
        sys.exit(f"FATAL: built {len(jobs)} job(s) with {len(ids)} distinct run_id(s); "
                 f"expected {expected} of each. CONTRACTS.md §8: a validation job must "
                 f"carry a real temperature and a real replicate index, or resume "
                 f"discards every draw but the first.")


def _check_prompt_matches_direct() -> None:
    """CONTRACTS.md §8: "The prompt is byte-identical to the direct prompt."""
    a = c.scenario_prompt(VALIDATION_SCENARIO, MODE)
    b = c.scenario_prompt(VALIDATION_SCENARIO, "direct")
    if a != b:
        sys.exit("FATAL: the validation prompt is not byte-identical to the direct "
                 "prompt. The comparison would be against a different question.")


# -------------------------------------------------------------- readback ----

def sampled_letter(record: dict) -> str | None:
    """The letter the model actually EMITTED, from `message.content`.

    Deliberately not from the logprobs: this run validates the provider's
    *sampling*, so the observed proportion has to come from what was sampled.
    """
    choices = (record.get("response") or {}).get("choices") or [{}]
    content = (choices[0].get("message") or {}).get("content")
    if not content:
        return None
    for ch in str(content).strip():
        up = ch.upper()
        if up in c.DIRECT_ANSWER_TOKENS:
            return up
        if not ch.isspace():
            return None          # emitted something else first; not an answer
    return None


def read_records(raw_path: Path, mode: str) -> list[dict]:
    if not raw_path.exists():
        return []
    out = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("mode") == mode:
            out.append(rec)
    return out


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval — behaves at proportions near 0 and 1, where the
    normal approximation gives bounds outside [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def predicted_from_grid(raw_path: Path) -> tuple[dict[str, float] | None, str, dict]:
    """The T=1.0 logprobs for the matching direct-grid cell, and its provenance.

    Returns (logprobs, status, provenance). The prediction has to come from the
    SAME (model, scenario, persona) cell — CONTRACTS.md §8 measured P(KEEP)
    moving from 0.854 to 0.986 across providers for one model, so a prediction
    read off a different provider's draw is not a prediction about this one.
    """
    want = c.run_id(VALIDATION_MODEL, VALIDATION_SCENARIO, VALIDATION_PERSONA,
                    "direct", c.DIRECT_TEMPERATURE, 0)
    for rec in read_records(raw_path, "direct"):
        if rec.get("run_id") != want:
            continue
        logprobs, status = curve_mod.extract_answer_logprobs(rec)
        return logprobs, status, {
            "provider": rec.get("provider"),
            "model_returned": rec.get("model_returned"),
            "ts": rec.get("ts"),
        }
    return None, "missing_direct_record", {}


# ---------------------------------------------------------------- report ----

def report(raw_path: Path) -> int:
    records = read_records(raw_path, MODE)
    if not records:
        print(f"no mode={MODE!r} records in {raw_path}. Run the sub-run first:\n"
              f"  python -m scripts.direct_validation --yes", file=sys.stderr)
        return 1

    logprobs, status, prov = predicted_from_grid(raw_path)

    print("=" * 78)
    print("TEMPERATURE VALIDATION — PLAN.md §8.3, CONTRACTS.md §8")
    print("=" * 78)
    print(f"model     {VALIDATION_MODEL}")
    print(f"scenario  {VALIDATION_SCENARIO}   persona {VALIDATION_PERSONA}")
    print(f"pin       {c.DIRECT_PROVIDER_PIN[VALIDATION_MODEL]}")
    print(f"records   {len(records)} of {len(VALIDATION_TEMPERATURES) * VALIDATION_REPLICATES}")
    print()
    print("PREDICTION SOURCE — the same cell's T=1.0 direct-grid logprobs")
    if status != "ok" or logprobs is None:
        print(f"  UNAVAILABLE: {status}")
        print("  The direct grid must land first (python -m src.runner --mode direct --yes);")
        print("  without it there is nothing to compare the observed draws against.")
    else:
        pk = math.exp(logprobs["K"])
        ps = math.exp(logprobs["S"])
        print(f"  provider={prov.get('provider')}  returned={prov.get('model_returned')}"
              f"  ts={prov.get('ts')}")
        print(f"  logprob(K)={logprobs['K']:+.4f}  logprob(S)={logprobs['S']:+.4f}")
        print(f"  P(KEEP | answer) at T=1.0 = {pk / (pk + ps):.4f}")
    print()

    header = (f"  {'T':>4}  {'n':>4}  {'KEEP':>5}  {'observed':>8}  {'95% Wilson':>16}  "
              f"{'predicted':>9}  {'diff':>7}  {'in CI':>5}")
    print("OBSERVED vs PREDICTED — P(KEEP | KEEP or SWITCH)")
    print(header)
    print("  " + "-" * (len(header) - 2))

    disagreements = []
    for T in VALIDATION_TEMPERATURES:
        at_T = [r for r in records if abs(float(r.get("temperature", -1)) - T) < 1e-9]
        letters = [sampled_letter(r) for r in at_T]
        keep = sum(1 for x in letters if x == "K")
        switch = sum(1 for x in letters if x == "S")
        neither = sum(1 for x in letters if x is None)
        n = keep + switch
        obs = keep / n if n else float("nan")
        lo, hi = wilson(keep, n)
        if status == "ok" and logprobs is not None:
            pred = curve_mod.rescale(logprobs, float(T))["K"]
            diff = obs - pred
            inside = lo <= pred <= hi
            print(f"  {T:>4.1f}  {n:>4}  {keep:>5}  {obs:>8.4f}  "
                  f"[{lo:.4f}, {hi:.4f}]  {pred:>9.4f}  {diff:>+7.4f}  "
                  f"{'yes' if inside else 'NO':>5}")
            if not inside:
                disagreements.append((T, obs, pred, lo, hi))
        else:
            print(f"  {T:>4.1f}  {n:>4}  {keep:>5}  {obs:>8.4f}  "
                  f"[{lo:.4f}, {hi:.4f}]  {'—':>9}  {'—':>7}  {'—':>5}")
        if neither:
            print(f"        ({neither} record(s) at T={T} emitted neither letter; "
                  f"excluded from n, as the curve's denominator is {{K, S}})")

    print()
    if status != "ok" or logprobs is None:
        print("VERDICT: not computable — no T=1.0 logprobs for this cell.")
    elif not disagreements:
        print("VERDICT: the analytic curve is inside the observed 95% interval at every")
        print("level tested. The direct grid's dashed overlay can be presented as is.")
    else:
        print("VERDICT: the analytic curve falls OUTSIDE the observed 95% interval at "
              f"{len(disagreements)} of {len(VALIDATION_TEMPERATURES)} level(s):")
        for T, obs, pred, lo, hi in disagreements:
            print(f"  T={T}: observed {obs:.4f} [{lo:.4f}, {hi:.4f}], predicted {pred:.4f}")
        print()
        print("PLAN.md §8.3 predicts exactly this — providers do not all apply")
        print("temperature identically, and top-p truncation interacts with it. This is")
        print("a reportable methods result, NOT a defect to patch: present the direct")
        print("grid's curve with this disagreement as its caveat. Do not adjust either")
        print("number to make them agree.")
    return 0


# ------------------------------------------------------------------ main ----

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=c.CONFIG_PATH)
    ap.add_argument("--raw", type=Path, default=c.RAW_PATH)
    ap.add_argument("--dry-run", action="store_true",
                    help="build and price the 200 jobs, call nothing")
    ap.add_argument("--report", action="store_true",
                    help="compare observed against predicted, call nothing")
    ap.add_argument("--yes", "-y", action="store_true", help="skip the confirmation")
    ap.add_argument("--concurrency", type=int, default=None)
    args = ap.parse_args(argv)

    if args.report:
        return report(args.raw)

    config = manifest_mod.load_config(args.config)
    _check_prompt_matches_direct()
    jobs = build_validation_jobs()
    for job in jobs:
        job["experiment"] = config["experiment"]
    _check_distinct(jobs)

    already = runner_mod.done_ids(args.raw)
    pending = [j for j in jobs if j["run_id"] not in already]

    print(f"validation jobs : {len(jobs)}  ({len(jobs) - len(pending)} already in {args.raw})")
    print(f"distinct run_ids: {len({j['run_id'] for j in jobs})}  <- must equal "
          f"{len(VALIDATION_TEMPERATURES) * VALIDATION_REPLICATES}")
    print(f"temperatures    : {sorted({j['temperature'] for j in jobs})}")
    print(f"replicates      : {min(j['replicate'] for j in jobs)}.."
          f"{max(j['replicate'] for j in jobs)}")
    print(f"mode            : {MODE}  (not in contracts.MODES, so the grid cannot "
          f"absorb it)")

    prices = {}
    try:
        prices = prices_mod.fetch_prices()
    except Exception as exc:                                    # pragma: no cover
        print(f"warning: no prices ({type(exc).__name__}: {exc})", file=sys.stderr)
    if prices and pending:
        total = manifest_mod.estimate_cost(pending, prices)["total"]
        print(f"estimated cost  : ${total['cost']:.4f} "
              f"(ceiling ${total['cost_ceiling']:.4f}) over {total['n_calls']} call(s)")

    if args.dry_run:
        print("\nDry run only. No model was called.")
        return 0
    if not pending:
        print("\nNothing to do. Use --report to compare.")
        return 0
    if not args.yes:
        reply = input(f"\nRun {len(pending)} call(s)? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("aborted")
            return 1

    api_key = c.load_api_key()
    if not api_key:
        print(f"OPENROUTER_API_KEY is not set. Export it, or put it in {c.ENV_PATH}.",
              file=sys.stderr)
        return 2

    state = asyncio.run(runner_mod.run_jobs(
        pending, api_key=api_key,
        max_concurrency=args.concurrency or int(config["max_concurrency"]),
        raw_path=args.raw, prices=prices, progress_every=50,
    ))
    print()
    print(runner_mod._summarise(state, len(jobs) - len(pending)))
    print()
    return report(args.raw)


if __name__ == "__main__":
    raise SystemExit(main())
