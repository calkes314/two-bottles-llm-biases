"""Deterministic job list + pre-flight cost estimate.

    python -m src.manifest --config config/budget.yaml --dry-run

`build_jobs(config)` returns the full list of job dicts specified by
CONTRACTS.md §3 — one per API call, self-describing, so the runner is a dumb
executor that adds nothing of its own. `estimate_cost(jobs, prices)` prices
that list per model, per mode, before a single call goes out.

Everything is derived from `config/budget.yaml` and `src.contracts`. Nothing
here hardcodes a grid size: CONTRACTS.md §3 is explicit that the direct count
follows `config.models[].logprobs` and must not be frozen at 45.

**Not built here, on purpose:** `direct_validation` jobs. CONTRACTS.md §8 fully
specifies that sub-run but says it is "a one-off script run after the direct
grid lands, not part of `manifest.py`'s default output" and that "the manifest
should not emit these jobs unless explicitly asked". `contracts.MODES` excludes
the mode so it cannot leak in through `answer_modes`. Whoever writes that script
must remember the trap CONTRACTS.md §8 names: validation jobs carry a *real*
temperature and a *real* replicate index, or all 200 draws collapse onto two
`run_id`s.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import yaml

from src import contracts as c
from src import prices as prices_mod

# Emission order. Fixed, because `build_jobs` must be byte-identical across
# runs and `--limit N` must select the same N jobs every time.
MODE_ORDER = (c.MODES[0], c.MODES[1], c.CONTAMINATION_MODE)   # direct, cot, contamination

#: Every key of a job dict, in CONTRACTS.md §3's order. The runner adds exactly
#: five more (§4) and renames nothing.
JOB_FIELDS = (
    "run_id", "experiment", "model", "tier", "scenario", "persona", "mode",
    "temperature", "replicate", "messages", "max_tokens", "logprobs",
    "top_logprobs", "provider_routing",
)

# ------------------------------------------------------------ token model ----

# CONTRACTS.md §10, measured 2026-09-03 on the real prompt: "**~210 prompt
# tokens** (199-201 without a persona, ~20 more with one)", against PLAN.md §6's
# estimated ~420. The anchor below is the `default_keep` / persona `none` / cot
# message set, which billed 201.
MEASURED_PROMPT_TOKENS = 201
MEASURED_PROMPT_REFERENCE = ("default_keep", "none", "cot")

# CONTRACTS.md §10 (and PLAN.md §6's revised table), measured medians.
# "Output tokens are not a constant and must not be modelled as one." They span
# two orders of magnitude here, which is exactly why this module never reports
# one blended number — a blended figure "would have hidden that the two rejected
# models would each have cost more than this entire study". A model missing from
# this table has no measurement and is estimated at its `max_tokens` ceiling,
# flagged as such.
#
# These are medians of a right-skewed distribution, not means, and the repo does
# not record which `max_tokens` cap they were measured under — so read them
# alongside the ceiling column, never on their own.
MEASURED_OUTPUT_TOKENS = {
    "qwen/qwen3-30b-a3b-instruct-2507": 3,
    "meta-llama/llama-3.1-8b-instruct": 218,
    "ibm-granite/granite-4.1-8b": 380,
    "google/gemma-4-26b-a4b-it": 476,
    "openai/gpt-4o-mini": 156,
}


def _chars_per_token() -> float:
    """Calibrate a chars→tokens ratio against the one real measurement we have.

    Scaling by characters rather than assuming a flat count per call keeps the
    scenario and persona differences in the estimate: `default_switch` is 110
    characters longer than `default_keep`, and a persona system message adds
    another ~120. Anchored this way the persona delta comes out at +14 to +27
    tokens, which brackets CONTRACTS.md §10's measured "~20".
    """
    scenario, persona, mode = MEASURED_PROMPT_REFERENCE
    reference = c.build_messages(scenario, persona, mode)
    return sum(len(m["content"]) for m in reference) / MEASURED_PROMPT_TOKENS


def estimate_prompt_tokens(job: dict) -> int:
    """Estimated billed prompt tokens for one job."""
    chars = sum(len(m["content"]) for m in job["messages"])
    return max(1, round(chars / _chars_per_token()))


def estimate_completion_tokens(job: dict) -> tuple[int, bool]:
    """`(tokens, measured)` for one job.

    The measured median is capped at the job's own `max_tokens`: a model whose
    CoT median is 476 tokens still cannot emit more than 4 in direct mode.
    """
    median = MEASURED_OUTPUT_TOKENS.get(job["model"])
    if median is None:
        return int(job["max_tokens"]), False
    return min(int(median), int(job["max_tokens"])), True


# ----------------------------------------------------------------- config ----

def load_config(path: str | Path = c.CONFIG_PATH) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Fail loudly on anything CONTRACTS.md §1 forbids, before any spend."""
    unknown = set(cfg["scenarios"]) - set(c.SCENARIOS)
    if unknown:
        raise ValueError(f"config scenarios not in contracts.SCENARIOS: {sorted(unknown)}")
    unknown = set(cfg["personas"]) - set(c.PERSONAS)
    if unknown:
        raise ValueError(f"config personas not in contracts.PERSONAS: {sorted(unknown)}")
    unknown = set(cfg["answer_modes"]) - set(c.MODES)
    if unknown:
        raise ValueError(
            f"config answer_modes not in contracts.MODES: {sorted(unknown)}. "
            f"{c.CONTAMINATION_MODE!r} and {c.DIRECT_VALIDATION_MODE!r} are side runs "
            "and must never appear here (CONTRACTS.md §8, §9)."
        )
    for name in cfg["scenarios"]:
        if not (c.SCENARIO_DIR / f"{name}.txt").exists():
            raise FileNotFoundError(f"scenarios/{name}.txt is missing")
    if int(cfg["replicates_cot"]) < 1:
        raise ValueError("replicates_cot must be >= 1")
    if int(cfg["max_concurrency"]) < 1:
        raise ValueError("max_concurrency must be >= 1")


# -------------------------------------------------------------- job build ----

def _job(cfg: dict, model: dict, scenario: str, persona: str, mode: str,
         temperature: float, replicate: int) -> dict:
    """One job dict, exactly the fields in CONTRACTS.md §3 and in that order."""
    if mode == c.CONTAMINATION_MODE:
        messages = c.build_contamination_messages(scenario)
        max_tokens = c.CONTAMINATION_MAX_TOKENS
    else:
        messages = c.build_messages(
            scenario, persona, mode, bool(cfg["allow_randomise_in_prompt"])
        )
        max_tokens = c.DIRECT_MAX_TOKENS if mode == "direct" else c.COT_MAX_TOKENS

    is_direct = mode == "direct"
    return {
        "run_id": c.run_id(model["id"], scenario, persona, mode, temperature, replicate),
        "experiment": cfg["experiment"],
        "model": model["id"],
        "tier": model["tier"],
        "scenario": scenario,
        "persona": persona,
        "mode": mode,
        "temperature": float(temperature),
        "replicate": int(replicate),
        "messages": messages,
        "max_tokens": int(max_tokens),
        "logprobs": is_direct,
        "top_logprobs": c.DIRECT_TOP_LOGPROBS if is_direct else None,
        # EVERY mode pins its provider, not just direct. Providers are not the
        # same computational object — measured, see contracts.COT_PROVIDER_PIN
        # — and leaving cot unpinned confounds PLAN.md's direct-vs-cot question
        # with a provider difference. `pin_providers: false` in the config
        # restores free routing for throughput; direct always pins regardless.
        # contracts.provider_pin returns a fresh dict per job, so no job can
        # mutate the shared pin.
        "provider_routing": c.provider_pin(
            model["id"], mode, enabled=bool(cfg.get("pin_providers", True))),
    }


def build_jobs(config: dict) -> list[dict]:
    """The full job list for one config. Deterministic: same config in, byte-
    identical list out.

    Emission order is mode (direct, cot, contamination), then model, scenario,
    persona, temperature, replicate — all in config order, never set or dict
    iteration order. `--limit N` therefore always selects the same N jobs.

    - **direct** (CONTRACTS.md §3, §8): one job per (model, scenario, persona),
      `temperature=1.0`, `replicate=0`, and *only* for models the config marks
      `logprobs: true`. Every such model must have a
      `contracts.DIRECT_PROVIDER_PIN` entry; see the error below for why.
    - **cot**: the full cross product including temperatures and replicates.
    - **contamination** (§9): 1 per (model, scenario), persona `none`. It is not
      part of the factorial grid and is deliberately *not* gated on
      `answer_modes`, which CONTRACTS.md §1 restricts to `direct`/`cot`. Filter
      it out at run time with the runner's `--mode` if you do not want it.
    """
    validate_config(config)
    modes = set(config["answer_modes"])
    jobs: list[dict] = []

    for mode in MODE_ORDER:
        if mode in c.MODES and mode not in modes:
            continue
        for model in config["models"]:
            if mode == "direct":
                if not model["logprobs"]:
                    # CONTRACTS.md §3: a model that cannot return logprobs can
                    # never contribute a temperature curve, so it gets no
                    # direct job at all.
                    continue
                if model["id"] not in c.DIRECT_PROVIDER_PIN:
                    raise KeyError(
                        f"{model['id']} is marked logprobs: true in the config but has no "
                        f"contracts.DIRECT_PROVIDER_PIN entry. CONTRACTS.md §8: an unpinned "
                        f"direct call gets routed to a provider that silently returns no "
                        f"logprobs, and the whole direct grid is wasted spend. Add a verified "
                        f"provider pin, or set logprobs: false to skip this model's direct arm."
                    )
            for scenario in config["scenarios"]:
                if mode == c.CONTAMINATION_MODE:
                    jobs.append(_job(config, model, scenario, "none", mode,
                                     c.CONTAMINATION_TEMPERATURE, 0))
                    continue
                for persona in config["personas"]:
                    if mode == "direct":
                        jobs.append(_job(config, model, scenario, persona, mode,
                                         c.DIRECT_TEMPERATURE, 0))
                        continue
                    for temperature in config["temperatures"]:
                        for replicate in range(int(config["replicates_cot"])):
                            jobs.append(_job(config, model, scenario, persona, mode,
                                             temperature, replicate))

    # `run_id` is the only join key and the only thing resume looks at
    # (CONTRACTS.md §2), so a collision is silent: the runner would write one
    # record and skip the rest, or three stages would disagree about which cell
    # a row belongs to. Cheap to check, impossible to notice later.
    seen: set[str] = set()
    clashes = [job for job in jobs
               if job["run_id"] in seen or seen.add(job["run_id"]) is not None]
    if clashes:
        first = clashes[0]
        raise AssertionError(
            f"{len(clashes)} duplicate run_id(s) in the manifest, e.g. "
            + repr(c.run_id_key(first["model"], first["scenario"], first["persona"],
                                first["mode"], first["temperature"], first["replicate"]))
        )
    return jobs


def count_jobs(jobs: list[dict]) -> dict:
    """Counts by mode, by model, and by (model, mode). Insertion-ordered, so
    printing them reproduces the emission order."""
    by_mode: dict[str, int] = {}
    by_model: dict[str, int] = {}
    by_model_mode: dict[str, dict[str, int]] = {}
    for job in jobs:
        by_mode[job["mode"]] = by_mode.get(job["mode"], 0) + 1
        by_model[job["model"]] = by_model.get(job["model"], 0) + 1
        cell = by_model_mode.setdefault(job["model"], {})
        cell[job["mode"]] = cell.get(job["mode"], 0) + 1
    return {"total": len(jobs), "by_mode": by_mode, "by_model": by_model,
            "by_model_mode": by_model_mode}


# ------------------------------------------------------------------ costs ----

#: No real OpenRouter model costs a thousandth of a dollar per *token* — that
#: would be $1,000 per million. A price above this is per-million numbers
#: arriving where per-token ones belong, which is a silent factor of 1e6.
IMPLAUSIBLE_PRICE_PER_TOKEN = 1e-3


def _check_price_unit(prices: dict[str, dict[str, float]]) -> None:
    """Refuse a price table that is obviously quoted per million tokens.

    PLAN.md §6's formula ends in `/ 1e6` because it assumes per-million prices;
    `prices.fetch_prices()` returns per-token. Mixing the two changes nothing a
    human would notice — $0.54 and $0.00000054 both read as "under $5" — which
    is exactly why the pre-spend gate has to catch it mechanically.
    """
    for model_id, pair in (prices or {}).items():
        for field in ("prompt", "completion"):
            value = pair.get(field)
            if isinstance(value, (int, float)) and value > IMPLAUSIBLE_PRICE_PER_TOKEN:
                raise ValueError(
                    f"{model_id} {field} price is {value}, which is not USD per token "
                    f"(that would be ${value * 1e6:,.0f} per million). estimate_cost "
                    f"expects prices.fetch_prices()'s unit: USD per token, no /1e6."
                )


def estimate_cost(jobs: list[dict], prices: dict[str, dict[str, float]]) -> dict:
    """Price a job list, **per model**, with an honest uncertainty band.

    PLAN.md §6's formula is

        cost = n_calls × (in_tokens × price_in + out_tokens × price_out) / 1e6

    where the `/ 1e6` assumes prices quoted per *million* tokens.
    `prices.fetch_prices()` returns OpenRouter's own unit, USD per *token*, so
    there is no divisor here. Do not reintroduce one.

    Two numbers come back for every model, and the gap between them is the
    point:

    - `cost` uses the measured median output length (PLAN.md §6). It is the
      realistic figure.
    - `cost_ceiling` assumes every call runs to `max_tokens`. It is what you
      pay if the model rambles, which verbose models do.

    A terse model returns 3 tokens and a verbose one runs to the cap, so a
    single blended per-call number would be wrong for every model in the set.
    There is a `total`, but read the breakdown before believing it — and check
    `models_missing_price`, whose calls are counted but contribute $0.

    The total covers **the grid only**. The annotation pass (one cheap call per
    CoT response, PLAN.md §6's third row) belongs to `src/annotate.py` and is
    not in this job list, so this figure is ~$0.09 short of CONTRACTS.md §10's
    "full study ~$0.54".
    """
    _check_price_unit(prices)
    by_model: dict[str, dict] = {}

    for job in jobs:
        model_id = job["model"]
        entry = by_model.get(model_id)
        if entry is None:
            pair = prices_mod.lookup(prices, model_id) if prices else None
            entry = by_model[model_id] = {
                "tier": job["tier"],
                "price_prompt": pair["prompt"] if pair else None,
                "price_completion": pair["completion"] if pair else None,
                "output_measured": model_id in MEASURED_OUTPUT_TOKENS,
                "n_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "completion_tokens_ceiling": 0,
                "cost": 0.0,
                "cost_ceiling": 0.0,
                "by_mode": {},
            }

        prompt_tokens = estimate_prompt_tokens(job)
        completion_tokens, _ = estimate_completion_tokens(job)
        ceiling_tokens = int(job["max_tokens"])

        price_in = entry["price_prompt"] or 0.0
        price_out = entry["price_completion"] or 0.0
        cost = prompt_tokens * price_in + completion_tokens * price_out
        cost_ceiling = prompt_tokens * price_in + ceiling_tokens * price_out

        cell = entry["by_mode"].setdefault(job["mode"], {
            "n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "completion_tokens_ceiling": 0, "cost": 0.0, "cost_ceiling": 0.0,
        })
        for target in (entry, cell):
            target["n_calls"] += 1
            target["prompt_tokens"] += prompt_tokens
            target["completion_tokens"] += completion_tokens
            target["completion_tokens_ceiling"] += ceiling_tokens
            target["cost"] += cost
            target["cost_ceiling"] += cost_ceiling

    total = {"n_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "completion_tokens_ceiling": 0, "cost": 0.0, "cost_ceiling": 0.0}
    for entry in by_model.values():
        for field in total:
            total[field] += entry[field]

    return {
        "unit": "usd",
        "by_model": by_model,
        "total": total,
        "models_missing_price": [m for m, e in by_model.items() if e["price_prompt"] is None],
        "models_without_measured_output": [m for m, e in by_model.items()
                                           if not e["output_measured"]],
    }


# --------------------------------------------------------------- printing ----

def _usd(amount: float) -> str:
    return f"${amount:,.4f}" if amount < 100 else f"${amount:,.2f}"


def format_dry_run(jobs: list[dict], estimate: dict, config: dict,
                   prices_available: bool = True) -> str:
    """The `--dry-run` report: counts by mode and model, then the cost
    breakdown, then the total. Also states which models were given a direct arm
    and which were skipped, as CONTRACTS.md §3 asks."""
    counts = count_jobs(jobs)
    lines: list[str] = []

    lines.append(f"experiment: {config['experiment']}   jobs: {counts['total']:,}")
    lines.append("")
    lines.append("JOBS BY MODE")
    for mode in MODE_ORDER:
        if mode in counts["by_mode"]:
            lines.append(f"  {mode:<16}{counts['by_mode'][mode]:>8,}")
    lines.append(f"  {'TOTAL':<16}{counts['total']:>8,}")

    lines.append("")
    lines.append("JOBS BY MODEL AND MODE")
    modes_seen = [m for m in MODE_ORDER if m in counts["by_mode"]]
    header = f"  {'model':<36}" + "".join(f"{m:>15}" for m in modes_seen) + f"{'total':>10}"
    lines.append(header)
    for model in config["models"]:
        model_id = model["id"]
        cells = counts["by_model_mode"].get(model_id, {})
        row = f"  {model_id:<36}" + "".join(f"{cells.get(m, 0):>15,}" for m in modes_seen)
        lines.append(row + f"{counts['by_model'].get(model_id, 0):>10,}")

    # CONTRACTS.md §3: "Whichever the manifest does, say so in its --dry-run
    # output." Derived from the jobs being reported, never from the config —
    # under a --models filter or a partial resume, a config-derived "5/5" would
    # assert a coverage this run will not deliver, in the one sentence a reader
    # trusts when deciding whether the temperature-curve arm is covered.
    if counts["by_mode"].get("direct"):
        direct_on = list(dict.fromkeys(job["model"] for job in jobs if job["mode"] == "direct"))
        present = list(dict.fromkeys(job["model"] for job in jobs))
        skipped = [model["id"] for model in config["models"]
                   if not model["logprobs"] and model["id"] in set(present)]
        lines.append("")
        lines.append(f"  direct arm present for {len(direct_on)}/{len(present)} of the models "
                     f"in this run, each pinned to a verified provider.")
        if skipped:
            lines.append(f"  direct arm SKIPPED for: {', '.join(skipped)} "
                         f"— logprobs: false, so no temperature curve is possible.")
        if len(direct_on) < len(config["models"]):
            lines.append(f"  (the full config has {len(config['models'])} models; this report "
                         f"covers only the jobs listed above.)")

    lines.append("")
    lines.append("ESTIMATED COST  (median output = realistic; ceiling = every call "
                 "runs to max_tokens)")
    if not prices_available:
        lines.append("  !! live prices unavailable and no cache — token counts only, no dollars.")
    lines.append(f"  {'model':<36}{'$/M in':>9}{'$/M out':>9}{'in tok':>12}"
                 f"{'out tok':>11}{'median':>11}{'ceiling':>11}")
    for model in config["models"]:
        entry = estimate["by_model"].get(model["id"])
        if entry is None:
            continue
        p_in = f"{entry['price_prompt'] * 1e6:.3f}" if entry["price_prompt"] is not None else "?"
        p_out = (f"{entry['price_completion'] * 1e6:.3f}"
                 if entry["price_completion"] is not None else "?")
        star = " " if entry["output_measured"] else "*"
        lines.append(
            f"  {model['id']:<36}{p_in:>9}{p_out:>9}{entry['prompt_tokens']:>12,}"
            f"{entry['completion_tokens']:>10,}{star}{_usd(entry['cost']):>11}"
            f"{_usd(entry['cost_ceiling']):>11}"
        )
    total = estimate["total"]
    lines.append(f"  {'TOTAL':<36}{'':>9}{'':>9}{total['prompt_tokens']:>12,}"
                 f"{total['completion_tokens']:>11,}{_usd(total['cost']):>11}"
                 f"{_usd(total['cost_ceiling']):>11}")

    lines.append("")
    lines.append(f"  ESTIMATE {_usd(total['cost'])}   "
                 f"CEILING {_usd(total['cost_ceiling'])}   "
                 f"over {total['n_calls']:,} calls")
    lines.append("  Per-model, not blended: output length is a model property spanning two "
                 "orders of")
    lines.append("  magnitude here (3 tokens to the cap), and the medians are of a "
                 "right-skewed")
    lines.append("  distribution. Actual spend comes from each response's usage.cost.")
    lines.append("  GRID ONLY — the annotation pass belongs to src/annotate.py and is not "
                 "counted here")
    lines.append("  (CONTRACTS.md §10 puts it at ~$0.09, for a full study of ~$0.54).")

    if estimate["models_without_measured_output"]:
        lines.append(f"  * no measured output median — estimated at the max_tokens ceiling: "
                     f"{', '.join(estimate['models_without_measured_output'])}")
    if estimate["models_missing_price"]:
        lines.append(f"  !! NOT PRICED, counted as $0, the total is an UNDER-estimate: "
                     f"{', '.join(estimate['models_missing_price'])}")

    return "\n".join(lines)


# -------------------------------------------------------------------- CLI ----

def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the job manifest and print a dry-run cost estimate. "
                    "Makes no model calls."
    )
    parser.add_argument("--config", type=Path, default=c.CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="print counts and the cost breakdown (the default)")
    parser.add_argument("--json", action="store_true",
                        help="write the job list as JSON to stdout instead")
    parser.add_argument("--refresh-prices", action="store_true",
                        help="ignore the 24h price cache and refetch")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    jobs = build_jobs(config)

    if args.json:
        json.dump(jobs, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0

    prices, available = {}, True
    try:
        prices = prices_mod.fetch_prices(force=args.refresh_prices)
    except Exception as exc:
        print(f"warning: no prices available ({type(exc).__name__}: {exc})", file=sys.stderr)
        available = False

    print(format_dry_run(jobs, estimate_cost(jobs, prices), config, available))
    print("\nDry run only. No model was called.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
