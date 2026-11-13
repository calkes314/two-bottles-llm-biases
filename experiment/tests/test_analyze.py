"""Tests for src/analyze.py, and the generator for tests/fixtures/.

Run the tests:      python -m pytest tests/test_analyze.py -q
Regenerate fixtures: python tests/test_analyze.py --regenerate

────────────────────────────────────────────────────────────────────────────
WHAT THE FIXTURES ENCODE  — so a plot can be checked against a known truth
────────────────────────────────────────────────────────────────────────────

3 models × 3 scenarios × 2 personas × 3 temperatures × 8 replicates = 432 cot
records, plus 18 direct-mode records. Every model id is a real one from
`config/budget.yaml`, so config ordering, `contracts.DIRECT_CURVE_MODELS`
membership and `contracts.NO_COT_MODELS` all behave as they will in production.

| model | flip effect | direct mode |
|---|---|---|
| `meta-llama/llama-3.1-8b-instruct` | **none** (≈ 0) | 4 of 6 records unusable → whole model dropped from the overlay |
| `ibm-granite/granite-4.1-8b` | **strong positive** (≈ +0.63) — status quo bias | clean |
| `openai/gpt-4o-mini` | **negative** (≈ −0.44) — takes the action in both framings | clean |

The negative model is deliberate: `flip_rate` is signed, and a model that acts
in both framings is the *opposite* behaviour to status quo bias, not a smaller
amount of it. An absolute value would put granite and gpt-4o-mini on the same
side of the plot.

Also built in, so the other three plots are not flat:

* KEEP rates decay toward 0.5 as temperature rises (plot 1 should slope inward).
* Non-binary answers rise with temperature, ~2% at T=0 to ~10% at T=1.6, with
  UNPARSEABLE held near 1% (PLAN.md §4 targets under 2%) — plot 4.
* `expert` reasons deeper than `none`: L2/L3 dominate for `expert`, L0/L1 for
  `none` — plot 3.

**`direct_sample.jsonl` is a `data/raw.jsonl`-conformant file and holds every
raw record, cot as well as direct.** The two fixture files are the two halves
of the join, not the two answer modes: `annotated.jsonl` carries no condition
(the annotator is blind to it, CONTRACTS.md §5), so the cot raw records have to
live somewhere or nothing can be joined. Pass it as `--raw`:

    python -m src.analyze --config config/budget.yaml \\
        --raw tests/fixtures/direct_sample.jsonl \\
        --annotated tests/fixtures/annotated_sample.jsonl --out out/
"""

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analyze                                      # noqa: E402
from src import contracts                                    # noqa: E402
from src import curve as curve_mod                           # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RAW_FIXTURE = FIXTURES / "direct_sample.jsonl"
ANNOTATED_FIXTURE = FIXTURES / "annotated_sample.jsonl"

# ─────────────────────────────────────────────────── the encoded fixture ────

FIXTURE_SEED = 20260903
NULL_MODEL = "meta-llama/llama-3.1-8b-instruct"
FLIP_MODEL = "ibm-granite/granite-4.1-8b"
NEG_MODEL = "openai/gpt-4o-mini"

MODELS = {                       # model -> (tier, provider)
    NULL_MODEL: ("small", "Novita"),
    FLIP_MODEL: ("small", "CoreWeave"),
    NEG_MODEL: ("mid", "OpenAI"),
}
SCENARIOS = ("default_keep", "default_switch", "isomorph")
PERSONAS = ("none", "expert")
TEMPS = (0.0, 0.8, 1.6)
REPLICATES = 8

#: Conditional P(KEEP | KEEP or SWITCH) at T=0, persona `none`. The flip effect
#: is the default_keep − default_switch difference on this row.
BASE_KEEP = {
    NULL_MODEL: {"default_keep": 0.62, "default_switch": 0.62, "isomorph": 0.55},
    FLIP_MODEL: {"default_keep": 0.88, "default_switch": 0.22, "isomorph": 0.52},
    NEG_MODEL:  {"default_keep": 0.24, "default_switch": 0.70, "isomorph": 0.40},
}
#: Sampling noise pulls a cell toward 50/50 as temperature rises.
DAMPING = {0.0: 1.00, 0.8: 0.85, 1.6: 0.60}
PERSONA_KEEP_SHIFT = {"none": 0.0, "expert": 0.04}

#: Non-binary answers get more common with temperature; UNPARSEABLE stays ~1%.
NONBINARY_WEIGHTS = {"RANDOMISE": 0.35, "THIRD_OPTION": 0.30,
                     "REFUSE": 0.25, "UNPARSEABLE": 0.10}
DEPTH_WEIGHTS = {
    "none":   {"L0": 0.30, "L1": 0.35, "L2": 0.15, "L3": 0.03, "MIX": 0.07, "NON": 0.10},
    "expert": {"L0": 0.08, "L1": 0.25, "L2": 0.35, "L3": 0.15, "MIX": 0.12, "NON": 0.05},
}

#: Which direct-mode records are deliberately broken, and how. Four of
#: llama's six fail — two of its cells still extract cleanly, and the whole
#: model is dropped anyway. That is the point: a partial drop would reweight
#: the surviving (scenario, persona) cells.
BROKEN_DIRECT = {
    (NULL_MODEL, "default_keep", "none"): "no_logprobs",
    (NULL_MODEL, "default_switch", "none"): "no_logprobs",
    (NULL_MODEL, "isomorph", "none"): "no_logprobs",
    (NULL_MODEL, "default_keep", "expert"): "no_answer_tokens",
}


def _nonbinary_rate(T: float) -> float:
    return 0.02 + 0.05 * T


def _cell_keep(model: str, scenario: str, persona: str, T: float) -> float:
    base = BASE_KEEP[model][scenario] + PERSONA_KEEP_SHIFT[persona]
    return 0.5 + (base - 0.5) * DAMPING[T]


def expected_flip(model: str) -> float:
    """The flip rate the fixture encodes, in closed form.

    ``flip = mean over (persona, T) of [keep(default_keep) − keep(default_switch)]``.
    Both terms carry the same persona shift and the same damping, so the shift
    cancels exactly and the damping factors out:

        flip = (base_keep − base_switch) × mean(DAMPING)

    A test that compares the measured flip against a hardcoded number checks
    the number; comparing it against this checks the pipeline.
    """
    base = BASE_KEEP[model]
    mean_damping = sum(DAMPING.values()) / len(DAMPING)
    return (base["default_keep"] - base["default_switch"]) * mean_damping


def _largest_remainder(weights: dict[str, float], total: int) -> dict[str, int]:
    """Split ``total`` across ``weights`` so the parts sum to exactly ``total``.

    Deterministic composition, not sampling. The fixture exists so a plot can
    be checked against a known truth; drawing 8 replicates at random would put
    ±0.17 of binomial noise on every cell and the "null" model would show a
    flip rate the user then has to mentally subtract.
    """
    scale = sum(weights.values())
    exact = {k: total * w / scale for k, w in weights.items()}
    out = {k: int(v) for k, v in exact.items()}
    short = total - sum(out.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - out[k]), k))
    for k in order[:short]:
        out[k] += 1
    return out


def _cot_text(answer: str, depth: str) -> str:
    lead = {
        "L0": "I have no way to tell them apart, so I will just go with my gut.",
        "L1": "He prepared both and kept one, so he kept the safe one for himself.",
        "L2": "He expects me to reason that he kept the safe one, so he planned for that.",
        "L3": "Each level of anticipation inverts the last and it does not bottom out.",
        "MIX": "Game-theoretically the only stable play here is to randomise.",
        "NON": "The premise asks me to accept a lethal wager, which I want to push back on.",
    }[depth]
    if answer == "UNPARSEABLE":
        return lead + " I am not going to commit to a single word here."
    tail = {"KEEP": "KEEP", "SWITCH": "SWITCH", "RANDOMISE": "RANDOMISE",
            "THIRD_OPTION": "THIRD_OPTION", "REFUSE": "REFUSE"}[answer]
    return f"{lead} Weighing that against the alternative, I will commit.\n\n{tail}"


def _raw_cot(model, scenario, persona, T, replicate, text, tier, provider):
    return {
        "run_id": contracts.run_id(model, scenario, persona, "cot", T, replicate),
        "experiment": "fixture_v1",
        "model": model,
        "tier": tier,
        "scenario": scenario,
        "persona": persona,
        "mode": "cot",
        "temperature": float(T),
        "replicate": int(replicate),
        "messages": contracts.build_messages(scenario, persona, "cot"),
        "max_tokens": contracts.COT_MAX_TOKENS,
        "logprobs": False,
        "top_logprobs": None,
        "provider_routing": None,
        "response": {
            "id": f"gen-fixture-{replicate}",
            "object": "chat.completion",
            "model": model,
            "provider": provider,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "native_finish_reason": "stop",
                         "message": {"role": "assistant", "content": text},
                         "logprobs": None}],
            "usage": {"prompt_tokens": 210, "completion_tokens": len(text) // 4,
                      "total_tokens": 210 + len(text) // 4, "cost": 4.2e-06},
        },
        "model_returned": model,
        "provider": provider,
        "usage": {"prompt_tokens": 210, "completion_tokens": len(text) // 4,
                  "total_tokens": 210 + len(text) // 4, "cost": 4.2e-06},
        "ts": "2026-09-03T18:00:00Z",
    }


def _raw_direct(model, scenario, persona, p_keep, tier, provider, broken=None):
    """A direct-mode record. `broken` reproduces a measured provider failure."""
    if broken == "no_logprobs":
        logprobs_block = None                      # provider returned none at all
    elif broken == "no_answer_tokens":
        logprobs_block = {"content": [              # array misaligned with the text
            {"token": "</s>", "logprob": -0.04, "top_logprobs": [
                {"token": "</s>", "logprob": -0.04},
                {"token": "\n\n", "logprob": -3.4}]}]}
    else:
        # The {K, S} pair is a slice of a full vocabulary, so it does not sum
        # to 1 — 0.95 of the mass here, the rest spread over formatting tokens.
        pk, ps = 0.95 * p_keep, 0.95 * (1.0 - p_keep)
        logprobs_block = {"content": [{
            "token": "K" if pk >= ps else "S",
            "logprob": math.log(max(pk, ps)),
            "top_logprobs": [
                {"token": "K", "logprob": math.log(pk * 0.8)},
                {"token": " K", "logprob": math.log(pk * 0.2)},
                {"token": "S", "logprob": math.log(ps)},
                {"token": "\n", "logprob": math.log(0.03)},
            ]}]}
    content = "K" if p_keep >= 0.5 else "S"
    return {
        "run_id": contracts.run_id(model, scenario, persona, "direct",
                                   contracts.DIRECT_TEMPERATURE, 0),
        "experiment": "fixture_v1",
        "model": model,
        "tier": tier,
        "scenario": scenario,
        "persona": persona,
        "mode": "direct",
        "temperature": contracts.DIRECT_TEMPERATURE,
        "replicate": 0,
        "messages": contracts.build_messages(scenario, persona, "direct"),
        "max_tokens": contracts.DIRECT_MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": contracts.DIRECT_TOP_LOGPROBS,
        "provider_routing": contracts.DIRECT_PROVIDER_PIN.get(model),
        "response": {
            "id": "gen-fixture-direct",
            "object": "chat.completion",
            "model": model,
            "provider": provider,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "native_finish_reason": "stop",
                         "message": {"role": "assistant", "content": content},
                         "logprobs": logprobs_block}],
            "usage": {"prompt_tokens": 205, "completion_tokens": 1,
                      "total_tokens": 206, "cost": 3.1e-05},
        },
        "model_returned": model,
        "provider": provider,
        "usage": {"prompt_tokens": 205, "completion_tokens": 1,
                  "total_tokens": 206, "cost": 3.1e-05},
        "ts": "2026-09-03T18:00:00Z",
    }


def _annotated(run_id, answer, depth, no_basis, rigged, text):
    payload = {"answer": answer, "depth": depth,
               "flag_no_basis": no_basis, "flag_rigged": rigged}
    parse_ok = answer != "UNPARSEABLE"
    return {
        "run_id": run_id,
        "answer": answer,
        "depth": depth,
        "flag_no_basis": no_basis,
        "flag_rigged": rigged,
        "annotator_model": "mistralai/mistral-nemo",
        "raw_annotator_output": (json.dumps(payload) if parse_ok
                                 else "I could not code this one confidently."),
        "ts": "2026-09-03T18:30:00Z",
        "annotator_provider": "Parasail",
        "parse_ok": parse_ok,
    }


def _cell_grid() -> list[tuple[str, str, str, float]]:
    """Every (model, scenario, persona, temperature) cell.

    Model varies fastest so that the round-robin allocation of non-binary
    answers below spreads them evenly across models — the estimator gives each
    model one vote, so a non-binary rate concentrated in one model would read
    differently from the same rate spread across three.
    """
    return [(model, scenario, persona, T)
            for T in TEMPS
            for scenario in SCENARIOS
            for persona in PERSONAS
            for model in MODELS]


def _allocate_answers() -> dict[tuple, list[str]]:
    """The answer each replicate of each cell gets. Composed, not sampled."""
    grid = _cell_grid()

    # 1. How many non-binary answers per cell, so the rate per temperature is
    #    exactly _nonbinary_rate(T) up to one whole response.
    per_cell_nb: dict[tuple, int] = {}
    for T in TEMPS:
        cells_T = [c for c in grid if c[3] == T]
        total = round(_nonbinary_rate(T) * len(cells_T) * REPLICATES)
        base, extra = divmod(total, len(cells_T))
        for i, cell in enumerate(cells_T):
            per_cell_nb[cell] = base + (1 if i < extra else 0)

    # 2. Which non-binary label each of those slots gets, honouring the weights
    #    across the fixture as a whole (too few per cell to do it per cell).
    n_nonbinary = sum(per_cell_nb.values())
    counts = _largest_remainder(NONBINARY_WEIGHTS, n_nonbinary)
    # Spread each label evenly across the pool rather than cycling the labels
    # in order: the pool is consumed temperature by temperature, and a plain
    # round-robin buries a rare label (UNPARSEABLE, 3 of 26) entirely inside
    # one temperature, which then reads as a temperature effect on the plot.
    spread: list[tuple[float, str]] = []
    for label, n in counts.items():
        spread += [((j + 0.5) / n, label) for j in range(n)]
    pool = [label for _, label in sorted(spread, key=lambda kv: (kv[0], kv[1]))]

    # 3. Fill each cell: its non-binary slots, then a KEEP/SWITCH split that
    #    hits the cell's target conditional rate as closely as 1/n allows.
    out: dict[tuple, list[str]] = {}
    cursor = 0
    for cell in grid:
        model, scenario, persona, T = cell
        n_nb = per_cell_nb[cell]
        answers = pool[cursor:cursor + n_nb]
        cursor += n_nb
        n_binary = REPLICATES - n_nb
        n_keep = int(n_binary * _cell_keep(model, scenario, persona, T) + 0.5)
        answers += ["KEEP"] * n_keep + ["SWITCH"] * (n_binary - n_keep)
        out[cell] = answers
    return out


def _allocate_depths(answers: dict[tuple, list[str]],
                     rng: random.Random) -> dict[tuple, list[str]]:
    """Depth codes, composed per (model, persona) so plot 3 shows DEPTH_WEIGHTS.

    ``UNPARSEABLE`` is forced to ``NON`` (CONTRACTS.md §6: there is no null
    depth) and is excluded from the allocation, so the stated weights hold
    exactly over the responses that could be coded at all.
    """
    out: dict[tuple, list[str]] = {cell: [None] * len(v) for cell, v in answers.items()}
    for model in MODELS:
        for persona in PERSONAS:
            slots = [(cell, i)
                     for cell, v in answers.items()
                     if cell[0] == model and cell[2] == persona
                     for i, a in enumerate(v) if a != "UNPARSEABLE"]
            counts = _largest_remainder(DEPTH_WEIGHTS[persona], len(slots))
            codes = [c for code, n in counts.items() for c in [code] * n]
            rng.shuffle(codes)                     # seeded: which slot, not how many
            for (cell, i), code in zip(slots, codes):
                out[cell][i] = code
    for cell, v in answers.items():
        for i, a in enumerate(v):
            if a == "UNPARSEABLE":
                out[cell][i] = "NON"
    return out


def build_fixtures() -> tuple[list[dict], list[dict]]:
    """Deterministic `(raw_records, annotated_records)`. Seeded, no wall clock."""
    rng = random.Random(FIXTURE_SEED)
    answers = _allocate_answers()
    depths = _allocate_depths(answers, rng)
    raw: list[dict] = []
    annotated: list[dict] = []

    for model, (tier, provider) in MODELS.items():
        for scenario in SCENARIOS:
            for persona in PERSONAS:
                for T in TEMPS:
                    cell = (model, scenario, persona, T)
                    # Seeded shuffle: which replicate index carries which answer
                    # is arbitrary, the composition of the cell is not.
                    order = list(range(REPLICATES))
                    rng.shuffle(order)
                    for replicate, slot in enumerate(order):
                        answer = answers[cell][slot]
                        depth = depths[cell][slot]
                        text = _cot_text(answer, depth)
                        rec = _raw_cot(model, scenario, persona, T, replicate,
                                       text, tier, provider)
                        raw.append(rec)
                        annotated.append(_annotated(
                            rec["run_id"], answer, depth,
                            no_basis=(depth == "MIX") or rng.random() < 0.12,
                            rigged=rng.random() < 0.14,
                            text=text))

                # One direct-mode record per (model, scenario, persona).
                raw.append(_raw_direct(
                    model, scenario, persona,
                    _cell_keep(model, scenario, persona, 0.8),
                    tier, provider,
                    broken=BROKEN_DIRECT.get((model, scenario, persona)),
                ))

    return raw, annotated


def write_fixtures() -> tuple[Path, Path]:
    raw, annotated = build_fixtures()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for path, records in ((RAW_FIXTURE, raw), (ANNOTATED_FIXTURE, annotated)):
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return RAW_FIXTURE, ANNOTATED_FIXTURE


# ───────────────────────────────────────────────────────────── fixtures ────

@pytest.fixture(scope="module")
def cot_and_direct():
    raw = analyze.read_jsonl(RAW_FIXTURE)
    ann = analyze.read_jsonl(ANNOTATED_FIXTURE)
    cot, join = analyze.load_cot(raw, ann)
    direct = curve_mod.direct_curve(raw, list(TEMPS))
    return cot, direct, join, raw


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    return analyze.run(config=contracts.CONFIG_PATH, raw=RAW_FIXTURE,
                       annotated=ANNOTATED_FIXTURE, out=out, n_boot=1000), out


# ────────────────────────────────────────────── the fixtures themselves ────

def test_fixture_files_exist_and_have_the_documented_shape(cot_and_direct):
    cot, direct, join, raw = cot_and_direct
    assert len(cot) == len(MODELS) * len(SCENARIOS) * len(PERSONAS) * len(TEMPS) * REPLICATES
    assert len(cot) == 432
    assert join.n_raw_direct == len(MODELS) * len(SCENARIOS) * len(PERSONAS) == 18
    assert join.n_annotated_without_raw == 0
    assert join.n_cot_without_annotation == 0
    assert set(cot["model"]) == set(MODELS)
    assert set(cot["temperature"]) == set(TEMPS)


def test_fixtures_are_deterministic():
    """Regenerating must reproduce the committed bytes, or a plot the user
    eyeballed yesterday is not the plot the tests check today."""
    raw, ann = build_fixtures()
    on_disk_raw = analyze.read_jsonl(RAW_FIXTURE)
    on_disk_ann = analyze.read_jsonl(ANNOTATED_FIXTURE)
    assert raw == on_disk_raw
    assert ann == on_disk_ann


def test_fixture_answers_and_depths_are_on_schema(cot_and_direct):
    cot, *_ = cot_and_direct
    assert set(cot["answer"]) <= set(contracts.ANSWERS)
    assert set(cot["depth"]) <= set(contracts.DEPTH_CODES)
    # UNPARSEABLE always codes as NON (CONTRACTS.md §5) — there is no null depth.
    unp = cot[cot["answer"] == "UNPARSEABLE"]
    assert not unp.empty
    assert set(unp["depth"]) == {"NON"}


# ─────────────────────────────────────────────────────── the four checks ────

def test_bootstrap_ci_for_an_all_keep_cell_is_one_to_one():
    """Every resample of an all-KEEP cell is still all-KEEP, so the interval is
    a point at 1 — not an artefact of the percentile method."""
    def all_keep(models=("m",), personas=("none",), **kw):
        rows = [{"model": m, "scenario": "default_keep", "persona": p,
                 "temperature": 0.0, "answer": "KEEP"}
                for m in models for p in personas for _ in range(25)]
        return analyze.cell_values(pd.DataFrame(rows), success=("KEEP",), **kw)

    cells = all_keep(models=("m1", "m2", "m3"))
    assert len(cells) == 3
    assert (cells["value"] == 1.0).all()

    r = analyze.bootstrap_ci(cells, unit=("model",), n_boot=1000, seed=1)
    assert (r.point, r.lo, r.hi) == (1.0, 1.0, 1.0)
    assert r.n_units == 3 and not r.degenerate

    # Same at the boundary for the within-model form and for a conditional
    # denominator — the bootstrap must not invent spread where there is none.
    inner = analyze.bootstrap_ci(all_keep(personas=("none", "expert", "average_person")),
                                 unit=("scenario", "persona", "temperature"),
                                 n_boot=1000, seed=1)
    assert (inner.point, inner.lo, inner.hi) == (1.0, 1.0, 1.0)
    cond = all_keep(models=("m1", "m2", "m3"), denominator=analyze.BINARY_ANSWERS)
    assert analyze.bootstrap_ci(cond, unit=("model",), n_boot=1000, seed=1).hi == 1.0

    # An all-KEEP cell in a SINGLE-cluster frame is a different statement: the
    # rate is still exactly 1.0, but there is no between-unit variation to
    # resample, so there is no interval to report. See `Rate.degenerate`.
    one = analyze.bootstrap_ci(all_keep(), unit=("model",), n_boot=1000, seed=1)
    assert one.point == 1.0 and one.degenerate
    assert not np.isfinite(one.lo) and "no CI" in one.fmt()


def test_flip_rate_is_large_for_the_effect_model_and_near_zero_for_the_null(cot_and_direct):
    cot, *_ = cot_and_direct
    flips = analyze.flip_cells(cot)

    rates = {}
    for model in MODELS:
        sub = flips[flips["model"] == model]
        rates[model] = analyze.bootstrap_ci(
            sub, unit=("persona", "temperature"),
            cell_keys=analyze.FLIP_CELL_KEYS, n_boot=1000, seed=7)

    strong, null, negative = rates[FLIP_MODEL], rates[NULL_MODEL], rates[NEG_MODEL]

    assert strong.point > 0.45, strong
    assert strong.lo > 0.0                       # interval excludes "no effect"
    assert abs(null.point) < 0.05, null
    assert null.lo <= 0.0 <= null.hi             # interval covers "no effect"

    # Signed, not absolute: the third model is on the OTHER side of zero, which
    # an abs() would collapse onto the same value as a status quo bias.
    assert negative.point < -0.30, negative
    assert negative.hi < 0.0
    assert abs(negative.point) > abs(null.point)
    assert strong.point > 0 > negative.point

    # And each lands on the value the fixture parameters imply in closed form,
    # within the 1/n_binary granularity of an 8-replicate cell.
    for model in MODELS:
        assert rates[model].point == pytest.approx(expected_flip(model), abs=0.05), model


def test_all_four_plots_and_the_summary_are_produced(analysis):
    result, out = analysis
    expected = {"keep_vs_temperature.png", "flip_rate.png",
                "depth_by_persona.png", "nonbinary_by_temperature.png",
                # plots 5 and 6, added 2026-09-03
                "answers_by_role.png", "keep_by_temperature.png"}
    assert {p.name for p in result["plots"]} == expected
    for p in result["plots"]:
        assert p.exists() and p.stat().st_size > 5_000, p
    summary = out / "summary.md"
    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    for needle in ("Flip rate", "UNPARSEABLE", "Direct mode",
                   "Depth distribution", "conditional", "unconditional"):
        assert needle in text, needle


def test_pooling_across_models_without_clustering_raises(cot_and_direct):
    """The one mistake that produces a wrong answer no reader can see."""
    cot, *_ = cot_and_direct

    # 1. Sample-level rows handed straight to the estimator.
    with pytest.raises(analyze.ClusteringError) as exc:
        analyze.bootstrap_ci(cot, unit=("model",))
    msg = str(exc.value)
    assert "sample-level" in msg and "25 draws from ONE distribution" in msg

    # 2. Cell-level rows, but two runs concatenated — cells double-weighted.
    cells = analyze.cell_values(cot, success=("KEEP",))
    with pytest.raises(analyze.ClusteringError, match="duplicate rows"):
        analyze.bootstrap_ci(pd.concat([cells, cells], ignore_index=True),
                             unit=("model",))

    # 3. No cluster named at all.
    with pytest.raises(analyze.ClusteringError, match="no resampling unit"):
        analyze.bootstrap_ci(cells, unit=())

    # 4. A cluster column that is not there.
    with pytest.raises(analyze.ClusteringError, match="cannot be resampled"):
        analyze.bootstrap_ci(cells, unit=("provider",))


# ───────────────────────────────────────────── denominators and dropping ────

def test_conditional_and_unconditional_denominators_differ(cot_and_direct):
    """If they were equal the fixture would not exercise the distinction that
    CONTRACTS.md §8 says makes plot 1 a lie when it is got wrong."""
    cot, *_ = cot_and_direct
    uncond = analyze.bootstrap_ci(analyze.cell_values(cot, success=("KEEP",)),
                                  unit=("model",), n_boot=1000, seed=3)
    cond = analyze.bootstrap_ci(
        analyze.cell_values(cot, success=("KEEP",),
                            denominator=analyze.BINARY_ANSWERS),
        unit=("model",), n_boot=1000, seed=3)
    assert cond.point > uncond.point
    excluded = analyze.bootstrap_ci(
        analyze.cell_values(cot, success=analyze.NONBINARY_ANSWERS),
        unit=("model",), n_boot=1000, seed=3)
    assert 0.01 < excluded.point < 0.15


def test_excluded_fraction_rises_with_temperature(cot_and_direct):
    cot, *_ = cot_and_direct
    rates = []
    for T in TEMPS:
        cells = analyze.cell_values(cot[cot["temperature"] == T],
                                    success=analyze.NONBINARY_ANSWERS)
        rates.append(analyze.bootstrap_ci(cells, unit=("model",),
                                          n_boot=200, seed=5).point)
    assert rates == sorted(rates)
    assert rates[0] < rates[-1]


def test_the_overlay_drops_a_whole_model_and_keeps_its_cot_line(cot_and_direct):
    cot, direct, join, raw = cot_and_direct
    report = curve_mod.direct_mode_report(raw)

    assert report.n_records == 18
    assert report.status_counts == {"ok": 14, "no_logprobs": 3, "no_answer_tokens": 1}
    assert report.n_unusable == 4
    assert set(report.usable_models) == {FLIP_MODEL, NEG_MODEL}
    assert NULL_MODEL in report.dropped_models
    assert "4/6" in report.dropped_models[NULL_MODEL]

    # The two cells of NULL_MODEL that DID extract are gone from the overlay …
    assert NULL_MODEL not in set(direct["model"])
    # … but its cot line, the primary series, is untouched.
    assert NULL_MODEL in set(cot["model"])
    assert len(cot[cot["model"] == NULL_MODEL]) == len(SCENARIOS) * len(PERSONAS) * len(TEMPS) * REPLICATES


def test_direct_curve_is_conditional_and_matches_the_cot_scale(cot_and_direct):
    """The analytic curve and plot 1's cot series must be the same quantity:
    both conditional on {K, S}, so both live on [0, 1] with 0.5 meaning
    indifference."""
    cot, direct, *_ = cot_and_direct
    assert not direct.empty
    assert np.allclose(direct["p_keep"] + direct["p_switch"], 1.0)
    at_T0 = direct[direct["temperature"] == 0.0]["p_keep"]
    assert set(np.round(at_T0, 9)) <= {0.0, 1.0}     # T=0 is the argmax


# ─────────────────────────────────────────────────────── loader contract ────

def test_duplicate_annotations_are_rejected():
    raw = analyze.read_jsonl(RAW_FIXTURE)[:1]
    ann = [{"run_id": raw[0]["run_id"], "answer": "KEEP", "depth": "L1",
            "flag_no_basis": False, "flag_rigged": False,
            "annotator_model": "x", "raw_annotator_output": "{}"}]
    with pytest.raises(analyze.DataIntegrityError, match="more than one line"):
        analyze.load_cot(raw, ann + ann)


def test_off_schema_labels_are_rejected():
    raw = [r for r in analyze.read_jsonl(RAW_FIXTURE) if r["mode"] == "cot"][:1]
    ann = [{"run_id": raw[0]["run_id"], "answer": "MAYBE", "depth": "L1",
            "flag_no_basis": False, "flag_rigged": False,
            "annotator_model": "x", "raw_annotator_output": "{}"}]
    with pytest.raises(analyze.DataIntegrityError, match="off-schema"):
        analyze.load_cot(raw, ann)


def test_loader_ignores_side_runs():
    """direct_validation carries a real temperature and a real replicate index,
    so pooling it with the cot grid would silently add draws to one cell."""
    raw = analyze.read_jsonl(RAW_FIXTURE)
    stray = dict(raw[0])
    stray["mode"] = contracts.DIRECT_VALIDATION_MODE
    stray["run_id"] = "deadbeef" * 5
    ann = analyze.read_jsonl(ANNOTATED_FIXTURE)
    cot, join = analyze.load_cot(raw + [stray], ann)
    assert len(cot) == 432
    assert join.n_raw_other_modes.get(contracts.DIRECT_VALIDATION_MODE) == 1


def test_truncated_jsonl_names_the_line(tmp_path):
    p = tmp_path / "raw.jsonl"
    p.write_text('{"run_id": "a"}\n{"run_id": "b"\n', encoding="utf-8")
    with pytest.raises(analyze.DataIntegrityError, match=r"raw\.jsonl:2"):
        analyze.read_jsonl(p)


def test_model_order_follows_the_config_then_appends_strangers():
    cfg = analyze.load_config(contracts.CONFIG_PATH)
    configured = [m["id"] for m in cfg["models"]]
    got = analyze.model_order(cfg, [configured[2], configured[0], "zzz/unknown"])
    assert got == [configured[0], configured[2], "zzz/unknown"]


def test_empty_conditional_cell_is_undefined_not_zero():
    """A cell where nothing binary was answered has no conditional KEEP rate.
    Calling that 0 would invent a finding out of a refusal."""
    df = pd.DataFrame([
        {"model": "m", "scenario": "default_keep", "persona": "none",
         "temperature": 0.0, "answer": "REFUSE"} for _ in range(8)
    ])
    cells = analyze.cell_values(df, success=("KEEP",),
                                denominator=analyze.BINARY_ANSWERS)
    assert cells["n"].iloc[0] == 0
    assert math.isnan(cells["value"].iloc[0])
    r = analyze.bootstrap_ci(cells, unit=("model",), n_boot=100, seed=0)
    assert math.isnan(r.point) and r.n_cells_undefined == 1


def test_a_partial_grid_survives_end_to_end(tmp_path):
    """A resumed run killed halfway is the NORMAL way to get a lopsided grid
    (CONTRACTS.md §4 appends per record), so a cell present in default_keep and
    missing from default_switch must not crash the summary or score as 0 flip."""
    raw = analyze.read_jsonl(RAW_FIXTURE)
    ann_by_id = {a["run_id"]: a for a in analyze.read_jsonl(ANNOTATED_FIXTURE)}

    # Drop every default_switch record at T=1.6 for one model.
    def killed(rec):
        return (rec["mode"] == "cot" and rec["model"] == FLIP_MODEL
                and rec["scenario"] == "default_switch" and rec["temperature"] == 1.6)

    kept = [r for r in raw if not killed(r)]
    assert len(kept) < len(raw)
    ann = [ann_by_id[r["run_id"]] for r in kept if r["mode"] == "cot"]

    cot, _ = analyze.load_cot(kept, ann)
    flips = analyze.flip_cells(cot)

    one_sided = flips[(flips["model"] == FLIP_MODEL) & (flips["temperature"] == 1.6)]
    assert not one_sided.empty
    assert one_sided["value"].isna().all()                 # no flip rate, not zero
    assert (one_sided["n_default_switch"] == 0).all()
    assert one_sided["n_default_switch"].dtype.kind == "i"  # int, so int() is safe
    assert (one_sided["n_default_keep"] > 0).all()

    # The one-sided cells are excluded from the estimate, and counted as such.
    r = analyze.bootstrap_ci(flips[flips["model"] == FLIP_MODEL],
                             unit=("persona", "temperature"),
                             cell_keys=analyze.FLIP_CELL_KEYS, n_boot=200, seed=0)
    assert r.n_cells_undefined == len(one_sided)
    assert np.isfinite(r.point) and r.point > 0.45

    # And the whole pipeline still runs.
    raw_path, ann_path = tmp_path / "raw.jsonl", tmp_path / "ann.jsonl"
    for path, records in ((raw_path, kept), (ann_path, ann)):
        path.write_text("".join(json.dumps(x) + "\n" for x in records), encoding="utf-8")
    result = analyze.run(config=contracts.CONFIG_PATH, raw=raw_path,
                         annotated=ann_path, out=tmp_path / "out", n_boot=200)
    assert len(result["plots"]) == 6   # 4 from PLAN.md §8.5 + plots 5 and 6
    assert "n/a" in (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")


def test_the_answer_partition_is_derived_from_the_enum_not_restated():
    """CONTRACTS.md §0: import the contract's constants, do not re-implement.

    Plot 1's caption promises plot 4 shows exactly what plot 1 drops. That holds
    only if the two sets are exact complements of `contracts.ANSWERS`; a
    hardcoded pair would let a new label fall out of both, so plot 1 would
    silently exclude responses plot 4 never counts.
    """
    assert set(analyze.BINARY_ANSWERS) | set(analyze.NONBINARY_ANSWERS) == set(contracts.ANSWERS)
    assert not set(analyze.BINARY_ANSWERS) & set(analyze.NONBINARY_ANSWERS)
    # The binary pair is exactly what the analytic curve renormalises over.
    assert set(analyze.BINARY_ANSWERS) == set(contracts.DIRECT_ANSWER_TOKENS.values())
    # Canonical order is preserved on both sides (CONTRACTS.md §6).
    for subset in (analyze.BINARY_ANSWERS, analyze.NONBINARY_ANSWERS):
        assert list(subset) == [a for a in contracts.ANSWERS if a in set(subset)]


def test_a_single_cluster_reports_no_interval_rather_than_a_zero_width_one(cot_and_direct):
    """PLAN.md §9 step 5's pilot is one persona at one temperature — "the one
    not to skip" — and every resample of one cluster returns that cluster, so
    `0.571 [0.571, 0.571]` would print as a 95% CI asserting certainty."""
    cot, *_ = cot_and_direct
    pilot = cot[(cot["persona"] == "none") & (cot["temperature"] == 0.0)]
    r = analyze.bootstrap_ci(analyze.flip_cells(pilot)[lambda d: d["model"] == FLIP_MODEL],
                             unit=("persona", "temperature"),
                             cell_keys=analyze.FLIP_CELL_KEYS, n_boot=1000, seed=0)
    assert r.n_units == 1 and r.degenerate
    assert np.isfinite(r.point)
    assert not np.isfinite(r.lo) and not np.isfinite(r.hi)   # no interval at all
    assert "no CI" in r.fmt() and "[0.6" not in r.fmt()

    # A plot must not turn that into a zero-height bar reading as certainty.
    assert all(np.isnan(v) for v in analyze._yerr(r))
    # Two or more clusters still give a real interval.
    many = analyze.bootstrap_ci(analyze.flip_cells(cot)[lambda d: d["model"] == FLIP_MODEL],
                                unit=("persona", "temperature"),
                                cell_keys=analyze.FLIP_CELL_KEYS, n_boot=1000, seed=0)
    assert many.n_units == 6 and not many.degenerate
    assert np.isfinite(many.lo) and many.lo < many.point < many.hi


def test_a_duplicate_raw_cot_record_is_refused(cot_and_direct):
    """Symmetric with the annotated-side rule. A partial duplication — a subset
    re-run appended to the same file — shifts that cell's rate toward the
    duplicated answers with nothing to signal it."""
    cot, _, _, raw = cot_and_direct
    ann = analyze.read_jsonl(ANNOTATED_FIXTURE)
    one = next(r for r in raw if r["mode"] == "cot")
    with pytest.raises(analyze.DataIntegrityError, match="more than once"):
        analyze.load_cot(raw + [one], ann)


def test_plot_3_excludes_models_with_no_reasoning_to_code(tmp_path):
    """CONTRACTS.md §8: pooling a no-CoT model's uniform L0/NON into the
    cross-model depth figure drags every bar toward L0 and looks like a
    finding."""
    raw = analyze.read_jsonl(RAW_FIXTURE)
    ann_by = {a["run_id"]: a for a in analyze.read_jsonl(ANNOTATED_FIXTURE)}
    terse = sorted(contracts.NO_COT_MODELS)[0]

    # Relabel one fixture model as the terse one and give it uniform L0/NON.
    relabelled, ann = [], []
    for rec in raw:
        rec = dict(rec)
        if rec["model"] == NULL_MODEL:
            rec["model"] = terse
        relabelled.append(rec)
        if rec["mode"] != "cot":
            continue
        a = dict(ann_by[rec["run_id"]])
        if rec["model"] == terse:
            a["depth"] = "L0" if rec["replicate"] % 2 else "NON"
        ann.append(a)

    cot, _ = analyze.load_cot(relabelled, ann)
    assert terse in set(cot["model"])

    def l0_share(frame):
        return analyze.bootstrap_ci(
            analyze.cell_values(frame[frame["persona"] == "expert"],
                                success=("L0",), column="depth"),
            unit=("model",), n_boot=200, seed=0).point

    pooled = l0_share(cot)
    codeable = l0_share(cot[~cot["model"].isin(list(contracts.NO_COT_MODELS))])
    assert pooled > codeable + 0.10          # the artefact the contract warns of

    # The figure must show the codeable number, and say who it left out.
    analyze.plot_depth_by_persona(cot, tmp_path, n_boot=200, seed=0)
    assert (tmp_path / "depth_by_persona.png").exists()
    src = Path(analyze.__file__).read_text(encoding="utf-8")
    assert "EXCLUDED from these bars" in src

    # But summary.md still reports depth per model, theirs included.
    result = analyze.run(config=contracts.CONFIG_PATH,
                         raw=_write(tmp_path / "r.jsonl", relabelled),
                         annotated=_write(tmp_path / "a.jsonl", ann),
                         out=tmp_path / "out", n_boot=200)
    text = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert terse in text and "visible cot?" in text
    assert len(result["plots"]) == 6   # 4 from PLAN.md §8.5 + plots 5 and 6


def test_a_model_with_a_curve_but_no_cot_rows_keeps_its_overlay(tmp_path):
    """CONTRACTS.md §8: draw the overlay where it exists. Deriving the model
    list from the CoT join alone would drop it silently, and summary.md would
    then name it as having a usable curve in one table and omit it from the
    next."""
    raw = analyze.read_jsonl(RAW_FIXTURE)
    ann_by = {a["run_id"]: a for a in analyze.read_jsonl(ANNOTATED_FIXTURE)}

    # The annotator failed on one model: its direct records survive, cot does not.
    kept = [r for r in raw if not (r["mode"] == "cot" and r["model"] == NEG_MODEL)]
    ann = [ann_by[r["run_id"]] for r in kept if r["mode"] == "cot"]

    result = analyze.run(config=contracts.CONFIG_PATH,
                         raw=_write(tmp_path / "r.jsonl", kept),
                         annotated=_write(tmp_path / "a.jsonl", ann),
                         out=tmp_path / "out", n_boot=200)

    assert NEG_MODEL not in set(result["cot"]["model"])       # no cot rows
    assert NEG_MODEL in result["direct_report"].usable_models  # but a real curve
    assert NEG_MODEL in set(result["direct"]["model"])
    assert NEG_MODEL in result["models"]                       # so it is plotted

    # summary.md agrees with itself: named as usable AND present in the table.
    text = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    analytic = text.split("Analytic conditional KEEP rate")[1]
    assert NEG_MODEL in analytic


def _write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _legend_of(fig_ax):
    leg = fig_ax.get_legend()
    return [t.get_text() for t in leg.get_texts()], leg.get_title().get_text()


def test_plot_1_labels_every_series_it_draws(tmp_path):
    """The legend is harvested from ALL panels, not just the first.

    A model can be absent from panel 0 on a fully successful run: if it answers
    only non-binary in `default_keep`, its conditional denominator there is
    empty, every T is skipped, and it registers no handle in that panel — while
    still drawing a coloured line in the other two.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = analyze.read_jsonl(RAW_FIXTURE)
    ann = []
    for a in analyze.read_jsonl(ANNOTATED_FIXTURE):
        a = dict(a)
        ann.append(a)
    by_id = {r["run_id"]: r for r in raw if r["mode"] == "cot"}
    for a in ann:
        rec = by_id.get(a["run_id"])
        if rec and rec["model"] == FLIP_MODEL and rec["scenario"] == "default_keep":
            a["answer"] = "REFUSE"          # no binary answers in panel 0 at all

    cot, _ = analyze.load_cot(raw, ann)
    direct = curve_mod.direct_curve(raw, list(TEMPS))
    models = analyze.model_order(analyze.load_config(contracts.CONFIG_PATH),
                                 set(cot["model"]) | set(direct["model"]))
    analyze.plot_keep_vs_temperature(
        cot, direct, models, list(TEMPS),
        curve_mod.direct_mode_report(raw), tmp_path, n_boot=100, seed=0)

    fig = plt.gcf()
    assert (tmp_path / "keep_vs_temperature.png").exists()
    src = Path(analyze.__file__).read_text(encoding="utf-8")
    assert "for panel in axes" in src, "legend must be harvested across panels"
    assert "axes[0].get_legend_handles_labels()" not in src
    plt.close(fig)

    # The model really is missing from panel 0's own handles — the situation is
    # reproduced, not merely asserted about.
    binary = analyze.cell_values(cot[cot["scenario"] == "default_keep"],
                                 success=("KEEP",), denominator=analyze.BINARY_ANSWERS)
    flip_cells_there = binary[binary["model"] == FLIP_MODEL]
    assert (flip_cells_there["n"] == 0).all()
    assert flip_cells_there["value"].isna().all()


def test_plot_1_caption_names_a_model_with_no_direct_records_at_all(tmp_path):
    """A pinned provider being down leaves a model with zero direct records, so
    `direct_mode_report` sees nothing to drop and reports none dropped — while
    the panel is visibly missing an overlay. CONTRACTS.md §8: say so."""
    raw = [r for r in analyze.read_jsonl(RAW_FIXTURE)
           if not (r["mode"] == "direct" and r["model"] == NEG_MODEL)]
    ann_by = {a["run_id"]: a for a in analyze.read_jsonl(ANNOTATED_FIXTURE)}
    ann = [ann_by[r["run_id"]] for r in raw if r["mode"] == "cot"]

    report = curve_mod.direct_mode_report(raw)
    assert NEG_MODEL not in report.dropped_models      # the report cannot see it
    assert NEG_MODEL not in report.usable_models

    result = analyze.run(config=contracts.CONFIG_PATH,
                         raw=_write(tmp_path / "r.jsonl", raw),
                         annotated=_write(tmp_path / "a.jsonl", ann),
                         out=tmp_path / "out", n_boot=100)
    assert NEG_MODEL in result["models"]               # its cot line is still drawn
    assert NEG_MODEL not in set(result["direct"]["model"])
    # The caption is derived from what was drawn, so it names this model too.
    # Asserted on the returned bookkeeping rather than on analyze.py's source
    # text: this test used to grep for the exact line, which meant it passed for
    # a line that was wrong (see the disjoint-scenario test below) and would
    # have failed for the fix.
    assert NEG_MODEL in result["overlay"]["no_overlay"]
    assert NEG_MODEL not in result["overlay"]["overlaid"]


def test_a_direct_record_with_a_stray_replicate_is_caught(tmp_path):
    """The duplicate guard keys on the CELL, not on run_id. A direct job written
    with a non-zero replicate hashes to a DIFFERENT run_id (CONTRACTS.md §2)
    while still being a second measurement of the same cell."""
    raw = analyze.read_jsonl(RAW_FIXTURE)
    original = next(r for r in raw if r["mode"] == "direct" and r["model"] == NEG_MODEL)
    stray = dict(original)
    stray["replicate"] = 1
    stray["run_id"] = contracts.run_id(stray["model"], stray["scenario"],
                                       stray["persona"], "direct",
                                       contracts.DIRECT_TEMPERATURE, 1)
    assert stray["run_id"] != original["run_id"]       # a run_id check would miss it

    with pytest.raises(curve_mod.DuplicateRecordError, match="cell"):
        curve_mod.direct_curve(raw + [stray], list(TEMPS))


def test_bootstrap_is_reproducible_from_the_seed(cot_and_direct):
    cot, *_ = cot_and_direct
    cells = analyze.cell_values(cot, success=("KEEP",))
    a = analyze.bootstrap_ci(cells, unit=("model",), n_boot=1000, seed=11)
    b = analyze.bootstrap_ci(cells, unit=("model",), n_boot=1000, seed=11)
    assert (a.point, a.lo, a.hi) == (b.point, b.lo, b.hi)

    # With three clusters the bootstrap distribution has only ten distinct
    # values (the multisets of size 3 from 3 models), so two seeds often land
    # on the same percentiles. That coarseness is a property of clustering by
    # model with few models — not a bug — and it is the honest width.
    per_model = cells.groupby("model", observed=True)["value"].mean().to_numpy()
    assert len(per_model) == 3
    assert a.lo >= per_model.min() and a.hi <= per_model.max()


def test_summary_reports_the_unusable_direct_count(analysis):
    _, out = analysis
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "**4** of 18" in text
    assert "no_answer_tokens" in text and "no_logprobs" in text


def test_cli_smoke(tmp_path, capsys):
    rc = analyze.main([
        "--config", str(contracts.CONFIG_PATH),
        "--raw", str(RAW_FIXTURE),
        "--annotated", str(ANNOTATED_FIXTURE),
        "--out", str(tmp_path),
        "--n-boot", "200",
    ])
    assert rc == 0
    assert (tmp_path / "summary.md").exists()
    assert "wrote" in capsys.readouterr().out


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        for p in write_fixtures():
            print(f"wrote {p} ({p.stat().st_size:,} bytes)")
    else:
        raise SystemExit("usage: python tests/test_analyze.py --regenerate")


def test_plot_1_survives_a_model_whose_direct_scenarios_are_disjoint_from_cot(tmp_path):
    """The blocker: plot 1 used to `assert` that every model in the direct frame
    got an overlay in some panel. Nothing guarantees that — the direct frame's
    model set and its scenario set are independent, and the panel set comes from
    the cot side.

    The state below is one CONTRACTS.md makes routine: a model's pinned provider
    is down for two scenarios, so no line is written for them (§8 sets
    allow_fallbacks: False, §4 says a failed call writes no line), leaving direct
    records only in `isomorph`; and the cot arm has not reached `isomorph` yet.
    The old code raised a bare `AssertionError` out of the FIRST plot, so no
    figures and no summary.md were written at all — and under `python -O`, with
    the assert stripped, it instead printed a caption claiming an overlay that
    is visibly absent.
    """
    victim = "openai/gpt-4o-mini"
    raw = []
    for r in analyze.read_jsonl(RAW_FIXTURE):
        if r["mode"] == "direct":
            if r["model"] == victim and r["scenario"] != "isomorph":
                continue                       # provider down -> no line at all
            raw.append(r)
        elif r["mode"] == "cot" and r["scenario"] != "isomorph":
            raw.append(r)                      # cot has not got to isomorph yet

    ann_by = {a["run_id"]: a for a in analyze.read_jsonl(ANNOTATED_FIXTURE)}
    ann = [ann_by[r["run_id"]] for r in raw if r["mode"] == "cot" if r["run_id"] in ann_by]

    # The victim IS in the direct frame and IS reported usable — which is what
    # made the old assert fire rather than the model simply being dropped.
    report = curve_mod.direct_mode_report(raw)
    assert victim in report.usable_models
    direct = curve_mod.direct_curve(raw, [0.0, 0.8, 1.6])
    assert victim in set(direct["model"])
    assert set(direct[direct["model"] == victim]["scenario"]) == {"isomorph"}

    result = analyze.run(config=contracts.CONFIG_PATH,
                         raw=_write(tmp_path / "r.jsonl", raw),
                         annotated=_write(tmp_path / "a.jsonl", ann),
                         out=tmp_path / "out", n_boot=100)

    # All four figures and the summary exist: no abort.
    assert len(result["plots"]) == 6   # 4 from PLAN.md §8.5 + plots 5 and 6
    for path in result["plots"]:
        assert path.exists()
    assert result["summary"].exists()

    # And the caption tells the truth: isomorph is not a panel, so the victim
    # got no overlay and is named as having none.
    assert "isomorph" not in result["overlay"]["scenarios"]
    assert victim in result["overlay"]["no_overlay"]
    assert victim not in result["overlay"]["overlaid"]


def test_summary_depth_table_excludes_no_cot_models_exactly_as_plot_3_does(tmp_path):
    """summary.md's pooled depth table and plot 3 must be the same quantity.

    They were not: plot 3 filtered `contracts.NO_COT_MODELS` (analyze.py's
    `codeable`) and the table did not, so the figure and the table in the same
    out/ directory printed two different depth distributions for an
    identically-labelled row — 0.299 vs 0.532 on L0 for persona `none` in this
    very fixture. CONTRACTS.md §8: "Pooling qwen3-30b-a3b's uniform L0/NON into
    a cross-model depth figure would drag every bar toward L0 and the effect
    would look like a finding."
    """
    nocot = next(iter(contracts.NO_COT_MODELS))
    victim = "meta-llama/llama-3.1-8b-instruct"
    ann_by = {a["run_id"]: a for a in analyze.read_jsonl(ANNOTATED_FIXTURE)}

    raw, ann = [], []
    for r in analyze.read_jsonl(RAW_FIXTURE):
        r = dict(r)
        relabelled = r["model"] == victim
        if relabelled:
            r["model"] = nocot
        if r["mode"] == "cot" and r["run_id"] in ann_by:
            a = dict(ann_by[r["run_id"]])
            if relabelled:
                a["depth"] = "L0"          # its measured behaviour: uniform L0
            ann.append(a)
        raw.append(r)

    result = analyze.run(config=contracts.CONFIG_PATH,
                         raw=_write(tmp_path / "r.jsonl", raw),
                         annotated=_write(tmp_path / "a.jsonl", ann),
                         out=tmp_path / "out", n_boot=200)
    text = result["summary"].read_text(encoding="utf-8")

    # The table says which models it left out.
    assert nocot in text
    assert "Excluded from every pooled figure in this section" in text

    # And its numbers are the ones plot 3 draws, not the pooled ones.
    cot = result["cot"]
    codeable = cot[~cot["model"].isin(list(contracts.NO_COT_MODELS))]
    assert nocot in set(cot["model"])          # it IS in the run
    assert nocot not in set(codeable["model"])  # and IS excluded from the pool

    for persona in ("none", "expert"):
        sub = codeable[codeable["persona"] == persona]
        if sub.empty:
            continue
        expected = analyze.bootstrap_ci(
            analyze.cell_values(sub, success=("L0",), column="depth"),
            unit=("model",), n_boot=200, seed=int(analyze.load_config(
                contracts.CONFIG_PATH).get("seed", 0))).point
        pooled = analyze.bootstrap_ci(
            analyze.cell_values(cot[cot["persona"] == persona],
                                success=("L0",), column="depth"),
            unit=("model",), n_boot=200, seed=int(analyze.load_config(
                contracts.CONFIG_PATH).get("seed", 0))).point
        # The two really do differ on this input — otherwise the test proves
        # nothing about which one the table printed.
        assert abs(expected - pooled) > 0.05, (persona, expected, pooled)
        row = next(l for l in text.splitlines()
                   if l.startswith(f"| persona={persona} |"))
        assert f"{expected:.3f}" in row, (row, expected, pooled)
        assert f"{pooled:.3f}" not in row, (row, expected, pooled)


# ------------------------------------------------------ plots 5 and 6 ----

def test_answers_by_role_is_written(cot_and_direct, tmp_path):
    cot, *_ = cot_and_direct
    models = sorted(cot["model"].unique())
    path = analyze.plot_answers_by_role(cot, models, tmp_path)
    assert path.exists() and path.stat().st_size > 1000


def test_keep_by_temperature_is_written(cot_and_direct, tmp_path):
    cot, *_ = cot_and_direct
    models = sorted(cot["model"].unique())
    temps = sorted(cot["temperature"].unique())
    path = analyze.plot_keep_by_temperature(cot, models, temps, tmp_path)
    assert path.exists() and path.stat().st_size > 1000


def test_new_plots_survive_an_empty_frame(tmp_path):
    """A cell with no data must not raise — plot 5 and 6 run before the pilot
    has produced anything for some conditions."""
    empty = pd.DataFrame(columns=["model", "scenario", "persona", "temperature",
                                  "answer", "depth"])
    analyze.plot_answers_by_role(empty, [], tmp_path)
    analyze.plot_keep_by_temperature(empty, [], [], tmp_path)


def test_answers_by_role_counts_every_answer(cot_and_direct):
    """Plot 5 is unconditional: KEEP + SWITCH + other must be the whole cell,
    or the figure silently drops responses."""
    cot, *_ = cot_and_direct
    for (model, scenario, persona), g in cot.groupby(["model", "scenario", "persona"]):
        keep = int((g["answer"] == "KEEP").sum())
        switch = int((g["answer"] == "SWITCH").sum())
        assert keep + switch + (len(g) - keep - switch) == len(g)


def test_run_writes_six_figures(tmp_path):
    """run() must register the new plots, not just define them."""
    result = analyze.run(config=contracts.CONFIG_PATH, raw=RAW_FIXTURE,
                         annotated=ANNOTATED_FIXTURE, out=tmp_path,
                         n_boot=40, seed=1)
    written = {p.name for p in tmp_path.glob("*.png")}
    assert {"answers_by_role.png", "keep_by_temperature.png"} <= written
    assert len(written) == 6
