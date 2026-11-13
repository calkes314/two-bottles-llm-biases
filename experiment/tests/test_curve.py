"""Tests for src/curve.py — the analytic direct-mode temperature curve.

The five named cases PLAN.md §8.3 and the analysis brief ask for, plus the
things that would quietly corrupt an overlay rather than crash it: an
unnormalised input, a model whose direct records only *partly* extract, and
`direct_validation` records leaking into the grid.

    python -m pytest tests/test_curve.py -q
"""

import math

import numpy as np
import pandas as pd
import pytest

from src import contracts
from src import curve


# --------------------------------------------------------------- helpers ----

def _logprobs(**probs):
    """{"K": 0.8, "S": 0.2} -> {"K": log 0.8, "S": log 0.2}, insertion-ordered."""
    return {k: math.log(v) for k, v in probs.items()}


def _top_logprobs(*pairs):
    return [{"token": t, "logprob": math.log(p)} for t, p in pairs]


def _direct_record(model="openai/gpt-4o-mini", scenario="default_keep",
                   persona="none", *, top=None, logprobs_block="auto",
                   provider="OpenAI", mode="direct"):
    """A `data/raw.jsonl`-shaped direct-mode record (CONTRACTS.md §3, §4).

    `logprobs_block=None` mimics a provider that returns no logprobs at all;
    `top=[...]` sets the top_logprobs of the single emitted position.
    """
    if top is None:
        top = _top_logprobs(("K", 0.8), ("S", 0.2))
    if logprobs_block == "auto":
        logprobs_block = {"content": [{"token": "K", "logprob": math.log(0.8),
                                       "top_logprobs": top}]}
    return {
        "run_id": contracts.run_id(model, scenario, persona, mode,
                                   contracts.DIRECT_TEMPERATURE, 0),
        "experiment": "test",
        "model": model,
        "tier": "mid",
        "scenario": scenario,
        "persona": persona,
        "mode": mode,
        "temperature": contracts.DIRECT_TEMPERATURE,
        "replicate": 0,
        "messages": contracts.build_messages(scenario, persona, "direct"),
        "max_tokens": contracts.DIRECT_MAX_TOKENS,
        "logprobs": True,
        "top_logprobs": contracts.DIRECT_TOP_LOGPROBS,
        "provider_routing": contracts.DIRECT_PROVIDER_PIN.get(model),
        "response": {
            "model": model,
            "provider": provider,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "K"},
                         "logprobs": logprobs_block}],
            "usage": {"prompt_tokens": 201, "completion_tokens": 1, "cost": 3.075e-05},
        },
        "model_returned": model,
        "provider": provider,
        "usage": {"prompt_tokens": 201, "completion_tokens": 1, "cost": 3.075e-05},
        "ts": "2026-09-03T15:02:04Z",
    }


# ---------------------------------------------- the five specified cases ----

def test_T1_returns_the_input_distribution_unchanged():
    """T = 1 is the identity on an already-normalised distribution."""
    lp = _logprobs(K=0.73, S=0.27)
    out = curve.rescale(lp, 1.0)
    assert out["K"] == pytest.approx(0.73, abs=1e-12)
    assert out["S"] == pytest.approx(0.27, abs=1e-12)


def test_T_approaching_zero_is_one_hot_on_the_argmax():
    """T = 1e-6, the numeric stand-in for the zero-temperature limit.

    This is the case a naive `exp(lp / T)` overflows on: 1/T = 1e6 multiplies
    the spread by a million. Subtracting the max first keeps it finite.
    """
    for lp, winner, loser in [(_logprobs(K=0.8, S=0.2), "K", "S"),
                              (_logprobs(K=0.2, S=0.8), "S", "K"),
                              (_logprobs(K=0.51, S=0.49), "K", "S")]:
        out = curve.rescale(lp, 1e-6)
        assert out[winner] == pytest.approx(1.0, abs=1e-12), lp
        assert out[loser] == pytest.approx(0.0, abs=1e-12), lp


def test_T_exactly_zero_is_the_argmax_not_a_division():
    out = curve.rescale(_logprobs(K=0.2, S=0.8), 0.0)
    assert out == {"K": 0.0, "S": 1.0}
    assert curve.rescale(_logprobs(K=0.9, S=0.1), 0) == {"K": 1.0, "S": 0.0}
    # A gap far below 1e-6 in log space: only *exactly* zero resolves it.
    assert curve.rescale(_logprobs(K=0.500001, S=0.499999), 0.0) == {"K": 1.0, "S": 0.0}


def test_T_zero_tie_breaking_is_the_contract_wart_not_the_limit():
    """Pins the KNOWN, deliberate divergence at T=0 so nobody "fixes" it locally.

    PLAN.md §8.3's reference code and CONTRACTS.md §8 both specify a one-hot on
    the argmax, and `argmax` takes the FIRST maximum. The key order here is the
    order the provider listed its top_logprobs, so on an exact tie the same
    indifferent model reports p_keep=1.0 or 0.0 depending on the API response —
    and that disagrees with this function's own T -> 0+ limit, which is 0.5.
    Changing it is a contract change and belongs in CONTRACTS.md, not here.
    """
    tie = math.log(0.5)
    assert curve.rescale({"K": tie, "S": tie}, 0.0) == {"K": 1.0, "S": 0.0}
    assert curve.rescale({"S": tie, "K": tie}, 0.0) == {"S": 1.0, "K": 0.0}

    # The limit from above disagrees. Documented, not accidental.
    assert curve.rescale({"K": tie, "S": tie}, 1e-9)["K"] == pytest.approx(0.5, abs=1e-12)

    # A unique maximum — the case that actually occurs — is unambiguous.
    assert curve.rescale(_logprobs(K=0.5000001, S=0.4999999), 0.0) == {"K": 1.0, "S": 0.0}


def test_1e6_is_not_actually_zero():
    """`T = 1e-6` one-hots any realistic distribution, but it is an
    approximation, not the T=0 branch: a log-gap of the same order as T is not
    resolved by it. Use `T == 0` when the argmax is what is meant."""
    out = curve.rescale(_logprobs(K=0.500001, S=0.499999), 1e-6)
    assert out["K"] == pytest.approx(1 / (1 + math.exp(-4.0)), abs=1e-9)
    assert out["K"] < 0.99


def test_T100_is_within_001_of_uniform():
    out = curve.rescale(_logprobs(K=0.8, S=0.2), 100.0)
    assert abs(out["K"] - 0.5) < 0.01
    assert abs(out["S"] - 0.5) < 0.01
    assert out["K"] + out["S"] == pytest.approx(1.0, abs=1e-12)


def test_known_two_token_case_matches_hand_computation():
    """p(K)=0.8, p(S)=0.2.

    T = 1/2:  p ∝ p²      → 0.64 : 0.04       = 16/17 : 1/17
    T = 2:    p ∝ sqrt(p)  → √0.8 : √0.2       =  2/3  : 1/3   (since √0.8/√0.2 = 2)
    """
    lp = _logprobs(K=0.8, S=0.2)

    half = curve.rescale(lp, 0.5)
    assert half["K"] == pytest.approx(16 / 17, abs=1e-9)
    assert half["S"] == pytest.approx(1 / 17, abs=1e-9)

    double = curve.rescale(lp, 2.0)
    assert double["K"] == pytest.approx(2 / 3, abs=1e-9)
    assert double["S"] == pytest.approx(1 / 3, abs=1e-9)


def test_leading_space_token_variants_are_merged():
    """"K", " K" and "k" are the same answer; "(K" is not a bare answer letter.

    The merging lives in contracts.answer_logprobs and must survive the wrapper:
    0.5 + 0.15 + 0.05 = 0.70 for K, 0.20 + 0.05 = 0.25 for S, and the 0.05 of
    "(K" is discarded.
    """
    rec = _direct_record(top=_top_logprobs(
        ("K", 0.50), (" K", 0.15), ("k", 0.05),
        ("S", 0.20), (" S", 0.05),
        ("(K", 0.05),                      # must NOT count
    ))
    logprobs, status = curve.extract_answer_logprobs(rec)
    assert status == "ok"
    assert math.exp(logprobs["K"]) == pytest.approx(0.70, abs=1e-12)
    assert math.exp(logprobs["S"]) == pytest.approx(0.25, abs=1e-12)

    # ... and the merged pair renormalises to the conditional rate, not to 0.70.
    p = curve.rescale(logprobs, 1.0)
    assert p["K"] == pytest.approx(0.70 / 0.95, abs=1e-12)


# ----------------------------------------------- rescale, the edge cases ----

def test_rescale_renormalises_over_the_given_tokens_only():
    """The {K, S} masses out of a full vocabulary do not sum to 1, and the
    conditional rate is exactly what plot 1 must be compared against."""
    out = curve.rescale(_logprobs(K=0.30, S=0.10), 1.0)
    assert out["K"] == pytest.approx(0.75, abs=1e-12)
    assert out["S"] == pytest.approx(0.25, abs=1e-12)


def test_rescale_is_monotone_towards_uniform_as_T_rises():
    lp = _logprobs(K=0.9, S=0.1)
    keeps = [curve.rescale(lp, T)["K"] for T in (0.25, 0.5, 1.0, 2.0, 4.0, 16.0)]
    assert keeps == sorted(keeps, reverse=True)
    assert keeps[0] > 0.98 and keeps[-1] < 0.55


def test_rescale_handles_a_zero_probability_token():
    out = curve.rescale({"K": math.log(0.6), "S": -math.inf}, 0.8)
    assert out == {"K": 1.0, "S": 0.0}


def test_rescale_rejects_nonsense():
    with pytest.raises(ValueError):
        curve.rescale({}, 1.0)
    with pytest.raises(ValueError):
        curve.rescale(_logprobs(K=0.5, S=0.5), -0.1)
    with pytest.raises(ValueError):
        curve.rescale({"K": -math.inf, "S": -math.inf}, 1.0)
    with pytest.raises(ValueError):
        curve.rescale({"K": float("nan"), "S": 0.0}, 1.0)


def test_curve_helper_returns_one_distribution_per_temperature():
    temps = [0.0, 0.4, 0.8, 1.2, 1.6]
    out = curve.curve(_logprobs(K=0.8, S=0.2), temps)
    assert list(out) == temps
    assert all(abs(sum(d.values()) - 1.0) < 1e-12 for d in out.values())


# ---------------------------------------- extraction statuses on records ----

def test_extract_reports_no_logprobs_when_the_provider_returned_none():
    """google/gemini-2.5-flash-lite's failure mode: content but no logprobs."""
    assert curve.extract_answer_logprobs(
        _direct_record(logprobs_block=None)) == (None, "no_logprobs")


def test_extract_reports_no_logprobs_on_a_broken_envelope():
    assert curve.extract_answer_logprobs({}) == (None, "no_logprobs")
    assert curve.extract_answer_logprobs({"response": None}) == (None, "no_logprobs")
    assert curve.extract_answer_logprobs({"response": {}}) == (None, "no_logprobs")
    assert curve.extract_answer_logprobs(
        {"response": {"choices": []}}) == (None, "no_logprobs")


def test_extract_reports_no_answer_tokens_on_a_misaligned_array():
    """mistral-nemo @ Parasail: logprobs[0] is '</s>' while content is 'K'."""
    rec = _direct_record(logprobs_block={"content": [
        {"token": "</s>", "logprob": -0.04,
         "top_logprobs": _top_logprobs(("</s>", 0.96), ("\n\n", 0.03))},
    ]})
    assert curve.extract_answer_logprobs(rec) == (None, "no_answer_tokens")


def test_extract_rejects_a_position_offering_only_one_letter():
    """Requiring both letters is what makes the renormalisation meaningful."""
    rec = _direct_record(top=_top_logprobs(("K", 0.9), ("\n", 0.05)))
    assert curve.extract_answer_logprobs(rec) == (None, "no_answer_tokens")


# ------------------------------------------------------- the tidy curve ----

_GOOD = "openai/gpt-4o-mini"
_ALSO_GOOD = "meta-llama/llama-3.1-8b-instruct"
_BROKEN = "ibm-granite/granite-4.1-8b"
TEMPS = [0.0, 0.4, 0.8, 1.2, 1.6]


def _grid(model, *, broken_personas=()):
    out = []
    for scenario in contracts.SCENARIOS:
        for persona in ("none", "expert"):
            kwargs = {"logprobs_block": None} if persona in broken_personas else {}
            out.append(_direct_record(model, scenario, persona, **kwargs))
    return out


def test_answer_letters_are_derived_from_the_contract():
    """CONTRACTS.md §0: import the contract's constants, do not restate them."""
    assert curve.KEEP_TOKEN == "K" and curve.SWITCH_TOKEN == "S"
    assert contracts.DIRECT_ANSWER_TOKENS[curve.KEEP_TOKEN] == "KEEP"
    assert contracts.DIRECT_ANSWER_TOKENS[curve.SWITCH_TOKEN] == "SWITCH"
    # And they are the keys rescale actually returns, so the two cannot drift.
    p = curve.rescale(_logprobs(K=0.8, S=0.2), 1.0)
    assert set(p) == {curve.KEEP_TOKEN, curve.SWITCH_TOKEN}


def test_direct_curve_shape_and_columns():
    df = curve.direct_curve(_grid(_GOOD), TEMPS)
    assert list(df.columns) == list(curve.DIRECT_CURVE_COLUMNS)
    assert len(df) == len(contracts.SCENARIOS) * 2 * len(TEMPS)
    assert set(df["temperature"]) == set(TEMPS)
    assert np.allclose(df["p_keep"] + df["p_switch"], 1.0)
    assert df["model"].unique().tolist() == [_GOOD]


def test_direct_curve_drops_the_whole_model_not_the_failing_cells():
    """CONTRACTS.md §8: a partial drop reweights the surviving (scenario,
    persona) cells of the same model, and the failures cluster by provider, so
    that reweighting is systematic rather than noise."""
    records = _grid(_GOOD) + _grid(_BROKEN, broken_personas={"expert"})
    df = curve.direct_curve(records, TEMPS)

    assert df["model"].unique().tolist() == [_GOOD]
    assert _BROKEN not in set(df["model"])
    # Specifically: its three *working* cells are gone too.
    assert len(df) == len(contracts.SCENARIOS) * 2 * len(TEMPS)

    report = curve.direct_mode_report(records)
    assert report.n_records == len(contracts.SCENARIOS) * 2 * 2   # 2 models
    # _GOOD contributes 2 personas x every scenario; _BROKEN contributes its
    # "none" persona as ok and its "expert" persona as no_logprobs.
    n_sc = len(contracts.SCENARIOS)
    assert report.status_counts == {"ok": n_sc * 3, "no_logprobs": n_sc,
                                    "no_answer_tokens": 0}
    assert report.n_unusable == len(contracts.SCENARIOS)
    assert report.usable_models == (_GOOD,)
    assert _BROKEN in report.dropped_models
    assert (f"{n_sc}/{n_sc * 2}" in report.dropped_models[_BROKEN])
    assert "no_logprobs" in report.dropped_models[_BROKEN]


def test_direct_mode_report_tallies_every_status_including_zeros():
    report = curve.direct_mode_report(_grid(_GOOD))
    assert set(report.status_counts) == set(curve.STATUSES)
    assert report.status_counts["no_answer_tokens"] == 0
    assert report.n_unusable == 0
    assert report.dropped_models == {}
    assert f"direct-mode records: {len(contracts.SCENARIOS) * 2}" in report.lines()[0]


def test_direct_mode_report_drops_models_outside_DIRECT_CURVE_MODELS():
    """A model can extract cleanly and still have no verified end-to-end curve."""
    unverified = "fixture/never-verified"
    assert unverified not in contracts.DIRECT_CURVE_MODELS
    report = curve.direct_mode_report(_grid(unverified))
    # _grid is contracts.SCENARIOS x 2 personas; extraction itself was fine
    assert report.status_counts["ok"] == len(contracts.SCENARIOS) * 2
    assert report.usable_models == ()
    assert "DIRECT_CURVE_MODELS" in report.dropped_models[unverified]
    assert curve.direct_curve(_grid(unverified), TEMPS).empty


def test_direct_curve_ignores_every_other_mode():
    """direct_validation is a sampling run and must never be pooled with the
    grid (CONTRACTS.md §8); cot and contamination are different questions."""
    records = (
        _grid(_GOOD)
        + [_direct_record(_ALSO_GOOD, mode=contracts.DIRECT_VALIDATION_MODE)]
        + [_direct_record(_ALSO_GOOD, mode="cot")]
        + [_direct_record(_ALSO_GOOD, mode=contracts.CONTAMINATION_MODE)]
    )
    df = curve.direct_curve(records, TEMPS)
    assert df["model"].unique().tolist() == [_GOOD]
    assert curve.direct_mode_report(records).n_records == len(contracts.SCENARIOS) * 2


def test_direct_curve_is_empty_but_typed_when_nothing_survives():
    df = curve.direct_curve([], TEMPS)
    assert df.empty
    assert list(df.columns) == list(curve.DIRECT_CURVE_COLUMNS)
    assert df["p_keep"].dtype == float


def test_direct_curve_p_keep_matches_rescale_on_the_same_logprobs():
    """No second implementation of the maths hiding inside direct_curve."""
    rec = _direct_record(_GOOD, top=_top_logprobs(("K", 0.15), ("S", 0.85)))
    df = curve.direct_curve([rec], [0.4])
    logprobs, status = curve.extract_answer_logprobs(rec)
    assert status == "ok"
    expected = curve.rescale(logprobs, 0.4)
    assert df["p_keep"].iloc[0] == pytest.approx(expected["K"], abs=1e-15)
    assert df["p_switch"].iloc[0] == pytest.approx(expected["S"], abs=1e-15)


def test_a_duplicate_direct_run_id_is_refused_not_averaged():
    """One call per (model, scenario, persona) — a repeat double-weights that
    cell in every persona mean, and the two records need not even agree, since
    provider routing changes the measured distribution (CONTRACTS.md §8)."""
    rec = _direct_record(_GOOD)
    contradicting = _direct_record(_GOOD, top=_top_logprobs(("K", 0.01), ("S", 0.99)))
    assert rec["run_id"] == contradicting["run_id"]        # same cell, opposite answer

    for records in ([rec, rec], [rec, contradicting]):
        with pytest.raises(curve.DuplicateRecordError, match="more than once"):
            curve.direct_curve(records, TEMPS)
        with pytest.raises(curve.DuplicateRecordError):
            curve.direct_mode_report(records)

    # A duplicate in ANOTHER mode is not this function's business.
    curve.direct_curve([rec, _direct_record(_GOOD, mode="cot"),
                        _direct_record(_GOOD, mode="cot")], TEMPS)


def test_a_record_with_no_model_field_is_surfaced_not_vanished():
    """CONTRACTS.md §3 mandates `model`. A record without one cannot be
    attributed, so it must appear in the audit table rather than be dropped by
    a groupby while still counting toward the totals."""
    good = _direct_record(_GOOD)
    orphan = _direct_record(_GOOD, scenario="isomorph")
    del orphan["model"]

    report = curve.direct_mode_report([good, orphan])
    assert report.n_records == 2
    # The audit table reconciles with its own totals.
    assert int(report.by_model["n"].sum()) == report.n_records
    assert curve.UNKNOWN_MODEL in set(report.by_model["model"])
    # And it is reported as dropped, with a reason, rather than silently gone.
    assert curve.UNKNOWN_MODEL in report.dropped_models
    assert report.usable_models == (_GOOD,)
    assert len(curve.direct_curve([good, orphan], TEMPS)) == len(TEMPS)


def test_direct_status_table_records_the_provider():
    """Failures cluster by provider, so the audit trail has to carry it."""
    table = curve.direct_status_table(_grid(_GOOD, broken_personas={"none"}))
    assert isinstance(table, pd.DataFrame)
    assert set(table["provider"]) == {"OpenAI"}
    assert (table.loc[table["persona"] == "none", "status"].tolist()
            == ["no_logprobs"] * len(contracts.SCENARIOS))
    assert (table.loc[table["persona"] == "expert", "status"].tolist()
            == ["ok"] * len(contracts.SCENARIOS))
