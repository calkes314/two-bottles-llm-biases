"""Direct-mode temperature rescaling — PLAN.md §8.3, CONTRACTS.md §8.

The point of direct mode is that the five temperature levels cost *one* call
instead of five. One call at ``temperature=1.0`` with ``logprobs=true`` gives
the model's base distribution over the answer letters; every other temperature
is then a closed-form rescaling of it:

    p_i(T)  ∝  p_i(1) ** (1 / T)

renormalised over the supplied tokens only. That renormalisation is the whole
reason the resulting number is comparable to anything: it is
``P(KEEP | the model answered KEEP or SWITCH)``, a *conditional* rate. Anything
plotted against it must use the same denominator (CONTRACTS.md §8, "Match the
denominators, or plot 1 is a lie").

Nothing here re-implements ``contracts.answer_logprobs``. ``extract_answer_logprobs``
is a thin wrapper that walks a raw record down to ``choices[0]`` and hands it to
the contract function, because that function's three-way status
(``ok`` / ``no_logprobs`` / ``no_answer_tokens``) is the thing callers must not
skip: models fail it in provider-specific ways, so the failures cluster by
model and provider rather than by experimental condition.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src import contracts

# The two letters the curve is renormalised over. DERIVED from the contract's
# token map rather than restated (CONTRACTS.md §0: "Import them. Do not
# re-implement them."), so that if the answer letters ever change, the keys
# `rescale` returns and the keys this module reads out of it change together.
KEEP_TOKEN = next(t for t, a in contracts.DIRECT_ANSWER_TOKENS.items() if a == "KEEP")
SWITCH_TOKEN = next(t for t, a in contracts.DIRECT_ANSWER_TOKENS.items() if a == "SWITCH")

# Every status contracts.answer_logprobs can return. Kept as an explicit tuple so
# a tally always has a row per status even when the count is zero — a status that
# silently never appears reads as "no failures" when it may mean "never checked".
STATUSES = ("ok", "no_logprobs", "no_answer_tokens")

DIRECT_CURVE_COLUMNS = (
    "model", "scenario", "persona", "temperature", "p_keep", "p_switch",
)

#: Stands in for a record whose `model` field is missing, so it still appears in
#: the per-model audit table instead of being dropped by a groupby. CONTRACTS.md
#: §3 mandates the field; a record without one cannot be attributed to a model
#: and must never be silently absorbed into the totals of one that can.
UNKNOWN_MODEL = "<missing model field>"


class DuplicateRecordError(RuntimeError):
    """Raised when raw.jsonl holds more than one record for the same run_id."""


# ------------------------------------------------------------- rescaling ----

def rescale(logprobs: Mapping[str, float], T: float) -> dict[str, float]:
    """Rescale a log-probability distribution to temperature ``T``.

    Implements ``p_i(T) ∝ p_i(1) ** (1 / T)``, renormalised over **the given
    tokens only**. Because of that renormalisation the input need not sum to 1;
    it usually does not, since ``contracts.answer_logprobs`` returns the raw
    ``{K, S}`` masses out of a full-vocabulary distribution. At ``T = 1.0`` the
    result is therefore the input *conditionalised* on those tokens, which is an
    identity when the input already sums to 1.

    ``T = 0`` returns a one-hot on the argmax — the zero-temperature limit, by
    convention rather than by division (CONTRACTS.md §8). Ties go to the first
    token in iteration order, which for a Python dict is insertion order; see
    the note in the body, which is a contract-level wart rather than a local one.

    Numerically stable: the exponent ``logprob / T`` is shifted by its own max
    before ``exp``, so a small ``T`` (which multiplies the spread by ``1/T``)
    cannot overflow. Underflow to exactly zero in the losing tokens is correct —
    that *is* the low-temperature limit.

    Raises ``ValueError`` on an empty distribution or a negative ``T``.
    """
    if not logprobs:
        raise ValueError("rescale() needs at least one token; got an empty mapping")
    if T < 0:
        raise ValueError(f"temperature must be >= 0, got {T!r}")

    keys = list(logprobs)
    lp = np.asarray([float(logprobs[k]) for k in keys], dtype=float)

    if not np.all(np.isfinite(lp)):
        # -inf is a legitimate logprob (probability exactly 0) but it makes the
        # T=0 argmax and the shift below ambiguous, and +inf/nan never is.
        if np.all(np.isneginf(lp)):
            raise ValueError("every token has probability 0; no distribution to rescale")
        if np.any(np.isnan(lp)) or np.any(np.isposinf(lp)):
            raise ValueError(f"non-finite logprob in {dict(logprobs)!r}")

    if T == 0:
        # One-hot on the argmax, per PLAN.md §8.3's reference implementation and
        # CONTRACTS.md §8 ("T = 0.0 is the argmax, by convention").
        #
        # KNOWN WART, left in deliberately because the contract specifies it:
        # on an EXACT tie `argmax` takes the first maximum, and the key order
        # here is the order the provider listed its `top_logprobs`. So a
        # genuinely indifferent model reports p_keep = 1.0 or 0.0 depending on
        # an invisible detail of the API response, and the T=0 column then
        # disagrees with this function's own T -> 0+ limit, which is 0.5.
        # Splitting the mass over the argmax set would fix both, but that is a
        # contract change, not a local one. Exact ties are vanishingly rare in
        # real logprobs; `direct_curve` callers who care should check for them.
        p = np.zeros_like(lp)
        p[int(np.argmax(lp))] = 1.0
        return dict(zip(keys, (float(x) for x in p)))

    z = lp / T                      # log(p ** (1/T)) = (1/T) · log p
    z = z - np.max(z)               # stabilise: the largest exponent becomes 0
    p = np.exp(z)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:      # pragma: no cover — guarded above
        raise ValueError(f"rescaling {dict(logprobs)!r} at T={T!r} produced no mass")
    p = p / total
    return dict(zip(keys, (float(x) for x in p)))


def curve(logprobs: Mapping[str, float], temps: Iterable[float]) -> dict[float, dict[str, float]]:
    """``{T: rescale(logprobs, T)}`` for each temperature. PLAN.md §8.3's shape."""
    return {float(T): rescale(logprobs, T) for T in temps}


# ------------------------------------------------ raw-record extraction ----

def extract_answer_logprobs(raw_record: Mapping) -> tuple[dict[str, float] | None, str]:
    """``(logprobs, status)`` for one ``data/raw.jsonl`` record.

    A thin wrapper around ``contracts.answer_logprobs`` — which sums probability
    across token variants (``"K"``, ``" K"``, ``"k"``) and rejects non-bare
    variants like ``"(K"`` — plus the walk down to ``choices[0]`` and the
    envelope checks that walk can fail.

    ``status`` is ``"ok"``, ``"no_logprobs"`` or ``"no_answer_tokens"``.
    **Callers must check it.** A non-``ok`` status means the record cannot feed
    the temperature curve, and those failures are not missing at random: they
    cluster by model and by provider (the provider decides whether logprobs come
    back at all, and whether the array is aligned with the emitted text), never
    by experimental condition. Dropping the failing *cells* would therefore
    reweight the surviving cells of the same model; drop the whole model
    instead. ``direct_curve`` does exactly that.
    """
    response = raw_record.get("response")
    if not isinstance(response, dict):
        return None, "no_logprobs"
    choices = response.get("choices")
    if not choices or not isinstance(choices[0], dict):
        return None, "no_logprobs"
    return contracts.answer_logprobs(choices[0])


def direct_status_table(raw_records: Iterable[Mapping]) -> pd.DataFrame:
    """One row per direct-mode raw record: how it fared in extraction.

    Columns: ``run_id, model, provider, scenario, persona, status``. This is the
    audit trail behind every "n direct-mode records were unusable" number in the
    summary, and the reason those numbers can be broken down by provider.
    """
    rows = []
    for rec in _direct_records(raw_records):
        _, status = extract_answer_logprobs(rec)
        rows.append({
            "run_id": rec.get("run_id"),
            # A None here would be dropped by the groupby in direct_mode_report,
            # so by_model would stop summing to n_records and the record would
            # leave the overlay without appearing in dropped_models.
            "model": rec.get("model") or UNKNOWN_MODEL,
            "provider": rec.get("provider"),
            "scenario": rec.get("scenario"),
            "persona": rec.get("persona"),
            "status": status,
        })
    return pd.DataFrame(rows, columns=["run_id", "model", "provider",
                                       "scenario", "persona", "status"])


@dataclass(frozen=True)
class DirectModeReport:
    """What direct mode gave us, and what it cost to find out.

    ``status_counts`` is a tally over every status in ``STATUSES`` (zeros
    included). ``dropped_models`` are the models excluded from the analytic
    overlay, each mapped to why.
    """

    n_records: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    by_model: pd.DataFrame = field(default_factory=pd.DataFrame)
    usable_models: tuple[str, ...] = ()
    dropped_models: dict[str, str] = field(default_factory=dict)

    @property
    def n_unusable(self) -> int:
        return self.n_records - self.status_counts.get("ok", 0)

    def lines(self) -> list[str]:
        """Human-readable summary lines, for a report or a log."""
        out = [
            f"direct-mode records: {self.n_records}",
            f"  usable (status ok):     {self.status_counts.get('ok', 0)}",
        ]
        for status in STATUSES:
            if status == "ok":
                continue
            out.append(f"  {status:<22}  {self.status_counts.get(status, 0)}")
        out.append(f"models with a usable curve: {len(self.usable_models)} "
                   f"{list(self.usable_models)}")
        for model, why in self.dropped_models.items():
            out.append(f"  dropped {model}: {why}")
        return out


def direct_mode_report(raw_records: Iterable[Mapping]) -> DirectModeReport:
    """Count every extraction status and decide which models keep a curve.

    The rule, from CONTRACTS.md §8: a model contributes an analytic curve only
    if **all** of its direct-mode records extracted cleanly, and only if it is
    in ``contracts.DIRECT_CURVE_MODELS`` (the set whose logprobs were verified
    end to end against a pinned provider). Anything else loses the whole model,
    not the offending cells — a partial drop would silently reweight the
    (scenario, persona) cells that survive, and since the failures cluster by
    provider that reweighting is systematic, not noise.
    """
    table = direct_status_table(raw_records)
    counts = {s: 0 for s in STATUSES}
    for status, n in Counter(table["status"]).items():
        counts[status] = counts.get(status, 0) + int(n)

    if table.empty:
        by_model = pd.DataFrame(columns=["model", "n", *STATUSES])
        return DirectModeReport(0, counts, by_model, (), {})

    by_model = (
        table.assign(n=1)
        .pivot_table(index="model", columns="status", values="n",
                     aggfunc="sum", fill_value=0)
        .reindex(columns=list(STATUSES), fill_value=0)
        .reset_index()
    )
    by_model.columns.name = None
    by_model.insert(1, "n", by_model[list(STATUSES)].sum(axis=1))

    usable: list[str] = []
    dropped: dict[str, str] = {}
    for row in by_model.itertuples(index=False):
        model = row.model
        bad = {s: getattr(row, s) for s in STATUSES if s != "ok" and getattr(row, s)}
        if model not in contracts.DIRECT_CURVE_MODELS:
            dropped[model] = ("not in contracts.DIRECT_CURVE_MODELS "
                              "(logprobs never verified end to end)")
        elif bad:
            detail = ", ".join(f"{n}× {s}" for s, n in sorted(bad.items()))
            dropped[model] = f"{sum(bad.values())}/{row.n} direct records unusable ({detail})"
        else:
            usable.append(model)

    return DirectModeReport(len(table), counts, by_model, tuple(usable), dropped)


# --------------------------------------------------------- the tidy curve ----

def direct_curve(raw_records: Iterable[Mapping],
                 temps: Sequence[float]) -> pd.DataFrame:
    """The analytic temperature curve, one row per (model, scenario, persona, T).

    Columns: ``model, scenario, persona, temperature, p_keep, p_switch``.

    ``p_keep`` is ``P(KEEP | KEEP or SWITCH)`` at that temperature — a
    *conditional* rate, because ``rescale`` renormalises over ``{K, S}`` only.
    Only models that survive ``direct_mode_report`` appear; a model with any
    unusable direct record is dropped whole (see that function for why).
    """
    records = list(_direct_records(raw_records))
    report = direct_mode_report(records)
    keep_models = set(report.usable_models)

    rows = []
    for rec in records:
        if (rec.get("model") or UNKNOWN_MODEL) not in keep_models:
            continue
        logprobs, status = extract_answer_logprobs(rec)
        if status != "ok" or logprobs is None:      # pragma: no cover — excluded above
            continue
        for T in temps:
            p = rescale(logprobs, float(T))
            rows.append({
                "model": rec.get("model"),
                "scenario": rec.get("scenario"),
                "persona": rec.get("persona"),
                "temperature": float(T),
                "p_keep": p[KEEP_TOKEN],
                "p_switch": p[SWITCH_TOKEN],
            })

    df = pd.DataFrame(rows, columns=list(DIRECT_CURVE_COLUMNS))
    if df.empty:
        return df.astype({"temperature": float, "p_keep": float, "p_switch": float})
    return df.sort_values(list(DIRECT_CURVE_COLUMNS[:4]), ignore_index=True)


def _direct_records(raw_records: Iterable[Mapping]) -> list[Mapping]:
    """Grid direct-mode records only, and exactly one per ``run_id``.

    ``direct_validation`` is a *sampling* run and must never be pooled with the
    grid (CONTRACTS.md §8); ``contamination`` and ``cot`` are different questions
    entirely. Filtering on ``mode`` is how all three stay out.

    A repeated **cell** is refused rather than averaged. Direct mode runs one
    call per (model, scenario, persona) — the whole temperature curve comes off
    that single logprob pair — so a second record for the same cell doubles its
    weight in every persona mean downstream, silently. And the two are not
    interchangeable draws: CONTRACTS.md §8 measured the same prompt to one model
    returning P(KEEP) from 0.854 to 0.986 across providers, so a duplicate is
    just as likely to be two contradictory numbers as two equal ones.
    ``analyze.load_cot`` applies the matching rule to the CoT records of the
    same file, and to ``annotated.jsonl``.
    """
    direct = [r for r in raw_records if r.get("mode") == "direct"]
    seen: dict[tuple, int] = {}
    for rec in direct:
        # Keyed on the CELL, not on `run_id`. A repeated run_id is necessarily a
        # repeated cell, so this subsumes that case — and it also catches the one
        # a run_id check misses: a direct record written with a non-zero
        # `replicate` hashes to a different run_id (CONTRACTS.md §2) while still
        # being a second measurement of the same cell.
        cell = (rec.get("model"), rec.get("scenario"), rec.get("persona"))
        seen[cell] = seen.get(cell, 0) + 1
    dupes = {c: n for c, n in seen.items() if n > 1}
    if dupes:
        shown = "; ".join(f"{'/'.join(str(x) for x in c)}×{n}"
                          for c, n in list(dupes.items())[:3])
        raise DuplicateRecordError(
            f"{len(dupes)} direct-mode (model, scenario, persona) cell(s) appear "
            f"more than once in raw.jsonl: {shown}{' …' if len(dupes) > 3 else ''}. "
            f"Direct mode is one call per cell — the whole temperature curve "
            f"comes off that single logprob pair — so a repeat double-weights "
            f"that cell in every average. And the two records need not agree: "
            f"CONTRACTS.md §8 measured the same prompt to one model returning "
            f"P(KEEP) from 0.854 to 0.986 across providers. This usually means "
            f"two runs were concatenated, the runner's resume was bypassed, or a "
            f"direct job was emitted with a non-zero replicate. De-duplicate "
            f"raw.jsonl before analysing it."
        )
    return direct


__all__ = [
    "rescale", "curve", "extract_answer_logprobs", "direct_curve",
    "direct_status_table", "direct_mode_report", "DirectModeReport",
    "DuplicateRecordError", "STATUSES", "DIRECT_CURVE_COLUMNS",
    "UNKNOWN_MODEL", "KEEP_TOKEN", "SWITCH_TOKEN",
]
