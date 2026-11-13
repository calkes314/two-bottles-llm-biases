"""Aggregate, interval-estimate and plot the two-bottle grid — PLAN.md §8.5.

    python -m src.analyze --config config/budget.yaml [--raw P] [--annotated P] [--out D]

Two arms come in and are never mixed:

* **cot** — ``data/annotated.jsonl`` joined to ``data/raw.jsonl`` on ``run_id``,
  which is the only join key (CONTRACTS.md §5). The annotation carries the coded
  answer and depth; the raw record carries the condition the annotator was
  deliberately blind to.
* **direct** — read through ``src.curve``, which turns one logprob pair per cell
  into the whole analytic temperature curve.

Everything downstream obeys three rules that are easy to break and expensive to
break quietly:

1. **The unit of analysis is the (model × condition) cell, never the individual
   sample.** See ``bootstrap_ci`` for why, at length.
2. **Denominators must match whatever they are plotted against.** The analytic
   curve is conditional on ``{KEEP, SWITCH}``; a raw KEEP rate is not. Plot 1
   uses the conditional denominator so the two series mean the same thing; plots
   2-4 use the unconditional one. Every figure says which in its caption.
3. **Records that cannot feed the curve are not missing at random** — they
   cluster by provider and model. So a model missing any direct-mode logprobs
   loses its whole overlay rather than the offending cells, and the counts are
   reported rather than absorbed.

Output: four PNGs and ``summary.md`` in ``--out``. The summary is numbers only;
the interpretation is the human's job.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src import contracts
from src import curve as curve_mod

# ------------------------------------------------------------- constants ----

#: The (model × condition) cell. PLAN.md §8.5's unit of analysis.
CELL_KEYS: tuple[str, ...] = ("model", "scenario", "persona", "temperature")

#: The two answers the analytic curve renormalises over. A "conditional" rate
#: has these as its denominator; an "unconditional" rate has every answer.
#: Derived from the direct-mode token map, which is the same set by definition —
#: `rescale` renormalises over exactly the letters in it.
BINARY_ANSWERS: tuple[str, ...] = tuple(
    a for a in contracts.ANSWERS if a in set(contracts.DIRECT_ANSWER_TOKENS.values())
)

#: Everything the conditional denominator excludes. Also plot 4's series, in
#: `contracts.ANSWERS` order. Derived rather than listed so the two stay exact
#: complements: plot 1's caption promises that plot 4 shows what plot 1 drops,
#: and a label added to the enum must land on one side or the other, never
#: vanish from both. Order is canonical (CONTRACTS.md §6).
NONBINARY_ANSWERS: tuple[str, ...] = tuple(
    a for a in contracts.ANSWERS if a not in BINARY_ANSWERS
)

#: The two scenarios that form the flip contrast. `isomorph` is the
#: contamination control and has no partner, so it has no flip rate.
FLIP_SCENARIOS: tuple[str, str] = ("default_keep", "default_switch")

#: Cell keys for the flip statistic: it is a contrast *across* scenario, so
#: scenario is consumed rather than indexed.
FLIP_CELL_KEYS: tuple[str, ...] = ("model", "persona", "temperature")

N_BOOT_DEFAULT = 1000
ALPHA = 0.05

#: Columns whose presence proves a frame is sample-level, not cell-level. Any
#: one of them reaching the estimator means someone is about to pool 5,625
#: draws as though they were 5,625 subjects.
_SAMPLE_LEVEL_TELLS: tuple[str, ...] = ("run_id", "replicate", "answer", "depth")


class ClusteringError(RuntimeError):
    """Raised when an estimator is handed sample-level rows, or no cluster.

    This is a guard against the one statistical mistake PLAN.md §8.5 says will
    stop a trained reader from reading further. It is deliberately an exception
    and not a warning: the wrong answer here looks completely plausible.
    """


class DataIntegrityError(RuntimeError):
    """Raised when raw.jsonl / annotated.jsonl violate CONTRACTS.md §4-5."""


# ------------------------------------------------------------------ load ----

def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict]:
    """Parse a JSONL file, naming the line number of anything that will not.

    A truncated final line is the normal shape of a killed run (CONTRACTS.md §4
    appends and flushes per record), so it gets its own message rather than a
    bare ``JSONDecodeError``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist")
    out: list[dict] = []
    with p.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DataIntegrityError(
                    f"{p}:{lineno} is not valid JSON ({exc.msg}). A truncated "
                    f"last line usually means the writer was killed mid-append; "
                    f"delete that line and re-run, or resume the run first."
                ) from exc
    return out


@dataclass(frozen=True)
class JoinReport:
    """What survived the raw × annotated join, and what did not."""

    n_raw_total: int = 0
    n_raw_cot: int = 0
    n_raw_direct: int = 0
    n_raw_other_modes: dict[str, int] = None            # type: ignore[assignment]
    n_annotated: int = 0
    n_joined: int = 0
    n_annotated_without_raw: int = 0
    n_cot_without_annotation: int = 0

    def lines(self) -> list[str]:
        other = ", ".join(f"{m}={n}" for m, n in sorted((self.n_raw_other_modes or {}).items()))
        return [
            f"raw records: {self.n_raw_total} "
            f"(cot={self.n_raw_cot}, direct={self.n_raw_direct}"
            + (f", {other}" if other else "") + ")",
            f"annotated records: {self.n_annotated}",
            f"joined cot records: {self.n_joined}",
            f"annotations with no raw record: {self.n_annotated_without_raw}",
            f"cot raw records with no annotation: {self.n_cot_without_annotation}",
        ]


def load_cot(raw_records: Iterable[Mapping],
             annotated_records: Iterable[Mapping]) -> tuple[pd.DataFrame, JoinReport]:
    """Join annotations to their raw records on ``run_id``, for cot only.

    The annotator is blind to condition by design (CONTRACTS.md §5), so
    ``model``, ``scenario``, ``persona`` and ``temperature`` exist only on the
    raw side. ``run_id`` is the only key that connects them.

    ``direct``, ``contamination`` and ``direct_validation`` records are filtered
    out here rather than later: ``direct_validation`` in particular is a
    *sampling* run at real temperatures, and pooling it with the grid would
    double-count the one model it covers.
    """
    raw = list(raw_records)
    ann = list(annotated_records)

    modes: dict[str, int] = {}
    for rec in raw:
        mode = rec.get("mode")
        modes[str(mode)] = modes.get(str(mode), 0) + 1

    cot_raw = [r for r in raw if r.get("mode") == "cot"]
    raw_ids = {r.get("run_id") for r in cot_raw}

    # Same rule on the raw side as on the annotated side below. A repeated raw
    # CoT line joins to its annotation a second time and adds a second sample to
    # that cell. A *full* duplication leaves the proportions intact and is merely
    # a wrong n; a partial one — a subset re-run appended after a config change,
    # or two runner processes overlapping — pulls that cell's rate toward the
    # duplicated answers with nothing to signal it.
    dupes: dict[object, int] = {}
    for rec in cot_raw:
        rid = rec.get("run_id")
        dupes[rid] = dupes.get(rid, 0) + 1
    repeated = {rid: n for rid, n in dupes.items() if n > 1}
    if repeated:
        shown = ", ".join(f"{rid!r}×{n}" for rid, n in list(repeated.items())[:3])
        raise DataIntegrityError(
            f"{len(repeated)} cot run_id(s) appear more than once in raw.jsonl: "
            f"{shown}{' …' if len(repeated) > 3 else ''}. The runner's resume "
            f"writes one line per run_id (CONTRACTS.md §4), so a repeat means "
            f"two runs were concatenated or resume was bypassed. Each repeat "
            f"adds another sample to that cell's rate. De-duplicate raw.jsonl."
        )

    seen: set[str] = set()
    for rec in ann:
        rid = rec.get("run_id")
        if rid in seen:
            raise DataIntegrityError(
                f"annotated.jsonl has more than one line for run_id {rid!r}. "
                f"CONTRACTS.md §5 requires one line per run_id: nothing "
                f"downstream can tell a re-annotation from a double-write, and "
                f"a duplicate silently doubles that cell's weight. Re-annotate "
                f"into a new file instead of appending."
            )
        seen.add(str(rid))

    ann_by_id = {str(r.get("run_id")): r for r in ann}

    rows = []
    for rec in cot_raw:
        rid = str(rec.get("run_id"))
        a = ann_by_id.get(rid)
        if a is None:
            continue
        rows.append({
            "run_id": rid,
            "model": rec.get("model"),
            "tier": rec.get("tier"),
            "scenario": rec.get("scenario"),
            "persona": rec.get("persona"),
            "temperature": float(rec.get("temperature")),
            "replicate": rec.get("replicate"),
            "provider": rec.get("provider"),
            "answer": a.get("answer"),
            "depth": a.get("depth"),
            "flag_no_basis": bool(a.get("flag_no_basis")),
            "flag_rigged": bool(a.get("flag_rigged")),
            "annotator_model": a.get("annotator_model"),
            "parse_ok": a.get("parse_ok"),
        })

    df = pd.DataFrame(rows)
    report = JoinReport(
        n_raw_total=len(raw),
        n_raw_cot=len(cot_raw),
        n_raw_direct=modes.get("direct", 0),
        n_raw_other_modes={m: n for m, n in modes.items() if m not in ("cot", "direct")},
        n_annotated=len(ann),
        n_joined=len(df),
        n_annotated_without_raw=sum(1 for r in ann if r.get("run_id") not in raw_ids),
        n_cot_without_annotation=len(cot_raw) - len(df),
    )

    if not df.empty:
        _validate_labels(df)
    return df, report


def _validate_labels(df: pd.DataFrame) -> None:
    """Every answer and depth code must be one the contract declares.

    An off-schema label is an annotator bug. Silently folding it into
    ``UNPARSEABLE`` would hide exactly the failure PLAN.md §4 wants hand-reviewed.
    """
    bad_answers = sorted(set(df["answer"]) - set(contracts.ANSWERS))
    bad_depths = sorted(set(df["depth"]) - set(contracts.DEPTH_CODES))
    if bad_answers or bad_depths:
        raise DataIntegrityError(
            f"off-schema annotation labels: answers={bad_answers} "
            f"depths={bad_depths}. Allowed answers {list(contracts.ANSWERS)}, "
            f"allowed depths {list(contracts.DEPTH_CODES)} (CONTRACTS.md §6)."
        )


def model_order(cfg: Mapping, present: Iterable[str]) -> list[str]:
    """Config order first (CONTRACTS.md §1: "Order is canonical for plots"),
    then anything else in the data, sorted, so an unexpected model is visible
    rather than dropped."""
    configured = [m["id"] for m in cfg.get("models", [])]
    present = set(present)
    ordered = [m for m in configured if m in present]
    return ordered + sorted(present - set(ordered))


# --------------------------------------------------------------- estimand ----

@dataclass(frozen=True)
class Rate:
    """A rate with a cluster-bootstrap 95% interval, and its bookkeeping."""

    point: float
    lo: float
    hi: float
    n_units: int
    n_cells: int
    n_samples: int
    unit: tuple[str, ...]
    n_boot: int
    n_cells_undefined: int = 0

    @property
    def degenerate(self) -> bool:
        """One cluster: there is no interval to report.

        Every resample of a single unit returns that unit, so the percentiles
        land exactly on the point estimate. Printed as ``0.571 [0.571, 0.571]``
        that reads as certainty — the precise failure PLAN.md §8.5 warns about,
        taken to its limit. It is not an exotic shape either: PLAN.md §9 step 5's
        pilot (one persona, one temperature, "the one not to skip") has it.
        ``bootstrap_ci`` returns NaN bounds here so a plot cannot draw a
        zero-height error bar and this cannot print as a real interval.
        """
        return self.n_units < 2

    def fmt(self, decimals: int = 3) -> str:
        if not np.isfinite(self.point):
            return "n/a"
        if self.degenerate or not (np.isfinite(self.lo) and np.isfinite(self.hi)):
            unit = "×".join(self.unit) or "?"
            return (f"{self.point:.{decimals}f} "
                    f"[no CI: {self.n_units} {unit} cluster]")
        return (f"{self.point:.{decimals}f} "
                f"[{self.lo:.{decimals}f}, {self.hi:.{decimals}f}]")

    def as_dict(self) -> dict:
        return {
            "point": self.point, "lo": self.lo, "hi": self.hi,
            "n_units": self.n_units, "n_cells": self.n_cells,
            "n_samples": self.n_samples, "unit": "×".join(self.unit),
            "n_boot": self.n_boot, "n_cells_undefined": self.n_cells_undefined,
        }


def cell_values(df: pd.DataFrame, *,
                success: Sequence[str],
                cell_keys: Sequence[str] = CELL_KEYS,
                column: str = "answer",
                denominator: Sequence[str] | None = None) -> pd.DataFrame:
    """Collapse samples to one row per cell: ``k``, ``n``, ``value = k / n``.

    This is the *only* place individual samples are counted, and the frame it
    returns is what every estimator downstream consumes. ``denominator=None``
    means unconditional (every answer counts toward ``n``); passing
    ``BINARY_ANSWERS`` gives the conditional rate the analytic curve must be
    compared against.

    A cell whose denominator is empty gets ``value = NaN`` rather than 0 — a
    cell where every response was ``REFUSE`` has no conditional KEEP rate, and
    calling that zero would invent a finding.
    """
    keys = list(cell_keys)
    if df.empty:
        return pd.DataFrame(columns=keys + ["k", "n", "n_all", "value"])

    frame = df
    if denominator is not None:
        frame = df[df[column].isin(list(denominator))]

    n_all = df.groupby(keys, observed=True, dropna=False).size().rename("n_all")
    n = (frame.groupby(keys, observed=True, dropna=False).size().rename("n")
         if not frame.empty else pd.Series(dtype=int, name="n"))
    hit = frame[frame[column].isin(list(success))] if not frame.empty else frame
    k = (hit.groupby(keys, observed=True, dropna=False).size().rename("k")
         if not hit.empty else pd.Series(dtype=int, name="k"))

    out = (pd.concat([n_all, n, k], axis=1)
           .reindex(n_all.index)
           .fillna(0)
           .reset_index())
    out["n"] = out["n"].astype(int)
    out["k"] = out["k"].astype(int)
    out["n_all"] = out["n_all"].astype(int)
    out["value"] = np.where(out["n"] > 0, out["k"] / out["n"].where(out["n"] > 0), np.nan)
    return out[keys + ["k", "n", "n_all", "value"]]


def _require_cell_level(cells: pd.DataFrame, cell_keys: Sequence[str]) -> None:
    tells = [c for c in _SAMPLE_LEVEL_TELLS if c in cells.columns]
    if tells:
        raise ClusteringError(
            f"refusing to estimate from a sample-level frame (columns {tells} "
            f"present). PLAN.md §8.5: 25 replicates from one model at one "
            f"temperature are 25 draws from ONE distribution, not 25 subjects. "
            f"Pooling them makes n look like {len(cells)} when the real number "
            f"of independent units is the number of models. Aggregate with "
            f"cell_values(...) first, then estimate."
        )
    missing = [c for c in cell_keys if c not in cells.columns]
    if missing:
        raise ClusteringError(
            f"cell frame is missing cell keys {missing}; cannot tell cells apart, "
            f"so cannot cluster. Expected {list(cell_keys)}."
        )
    dup = cells.duplicated(subset=list(cell_keys)).sum()
    if dup:
        raise ClusteringError(
            f"{dup} duplicate rows for the same {list(cell_keys)} cell. Either "
            f"these are sample-level rows (aggregate with cell_values first) or "
            f"two runs were concatenated, in which case the affected cells carry "
            f"double weight. Either way the interval below would be wrong."
        )


def _require_cluster(cells: pd.DataFrame, unit: Sequence[str] | None) -> None:
    if not unit:
        raise ClusteringError(
            "no resampling unit given. An interval computed by pooling every "
            "sample across models is far too narrow (PLAN.md §8.5); name the "
            "cluster explicitly, e.g. unit=('model',)."
        )
    missing = [c for c in unit if c not in cells.columns]
    if missing:
        raise ClusteringError(
            f"cluster column(s) {missing} not in the cell frame, so models "
            f"cannot be resampled. Columns present: {list(cells.columns)}."
        )


def bootstrap_ci(cells: pd.DataFrame, *,
                 unit: Sequence[str] = ("model",),
                 value: str = "value",
                 cell_keys: Sequence[str] = CELL_KEYS,
                 n_boot: int = N_BOOT_DEFAULT,
                 seed: int = 0,
                 alpha: float = ALPHA) -> Rate:
    """Cluster bootstrap over ``unit`` — by default, over models.

    **Why pooling samples across models would be wrong.** PLAN.md §8.5 is blunt
    about this and it is the one statistical point that decides whether a
    reader keeps reading. 25 replicates from one model at one temperature are
    *not* 25 independent subjects; they are 25 draws from a single distribution
    whose parameters that model fixes. Pooling them treats a sample of five
    models as a sample of 5,625, and the standard error of a proportion shrinks
    as 1/sqrt(n): the interval comes out roughly sqrt(5625/5) ≈ 34× too narrow.
    Worse, the error is not conservative in a harmless direction — every
    between-model difference becomes "significant", including differences that
    are entirely an artefact of which provider served which model, which
    quantization it was serving, and how that provider implements temperature.
    The replicates measure *within-model* sampling noise, which is real but is
    not the quantity a claim about models is about.

    So: collapse samples to cells (``cell_values``), average cells within a
    unit, and resample **units** with replacement. With ``unit=('model',)`` the
    point estimate is the unweighted mean of the per-model means — the cluster
    mean estimator — which gives every model one vote regardless of how many
    replicates it happens to have, and the interval reflects variation between
    models, which is what generalising over models requires.

    For a statistic about *one* model there are no models to resample, so the
    unit becomes that model's own cells (``unit=('scenario', 'persona',
    'temperature')`` or whatever remains): the cell is still the unit of
    analysis, and the interval then reflects variation across conditions within
    that model. ``per_model_rate`` does this and labels it, so a within-model
    interval is never mistaken for a between-model one.

    Cells whose denominator is empty (``value`` is NaN) are excluded and
    counted in ``Rate.n_cells_undefined``; a conditional rate is genuinely
    undefined where nothing binary was answered.
    """
    _require_cell_level(cells, cell_keys)
    _require_cluster(cells, unit)

    unit = tuple(unit)
    usable = cells[np.isfinite(cells[value].to_numpy(dtype=float))] if len(cells) else cells
    n_undefined = len(cells) - len(usable)
    n_samples = int(usable["n"].sum()) if "n" in usable.columns and len(usable) else 0

    if usable.empty:
        return Rate(np.nan, np.nan, np.nan, 0, 0, 0, unit, n_boot, n_undefined)

    unit_means = np.asarray(
        usable.groupby(list(unit), observed=True, dropna=False)[value].mean(),
        dtype=float,
    )
    m = unit_means.size
    point = float(unit_means.mean())

    if m < 2:
        # One cluster: between-unit variation is not estimable. Every resample
        # returns the same unit, so the percentiles collapse and `lo == point
        # == hi` would print as a 95% interval asserting certainty — the limit
        # case of the "far too narrow" interval PLAN.md §8.5 warns about. NaN
        # bounds instead, so `Rate.fmt` says so and `_yerr` draws no whisker.
        return Rate(point, np.nan, np.nan, m, len(usable), n_samples, unit,
                    int(n_boot), n_undefined)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, m, size=(int(n_boot), m))
    reps = unit_means[draws].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(reps, [100 * alpha / 2,
                                                     100 * (1 - alpha / 2)]))
    return Rate(point, lo, hi, m, len(usable), n_samples, unit, int(n_boot), n_undefined)


def per_model_rate(cells: pd.DataFrame, model: str, *,
                   value: str = "value",
                   cell_keys: Sequence[str] = CELL_KEYS,
                   n_boot: int = N_BOOT_DEFAULT,
                   seed: int = 0,
                   alpha: float = ALPHA) -> Rate:
    """A single model's rate, with the interval taken over *its own cells*.

    A between-model cluster bootstrap is degenerate here — there is one model,
    so every resample returns it and the interval collapses to a point. The
    honest interval for a one-model claim is over the unit of analysis one level
    down: the model's (scenario, persona, temperature) cells. ``Rate.unit``
    records which it was, so the two never get confused in a table.
    """
    subset = cells[cells["model"] == model]
    inner = tuple(k for k in cell_keys if k != "model")
    return bootstrap_ci(subset, unit=inner, value=value, cell_keys=cell_keys,
                        n_boot=n_boot, seed=seed, alpha=alpha)


# ------------------------------------------------------------- flip rate ----

def flip_cells(cot: pd.DataFrame, *,
               n_boot: int = N_BOOT_DEFAULT) -> pd.DataFrame:
    """The headline statistic, one row per (model, persona, temperature).

        flip_rate = keep_rate(default_keep) - keep_rate(default_switch)

    **Signed, never absolute.** The sign *is* the finding. Positive means the
    model follows whichever action is the default in both framings — status quo
    bias, the model analogue of the 90/10 human result. Negative means it takes
    the action in both framings, which is the opposite behaviour and has already
    been observed in probing. An absolute value maps both onto the same number
    and throws the result away.

    ``keep_rate`` is conditional on ``{KEEP, SWITCH}`` (CONTRACTS.md §8) so the
    two scenarios sit on the same footing and neither is penalised for a
    different rate of refusals. ``isomorph`` is the contamination control and
    has no partner scenario, so it has no flip rate and is absent here.
    """
    cells = cell_values(cot, success=("KEEP",), denominator=BINARY_ANSWERS)
    wide = (cells[cells["scenario"].isin(FLIP_SCENARIOS)]
            .pivot_table(index=list(FLIP_CELL_KEYS), columns="scenario",
                         values=["value", "k", "n"], observed=True))
    if wide.empty:
        return pd.DataFrame(columns=list(FLIP_CELL_KEYS) +
                            ["keep_rate_default_keep", "keep_rate_default_switch",
                             "n_default_keep", "n_default_switch", "value"])

    out = pd.DataFrame(index=wide.index)
    for scenario in FLIP_SCENARIOS:
        out[f"keep_rate_{scenario}"] = wide[("value", scenario)] if ("value", scenario) in wide else np.nan
        # A cell the runner never reached in one scenario pivots to NaN. Its
        # *count* is genuinely 0 — and must stay an int, because a resumed run
        # killed halfway (CONTRACTS.md §4) is the normal way to get here. Its
        # keep_rate stays NaN, so `value` is NaN and bootstrap_ci excludes the
        # cell rather than scoring a one-sided contrast as zero flip.
        counts = wide[("n", scenario)] if ("n", scenario) in wide else np.nan
        out[f"n_{scenario}"] = pd.Series(counts, index=out.index).fillna(0).astype(int)
    out["value"] = (out[f"keep_rate_{FLIP_SCENARIOS[0]}"]
                    - out[f"keep_rate_{FLIP_SCENARIOS[1]}"])
    # `n` is what bootstrap_ci reports as the underlying sample count.
    out["n"] = out[[f"n_{s}" for s in FLIP_SCENARIOS]].sum(axis=1).astype(int)
    return out.reset_index()


# ----------------------------------------------------------------- plots ----

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

#: Categorical slots 1-5, in fixed order, never cycled (dataviz skill). Models
#: take them in config order, so a model keeps its colour when the set is
#: filtered. Markers are the secondary encoding: three of these slots sit below
#: 3:1 against the surface, so identity never rests on hue alone.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

#: Recursion depth is *ordinal* (L0 < L1 < L2 < L3), so L0-L3 is one hue
#: light->dark rather than four unrelated hues. MIX and NON are not on that
#: scale: MIX takes a categorical hue, NON the neutral, which is the same
#: convention as a diverging midpoint — grey reads as "off the scale".
DEPTH_COLORS = {
    "L0": "#86b6ef", "L1": "#3987e5", "L2": "#1c5cab", "L3": "#0d366b",
    "MIX": "#eb6834", "NON": "#898781",
}

#: Diverging pair for the signed flip rate: a real zero with two opposite
#: meanings on either side.
POS_COLOR = "#2a78d6"
NEG_COLOR = "#e34948"

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.bbox": "tight", "savefig.dpi": 200,
    "font.size": 7.0, "axes.titlesize": 7.5, "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.0,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 3.2,
    "legend.frameon": False, "figure.titlesize": 8.0,
    "axes.titlelocation": "left", "axes.titlepad": 3.0,
}

#: Single-column width, in inches. Figures are laid out to be legible at this
#: width, so multi-panel figures stack vertically rather than side by side.
COL_W = 3.4


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


#: Characters that fit across COL_W at the caption's 5.6pt. matplotlib's own
#: `wrap=True` measures against the axes, not the figure, and overflows.
_CAPTION_CHARS = 84


def _caption(fig, text: str) -> None:
    fig.text(0.0, -0.02, textwrap.fill(" ".join(text.split()), _CAPTION_CHARS),
             ha="left", va="top", fontsize=5.6, color=INK_2, linespacing=1.4,
             transform=fig.transFigure)


def _style_axes(ax) -> None:
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)


def data_temperatures(cot: pd.DataFrame, temps: Sequence[float]) -> list[float]:
    """Temperatures actually sampled in cot, in config order.

    The config's ``temperatures`` are what the *analytic* curve is evaluated on
    — it is free at any T — but the empirical series exists only where the
    runner actually sampled. Plotting the empirical series on the config grid
    puts NaN gaps in it, which matplotlib renders as a broken line and a reader
    renders as "the rate went to zero".
    """
    present = {float(t) for t in cot["temperature"]} if not cot.empty else set()
    return sorted(present)


def _yerr(r: "Rate | None") -> tuple[float, float]:
    """Asymmetric error-bar offsets for one Rate, NaN when there is no interval.

    matplotlib skips a NaN whisker entirely. That is the point: a single-cluster
    Rate has no estimable interval (see ``Rate.degenerate``), and drawing it as a
    zero-height bar would assert certainty. No bar at all is the honest mark.
    """
    if r is None or not np.isfinite(r.point) or not (np.isfinite(r.lo) and np.isfinite(r.hi)):
        return float("nan"), float("nan")
    return r.point - r.lo, r.hi - r.point


def plot_keep_vs_temperature(cot: pd.DataFrame, direct: pd.DataFrame,
                             models: Sequence[str], temps: Sequence[float],
                             report: curve_mod.DirectModeReport, out: Path, *,
                             n_boot: int, seed: int) -> Path:
    """Plot 1 — conditional KEEP rate vs temperature, one panel per scenario.

    Solid: empirical cot. Dashed, same colour: the analytic direct-mode curve
    for that model, where one exists. **Both series use the conditional
    denominator** ``n(KEEP) / (n(KEEP) + n(SWITCH))`` — the analytic curve
    renormalises over ``{K, S}``, so an unconditional cot rate would sit below
    it by exactly the non-binary fraction and the gap would be pure
    bookkeeping. The excluded fraction is plot 4.
    """
    plt = _pyplot()
    cells = cell_values(cot, success=("KEEP",), denominator=BINARY_ANSWERS)
    scenarios = [s for s in contracts.SCENARIOS if s in set(cot["scenario"])] or list(contracts.SCENARIOS)
    # Filled in below and returned through `run()`, so which models got an
    # overlay is a value a caller (or a test) can read rather than a claim only
    # the figure's caption makes. This used to be asserted, and the assert was
    # wrong; see the comment at `no_overlay`.
    overlay_report: dict[str, list[str]] = {"overlaid": [], "no_overlay": [],
                                            "scenarios": list(scenarios)}
    colors = {m: SERIES[i % len(SERIES)] for i, m in enumerate(models)}
    marks = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(models)}
    sampled = data_temperatures(cot, temps)
    ticks = sorted({*sampled, *(float(t) for t in temps)})

    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(scenarios), 1, figsize=(COL_W, 2.15 * len(scenarios)),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        overlaid: list[str] = []

        for ax, scenario in zip(axes, scenarios):
            _style_axes(ax)
            for model in models:
                # The dashed overlay is drawn FIRST and independently of the cot
                # series. They are two different measurements and either can be
                # missing on its own: a model can have a clean direct curve and
                # no cot rows (the annotator failed on it, or this is being run
                # between the two grids), or cot rows and no usable logprobs.
                # Nesting the overlay under the cot guard would drop a curve that
                # exists while the caption below still claimed it was drawn.
                dsub = direct[(direct["model"] == model) & (direct["scenario"] == scenario)]
                if not dsub.empty:
                    # The analytic curve is free at any T, so it keeps the full
                    # config grid even where cot was not sampled.
                    an = (dsub.groupby("temperature", observed=True)["p_keep"]
                          .mean().sort_index())
                    ax.plot(an.index, an.to_numpy(), color=colors[model],
                            linestyle=(0, (3, 1.6)), linewidth=1.0, alpha=0.85,
                            zorder=2, label=model)
                    overlaid.append(model)

                sub = cells[(cells["model"] == model) & (cells["scenario"] == scenario)]
                if sub.empty:
                    continue
                xs, ys, los, his = [], [], [], []
                for T in sampled:
                    at_T = sub[sub["temperature"] == float(T)]
                    if at_T.empty:
                        continue
                    r = bootstrap_ci(at_T, unit=("persona",), cell_keys=CELL_KEYS,
                                     n_boot=n_boot, seed=seed)
                    if not np.isfinite(r.point):
                        continue
                    xs.append(float(T)); ys.append(r.point)
                    dlo, dhi = _yerr(r); los.append(dlo); his.append(dhi)
                if not xs:
                    continue
                ax.errorbar(xs, ys, yerr=[los, his], color=colors[model],
                            marker=marks[model], markeredgecolor=SURFACE,
                            markeredgewidth=0.4, elinewidth=0.7, capsize=0,
                            alpha=0.95, zorder=3, label=model)

            ax.set_title(scenario, color=INK)
            ax.set_ylim(-0.04, 1.04)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.axhline(0.5, color=BASELINE, linewidth=0.6, linestyle=":", zorder=1)

        axes[len(scenarios) // 2].set_ylabel("P(KEEP | KEEP or SWITCH)")
        axes[-1].set_xlabel("temperature")
        axes[-1].set_xticks(ticks)

        # Across ALL panels, not just the first. A model can be absent from one
        # panel on a fully successful run: if its conditional denominator is
        # empty there (every response non-binary in that framing — exactly what
        # plot 4 exists to show), every T is skipped and it registers no handle.
        # Harvesting from axes[0] alone would leave its coloured series
        # unlabelled in the panels where it does appear.
        pairs: dict[str, object] = {}
        for panel in axes:
            h, lab = panel.get_legend_handles_labels()
            pairs.update(dict(zip(lab, h)))
        labels = [m for m in models if m in pairs]      # config order is canonical
        handles = [pairs[m] for m in labels]
        # Which models have no overlay, derived from what was actually DRAWN.
        # There are three reasons a model can lack one and the caption has to
        # cover all three, because it is read beside the figure:
        #   1. dropped by `direct_mode_report` — logprobs unusable;
        #   2. no direct records in raw.jsonl at all, which happens whenever a
        #      pinned provider is down (CONTRACTS.md §8 sets
        #      allow_fallbacks: False, and §4 says a failed call writes no line);
        #   3. direct records that exist only in scenarios this figure is not
        #      drawing, because the panel set comes from the cot side.
        # This used to read `[m for m in models if m not in set(direct["model"])]`
        # with an `assert` that the two counts agreed. That assert encodes an
        # invariant nothing guarantees — reason 3 breaks it, since the direct
        # frame's model set and its scenario set are independent — and it fired
        # as a bare AssertionError from the FIRST plot, so a partial provider
        # outage produced no figures and no summary.md at all. Under `python -O`
        # it was worse than useless: the assert vanished and the caption then
        # claimed an overlay that is visibly absent. `overlaid` is the honest
        # source, and it is the number the caption already claims to report.
        no_overlay = [m for m in models if m not in set(overlaid)]
        n_over = len(set(overlaid))
        overlay_report["overlaid"] = sorted(set(overlaid))
        overlay_report["no_overlay"] = list(no_overlay)
        note = (f"solid = cot · dashed = analytic direct curve "
                f"({n_over} of {len(models)} models)")
        axes[0].legend(handles, labels, loc="lower left", ncols=1,
                       bbox_to_anchor=(0.0, 1.06), title=note,
                       title_fontsize=5.8, alignment="left")

        dropped = ", ".join(no_overlay) or "none"
        unsampled = [t for t in ticks if t not in sampled]
        gap = (f" cot was not sampled at T={', '.join(f'{t:.1f}' for t in unsampled)}, "
               f"where only the analytic curve exists." if unsampled else "")
        _caption(fig, (
            "Plot 1. CONDITIONAL denominator: n(KEEP)/(n(KEEP)+n(SWITCH)), so the "
            "empirical cot series and the analytic direct curve are the same "
            "quantity. Error bars are 95% bootstrap over persona cells within a "
            "model (see summary.md for between-model intervals). The dashed "
            "overlay exists only for models with usable direct-mode logprobs; "
            f"no overlay for: {dropped}. Their cot lines are kept — the cot "
            "series is the primary one and exists for every model. Excluded "
            f"(non-binary) fraction is plot 4.{gap}"))
        path = out / "keep_vs_temperature.png"
        fig.savefig(path)
        plt.close(fig)
    plot_keep_vs_temperature.last_overlay_report = overlay_report
    return path


#: Set by the most recent `plot_keep_vs_temperature` call: which models got a
#: dashed analytic overlay, which did not, and in which scenario panels. Read by
#: `run()` and by the tests, so the caption's claim is checkable.
plot_keep_vs_temperature.last_overlay_report = {}


def plot_flip_rate(cot: pd.DataFrame, models: Sequence[str], out: Path, *,
                   n_boot: int, seed: int) -> Path:
    """Plot 2 — the headline.

    Top: unconditional KEEP rate per model in each of the two framings, the
    comparison PLAN.md §8.5 asks for. Bottom: the signed flip rate as
    CONTRACTS.md §8 defines it, which is conditional on ``{KEEP, SWITCH}`` so
    the two framings sit on the same footing. Both denominators are stated in
    the caption because they differ, deliberately.
    """
    plt = _pyplot()
    uncond = cell_values(cot, success=("KEEP",))
    flips = flip_cells(cot)
    x = np.arange(len(models), dtype=float)

    with plt.rc_context(RC):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(COL_W, 4.5),
                                       gridspec_kw={"height_ratios": [1.15, 1.0]})
        _style_axes(ax1); _style_axes(ax2)

        width = 0.26
        for j, scenario in enumerate(FLIP_SCENARIOS):
            pts, los, his = [], [], []
            for model in models:
                sub = uncond[(uncond["model"] == model) & (uncond["scenario"] == scenario)]
                r = (per_model_rate(sub, model, n_boot=n_boot, seed=seed)
                     if not sub.empty else None)
                pts.append(r.point if r else np.nan)
                dlo, dhi = _yerr(r); los.append(dlo); his.append(dhi)
            # 2px surface gap between adjacent bars: the edge is the surface.
            ax1.bar(x + (j - 0.5) * width, pts, width * 0.94, label=scenario,
                    color=SERIES[j], edgecolor=SURFACE, linewidth=0.8, zorder=3)
            ax1.errorbar(x + (j - 0.5) * width, pts, yerr=[los, his], fmt="none",
                         ecolor=INK_2, elinewidth=0.7, capsize=0, zorder=4)

        ax1.set_xticks(x); ax1.set_xticklabels([])
        ax1.set_ylim(0, 1.04); ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax1.set_ylabel("KEEP rate (unconditional)")
        ax1.set_title("KEEP rate by framing", color=INK)
        ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.04), ncols=2)

        pts, los, his = [], [], []
        for model in models:
            sub = flips[flips["model"] == model]
            r = (bootstrap_ci(sub, unit=("persona", "temperature"),
                              cell_keys=FLIP_CELL_KEYS, n_boot=n_boot, seed=seed)
                 if not sub.empty else None)
            pts.append(r.point if r else np.nan)
            dlo, dhi = _yerr(r); los.append(dlo); his.append(dhi)

        cols = [POS_COLOR if (np.isfinite(p) and p >= 0) else NEG_COLOR for p in pts]
        ax2.bar(x, pts, 0.34, color=cols, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax2.errorbar(x, pts, yerr=[los, his], fmt="none", ecolor=INK_2,
                     elinewidth=0.7, capsize=0, zorder=4)
        ax2.axhline(0.0, color=INK, linewidth=0.8, zorder=2)
        for xi, p in zip(x, pts):
            if np.isfinite(p):
                ax2.annotate(f"{p:+.2f}", (xi, p), textcoords="offset points",
                             xytext=(0, 3 if p >= 0 else -9), ha="center",
                             fontsize=5.8, color=INK_2)
        # Symmetric about zero, so the sign of the flip reads off the axis
        # rather than off the bar length. `finite` may be empty: a run with
        # neither default_keep nor default_switch (isomorph alone) has no flip
        # rate at all, and the panel is then correctly blank rather than warning.
        finite = np.abs(np.asarray(pts, dtype=float))
        finite = finite[np.isfinite(finite)]
        lim = max(0.25, float(finite.max()) * 1.45) if finite.size else 0.25
        ax2.set_ylim(-lim, lim)
        ax2.set_xticks(x)
        ax2.set_xticklabels([_short(m) for m in models], rotation=18,
                            ha="right", fontsize=5.6)
        ax2.set_xlim(-0.6, len(models) - 0.4)
        ax1.set_xlim(-0.6, len(models) - 0.4)
        ax2.set_ylabel("flip rate (signed)")
        ax2.set_title("flip rate = KEEP(default_keep) − KEEP(default_switch)", color=INK)

        _caption(fig, (
            "Plot 2. TOP panel: UNCONDITIONAL KEEP rate (denominator = all "
            "answers, including RANDOMISE / THIRD_OPTION / REFUSE / "
            "UNPARSEABLE). BOTTOM panel: the flip rate as CONTRACTS.md §8 "
            "defines it, from CONDITIONAL keep rates n(KEEP)/(n(KEEP)+n(SWITCH)), "
            "so neither framing is penalised for a different refusal rate. The "
            "flip rate is SIGNED: positive (blue) = follows the default action "
            "in both framings (status quo bias); negative (red) = takes the "
            "action in both. Error bars are 95% bootstrap over that model's own "
            "cells; isomorph has no partner scenario and so no flip rate."))
        path = out / "flip_rate.png"
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_depth_by_persona(cot: pd.DataFrame, out: Path, *,
                          n_boot: int, seed: int) -> Path:
    """Plot 3 — depth codes by persona, stacked.

    Pooled over models with **cells weighted equally**: the share of each depth
    code is computed per (model, scenario, persona, temperature) cell, then
    averaged, so a model does not gain influence by having more replicates.
    Unconditional denominator (every cot response has a depth code; an
    unparseable annotation is ``NON``, not a missing value).

    **``contracts.NO_COT_MODELS`` are excluded from the bars**, as CONTRACTS.md
    §8 requires: a model that answers in two tokens has no reasoning to code, so
    its depth codes are uniformly L0/NON and are a property of its verbosity
    rather than of the persona. Pooling them in drags every bar toward L0 by a
    fixed fraction — with the shipped five-model config, one whole vote in five —
    and that artefact reads as a finding. They keep their row in summary.md's
    per-model depth table, which is where the contract wants them reported.
    """
    plt = _pyplot()
    personas = [p for p in contracts.PERSONAS if p in set(cot["persona"])]
    shares: dict[str, list[float]] = {code: [] for code in contracts.DEPTH_CODES}

    no_cot = sorted(set(cot["model"]) & set(contracts.NO_COT_MODELS))
    codeable = cot[~cot["model"].isin(list(contracts.NO_COT_MODELS))]
    n_shown = int(codeable["model"].nunique())
    if codeable.empty:
        # Nothing left to plot honestly. Draw the empty panel and say why,
        # rather than silently falling back to the pooled version the contract
        # forbids.
        codeable = cot.iloc[0:0]

    for persona in personas:
        sub = codeable[codeable["persona"] == persona]
        for code in contracts.DEPTH_CODES:
            cells = cell_values(sub, success=(code,), column="depth")
            r = bootstrap_ci(cells, unit=("model",), n_boot=n_boot, seed=seed)
            shares[code].append(r.point if np.isfinite(r.point) else 0.0)

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(COL_W, 2.7))
        _style_axes(ax)
        x = np.arange(len(personas), dtype=float)
        bottom = np.zeros(len(personas))
        for code in contracts.DEPTH_CODES:
            vals = np.asarray(shares[code], dtype=float)
            # 2px surface gap between stacked segments.
            ax.bar(x, vals, 0.38, bottom=bottom, label=code,
                   color=DEPTH_COLORS[code], edgecolor=SURFACE, linewidth=0.9,
                   zorder=3)
            for xi, v, b in zip(x, vals, bottom):
                if v >= 0.09:                      # selective direct labels only
                    ax.annotate(f"{v:.0%}", (xi, b + v / 2), ha="center",
                                va="center", fontsize=5.6,
                                color=SURFACE if code in ("L1", "L2", "L3") else INK)
            bottom = bottom + vals

        ax.set_xticks(x)
        ax.set_xticklabels(personas, fontsize=6.4)
        ax.set_xlim(-0.55, len(personas) - 0.45)
        ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("share of cot responses")
        title = "recursion depth by persona"
        if no_cot:
            title += f"  ({n_shown} of {cot['model'].nunique()} models)"
        ax.set_title(title, color=INK)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), ncols=1,
                  title="depth", title_fontsize=5.8, alignment="left")

        note = ""
        if no_cot:
            note = (f" EXCLUDED from these bars, per CONTRACTS.md §8: "
                    f"{', '.join(no_cot)} — no visible reasoning to code, so "
                    f"their uniform L0/NON would drag every bar toward L0 and "
                    f"look like a persona effect. summary.md reports depth per "
                    f"model, including theirs.")
        _caption(fig, (
            "Plot 3. UNCONDITIONAL denominator: every cot response carries a "
            "depth code, and an unparseable annotation is NON rather than "
            "missing. Shares are computed per (model × scenario × persona × "
            "temperature) cell and pooled with CELLS WEIGHTED EQUALLY, then "
            f"averaged over the {n_shown} model(s) with codeable reasoning, one "
            "vote each. L0-L3 is one hue "
            "light-to-dark because depth is ordinal; MIX and NON are not on "
            "that scale." + note))
        path = out / "depth_by_persona.png"
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_answers_by_role(cot: pd.DataFrame, models: Sequence[str], out: Path) -> Path:
    """Plot 5 — plain KEEP/SWITCH counts per model, coloured by role.

    One panel per scenario, because scenario IS the manipulation and pooling it
    would hide the flip the study exists to measure. Within a panel: one bar
    pair per (model, persona) — KEEP solid, SWITCH hatched in the same hue — so
    the two answers are read directly rather than inferred from one bar and a
    remainder.

    Deliberately **unconditional**: the bars are counts, and RANDOMISE /
    THIRD_OPTION / REFUSE / UNPARSEABLE are drawn as a third neutral segment
    rather than dropped. A plain count plot that silently discards a fifth of
    its data is the kind of figure that gets a paper rejected.
    """
    plt = _pyplot()
    scenarios = [s for s in contracts.SCENARIOS if s in set(cot["scenario"])]
    personas = [p for p in contracts.PERSONAS if p in set(cot["persona"])]
    if not scenarios or not personas or cot.empty:
        return out / "answers_by_role.png"
    colors = {p: SERIES[i % len(SERIES)] for i, p in enumerate(personas)}

    with plt.rc_context(RC):
        # sharex too: the models are identical in every panel, so only the
        # bottom one is labelled. Labelling each one puts rotated model names
        # through the title of the panel below.
        fig, axes = plt.subplots(len(scenarios), 1,
                                 figsize=(COL_W, 2.35 * len(scenarios)),
                                 sharey=True, sharex=True)
        axes = np.atleast_1d(axes)
        width = 0.36
        for ax, scenario in zip(axes, scenarios):
            sub = cot[cot["scenario"] == scenario]
            ticks, labels = [], []
            slot = 0.0
            for model in models:
                start = slot
                for persona in personas:
                    g = sub[(sub["model"] == model) & (sub["persona"] == persona)]
                    n = len(g)
                    if n:
                        keep = int((g["answer"] == "KEEP").sum())
                        switch = int((g["answer"] == "SWITCH").sum())
                        other = n - keep - switch
                        ax.bar(slot, keep / n, width, color=colors[persona],
                               edgecolor=SURFACE, linewidth=0.7, zorder=3)
                        ax.bar(slot + width, switch / n, width, color=colors[persona],
                               edgecolor=SURFACE, linewidth=0.7, alpha=0.5,
                               hatch="///", zorder=3)
                        if other:
                            ax.bar(slot + width, other / n, width,
                                   bottom=switch / n, color=MUTED, alpha=0.45,
                                   edgecolor=SURFACE, linewidth=0.7, zorder=3)
                    slot += 2 * width + 0.10
                ticks.append((start + slot - 2 * width - 0.10 + width) / 2)
                labels.append(model.split("/")[-1])
                slot += 0.34
            _style_axes(ax)
            ax.set_title(scenario)
            ax.set_ylim(0, 1)
            ax.set_ylabel("share of responses")
            ax.set_xticks(ticks)
        axes[-1].set_xticklabels(labels, rotation=18, ha="right")
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in personas]
        handles += [plt.Rectangle((0, 0), 1, 1, color=INK_2, alpha=0.5, hatch="///"),
                    plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=0.45)]
        axes[0].legend(handles, list(personas) + ["(hatched) SWITCH", "other answers"],
                       ncol=2, loc="upper left", bbox_to_anchor=(0, 1.42))
        _caption(fig, "Plot 5. Left bar KEEP, right bar SWITCH, hue = role. "
                      "Unconditional: RANDOMISE / THIRD_OPTION / REFUSE / UNPARSEABLE "
                      "stack on the SWITCH bar in grey rather than being dropped. "
                      "One panel per scenario; the KEEP bars moving between panels "
                      "for one hue is the flip.")
        path = out / "answers_by_role.png"
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_keep_by_temperature(cot: pd.DataFrame, models: Sequence[str],
                             temps: Sequence[float], out: Path) -> Path:
    """Plot 6 — plain conditional KEEP rate vs temperature, one line per model.

    Plot 1 answers "does the empirical curve match the analytic one" and carries
    a dashed overlay per model to do it. This answers the plainer question —
    what does each model do as temperature rises — and omits the overlay so the
    empirical lines are not read against a reference that only exists for some
    models. Same conditional denominator as plot 1, so the two are comparable.
    """
    plt = _pyplot()
    scenarios = [s for s in contracts.SCENARIOS if s in set(cot["scenario"])]
    sampled = data_temperatures(cot, temps)
    if not scenarios or not sampled or cot.empty:
        return out / "keep_by_temperature.png"
    cells = cell_values(cot, success=("KEEP",), denominator=BINARY_ANSWERS)
    colors = {m: SERIES[i % len(SERIES)] for i, m in enumerate(models)}
    marks = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(models)}

    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(scenarios), 1,
                                 figsize=(COL_W, 2.1 * len(scenarios)),
                                 sharex=True, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, scenario in zip(axes, scenarios):
            ax.axhline(0.5, color=BASELINE, linewidth=0.8, zorder=1)
            for model in models:
                pts = []
                for t in sampled:
                    g = cells[(cells["model"] == model) & (cells["scenario"] == scenario)
                              & (cells["temperature"] == t)]
                    # Cells are (model x scenario x persona x temperature); one
                    # point per temperature means averaging the personas with
                    # equal weight, not pooling their samples — a persona with
                    # more usable answers must not count for more.
                    vals = g["value"].dropna()
                    pts.append(float(vals.mean()) if len(vals) else float("nan"))
                ax.plot(sampled, pts, marker=marks[model], color=colors[model],
                        label=model.split("/")[-1], zorder=3)
            _style_axes(ax)
            ax.set_title(scenario)
            ax.set_ylim(-0.02, 1.02)
            ax.set_ylabel("P(KEEP | KEEP or SWITCH)")
            ax.set_xticks(sampled)
        axes[-1].set_xlabel("temperature")
        axes[0].legend(ncol=2, loc="upper left", bbox_to_anchor=(0, 1.34))
        _caption(fig, "Plot 6. Conditional KEEP rate, personas averaged with equal "
                      "weight per cell. No analytic overlay — that is plot 1. The "
                      "line at 0.5 is indifference; SWITCH rate is one minus this.")
        path = out / "keep_by_temperature.png"
        fig.savefig(path)
        plt.close(fig)
    return path


def plot_nonbinary_by_temperature(cot: pd.DataFrame, temps: Sequence[float],
                                  out: Path, *, n_boot: int, seed: int) -> Path:
    """Plot 4 — non-binary answer rates vs temperature.

    Also carries the **excluded fraction**: the total of these four rates is
    exactly what plot 1's conditional denominator leaves out, which is why
    nothing is lost by conditioning there. Unconditional denominator.
    """
    plt = _pyplot()
    xs = data_temperatures(cot, temps)

    def series(success: Sequence[str]) -> tuple[list[float], list[float], list[float]]:
        pts, lo, hi = [], [], []
        for T in xs:
            r = bootstrap_ci(cell_values(cot[cot["temperature"] == T], success=success),
                             unit=("model",), n_boot=n_boot, seed=seed)
            pts.append(r.point); lo.append(r.lo); hi.append(r.hi)
        return pts, lo, hi

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(COL_W, 2.7))
        _style_axes(ax)

        for i, answer in enumerate(NONBINARY_ANSWERS):
            pts, lo, hi = series((answer,))
            # Bands, not error bars: five series of vertical whiskers at three
            # shared x positions is unreadable, and with only three clusters
            # the intervals are wide enough to swamp the marks.
            ax.fill_between(xs, lo, hi, color=SERIES[i], alpha=0.10,
                            linewidth=0, zorder=2)
            ax.plot(xs, pts, color=SERIES[i], marker=MARKERS[i],
                    markeredgecolor=SURFACE, markeredgewidth=0.4,
                    label=answer, zorder=3)

        pts, lo, hi = series(NONBINARY_ANSWERS)
        ax.fill_between(xs, lo, hi, color=INK, alpha=0.07, linewidth=0, zorder=2)
        ax.plot(xs, pts, color=INK, marker="D", markersize=2.6, linestyle="--",
                linewidth=1.1, markeredgecolor=SURFACE, markeredgewidth=0.4,
                label="excluded from plot 1", zorder=4)

        ax.set_xticks(xs)
        ax.set_xlabel("temperature")
        ax.set_ylabel("rate (unconditional)")
        ax.set_ylim(bottom=0)
        ax.set_title("non-binary answers by temperature", color=INK)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.02), ncols=1,
                  alignment="left")
        _caption(fig, (
            "Plot 4. UNCONDITIONAL denominator: all cot answers in the cell. "
            "Rates are per-model cell means pooled over models with one vote "
            "each; bands are 95% cluster bootstrap over models. The dashed black "
            "series is the sum of the four, which is exactly the fraction plot "
            "1's conditional denominator excludes. UNPARSEABLE is a QA flag, "
            "not a result: PLAN.md §4 targets under 2% and wants every instance "
            "hand-reviewed. x shows only the temperatures cot was sampled at."))
        path = out / "nonbinary_by_temperature.png"
        fig.savefig(path)
        plt.close(fig)
    return path


def _short(model: str) -> str:
    return model.split("/", 1)[-1]


# --------------------------------------------------------------- summary ----

def write_summary(path: Path, *, cfg: Mapping, cot: pd.DataFrame,
                  direct: pd.DataFrame, models: Sequence[str],
                  temps: Sequence[float], join: JoinReport,
                  direct_report: curve_mod.DirectModeReport,
                  n_boot: int, seed: int) -> Path:
    """Numbers only. The interpretation is the human's."""
    L: list[str] = []
    w = L.append

    w("# Summary — two-bottle budget grid")
    w("")
    w(f"Experiment `{cfg.get('experiment')}` · "
      f"{len(models)} models · seed {seed} · "
      f"{n_boot} bootstrap iterations · 95% intervals.")
    w("")
    w("Plain numbers only; no interpretation.")
    w("")

    w("## Method")
    w("")
    w("- Unit of analysis: the (model × scenario × persona × temperature) cell. "
      "Samples are collapsed to cells before anything is estimated.")
    w("- Intervals labelled **between models** resample models with replacement "
      f"({n_boot} iterations) and average the per-model cell means, one vote per "
      "model. Intervals labelled **within model** resample that model's own "
      "cells, because a single-model statistic has no models to resample.")
    w("- `conditional` = n(KEEP) / (n(KEEP) + n(SWITCH)). `unconditional` = "
      "denominator includes RANDOMISE, THIRD_OPTION, REFUSE and UNPARSEABLE.")
    w("")

    w("## Data")
    w("")
    for line in join.lines():
        w(f"- {line}")
    w("")

    # ---- KEEP rate per (model, scenario) ----
    uncond = cell_values(cot, success=("KEEP",))
    cond = cell_values(cot, success=("KEEP",), denominator=BINARY_ANSWERS)
    w("## KEEP rate per (model, scenario)")
    w("")
    w("Within-model intervals over that model's (persona × temperature) cells.")
    w("")
    w("| model | scenario | KEEP unconditional [95% CI] | KEEP conditional [95% CI] | n samples | cells |")
    w("|---|---|---|---|---|---|")
    for model in models:
        for scenario in [s for s in contracts.SCENARIOS if s in set(cot["scenario"])]:
            u = uncond[(uncond["model"] == model) & (uncond["scenario"] == scenario)]
            c = cond[(cond["model"] == model) & (cond["scenario"] == scenario)]
            if u.empty:
                continue
            ru = per_model_rate(u, model, n_boot=n_boot, seed=seed)
            rc = per_model_rate(c, model, n_boot=n_boot, seed=seed)
            w(f"| `{model}` | {scenario} | {ru.fmt()} | {rc.fmt()} | "
              f"{int(u['n_all'].sum())} | {ru.n_cells} |")
    w("")

    w("### Pooled over models (one vote per model, between-model interval)")
    w("")
    w("| scenario | KEEP unconditional [95% CI] | KEEP conditional [95% CI] | models |")
    w("|---|---|---|---|")
    for scenario in [s for s in contracts.SCENARIOS if s in set(cot["scenario"])]:
        u = uncond[uncond["scenario"] == scenario]
        c = cond[cond["scenario"] == scenario]
        ru = bootstrap_ci(u, unit=("model",), n_boot=n_boot, seed=seed)
        rc = bootstrap_ci(c, unit=("model",), n_boot=n_boot, seed=seed)
        w(f"| {scenario} | {ru.fmt()} | {rc.fmt()} | {ru.n_units} |")
    w("")

    # ---- flip rates ----
    flips = flip_cells(cot)
    w("## Flip rate — the headline")
    w("")
    w("`flip_rate = keep_rate(default_keep) − keep_rate(default_switch)`, "
      "conditional on {KEEP, SWITCH} (CONTRACTS.md §8). **Signed**: positive "
      "means the model follows the default action in both framings; negative "
      "means it takes the action in both. `isomorph` has no partner scenario "
      "and so no flip rate.")
    w("")
    w("### Per model (within-model interval over its persona × temperature cells)")
    w("")
    w("| model | flip rate [95% CI] | KEEP cond. default_keep | KEEP cond. default_switch | cells |")
    w("|---|---|---|---|---|")
    for model in models:
        sub = flips[flips["model"] == model]
        if sub.empty:
            continue
        r = bootstrap_ci(sub, unit=("persona", "temperature"),
                         cell_keys=FLIP_CELL_KEYS, n_boot=n_boot, seed=seed)
        w(f"| `{model}` | {r.fmt()} | "
          f"{sub['keep_rate_default_keep'].mean():.3f} | "
          f"{sub['keep_rate_default_switch'].mean():.3f} | {r.n_cells} |")
    w("")
    pooled = bootstrap_ci(flips, unit=("model",), cell_keys=FLIP_CELL_KEYS,
                          n_boot=n_boot, seed=seed)
    w(f"Pooled over models (between-model interval): **{pooled.fmt()}** "
      f"over {pooled.n_units} models, {pooled.n_cells} cells.")
    w("")
    w("### Per (model, persona, temperature) cell")
    w("")
    w("| model | persona | temperature | flip rate | n(K or S) default_keep | default_switch |")
    w("|---|---|---|---|---|---|")
    for row in (flips.sort_values(list(FLIP_CELL_KEYS))
                .itertuples(index=False)):
        # A one-sided cell has no flip rate. Printing it as n/a beside its two
        # counts shows *which* scenario is missing, which a bare gap would not.
        flip = f"{row.value:+.3f}" if np.isfinite(row.value) else "n/a"
        w(f"| `{row.model}` | {row.persona} | {row.temperature:.1f} | "
          f"{flip} | {int(row.n_default_keep)} | "
          f"{int(row.n_default_switch)} |")
    w("")

    # ---- non-binary + unparseable ----
    w("## Non-binary answers (unconditional)")
    w("")
    w("| answer | rate pooled over models [95% CI] | n cells |")
    w("|---|---|---|")
    for answer in NONBINARY_ANSWERS:
        cells = cell_values(cot, success=(answer,))
        r = bootstrap_ci(cells, unit=("model",), n_boot=n_boot, seed=seed)
        w(f"| {answer} | {r.fmt(4)} | {r.n_cells} |")
    excl = bootstrap_ci(cell_values(cot, success=NONBINARY_ANSWERS),
                        unit=("model",), n_boot=n_boot, seed=seed)
    w(f"| **excluded from plot 1** | **{excl.fmt(4)}** | {excl.n_cells} |")
    w("")
    n_unp = int((cot["answer"] == "UNPARSEABLE").sum()) if not cot.empty else 0
    unp = bootstrap_ci(cell_values(cot, success=("UNPARSEABLE",)),
                       unit=("model",), n_boot=n_boot, seed=seed)
    w(f"UNPARSEABLE: {n_unp} of {len(cot)} joined cot records "
      f"({(n_unp / len(cot) if len(cot) else float('nan')):.4f} raw share); "
      f"pooled over models {unp.fmt(4)}. PLAN.md §4 targets under 2% and wants "
      f"every instance hand-reviewed.")
    w("")
    w("### UNPARSEABLE per model")
    w("")
    w("| model | n UNPARSEABLE | n cot records | share |")
    w("|---|---|---|---|")
    for model in models:
        sub = cot[cot["model"] == model]
        if sub.empty:
            continue
        k = int((sub["answer"] == "UNPARSEABLE").sum())
        w(f"| `{model}` | {k} | {len(sub)} | {k / len(sub):.4f} |")
    w("")

    # ---- direct mode ----
    w("## Direct mode")
    w("")
    w("```")
    for line in direct_report.lines():
        w(line)
    w("```")
    w("")
    if not direct_report.by_model.empty:
        w("| model | n direct records | " +
          " | ".join(curve_mod.STATUSES) + " | overlay |")
        w("|---|---|" + "---|" * (len(curve_mod.STATUSES) + 1))
        for row in direct_report.by_model.itertuples(index=False):
            drawn = "yes" if row.model in direct_report.usable_models else "no"
            w(f"| `{row.model}` | {row.n} | " +
              " | ".join(str(getattr(row, s)) for s in curve_mod.STATUSES) +
              f" | {drawn} |")
        w("")
    w(f"Records that could not feed the curve: **{direct_report.n_unusable}** of "
      f"{direct_report.n_records}. These failures cluster by provider and model, "
      f"not by condition, so the whole model is dropped from the overlay rather "
      f"than the individual cells — dropping cells would reweight the survivors.")
    w("")
    if not direct.empty:
        w("### Analytic conditional KEEP rate, P(KEEP | K or S)")
        w("")
        w("| model | scenario | " + " | ".join(f"T={T:.1f}" for T in temps) + " |")
        w("|---|---|" + "---|" * len(temps))
        for model in [m for m in models if m in set(direct["model"])]:
            for scenario in [s for s in contracts.SCENARIOS if s in set(direct["scenario"])]:
                sub = direct[(direct["model"] == model) & (direct["scenario"] == scenario)]
                if sub.empty:
                    continue
                per_T = sub.groupby("temperature", observed=True)["p_keep"].mean()
                cellvals = " | ".join(
                    f"{per_T.get(float(T), float('nan')):.3f}" for T in temps)
                w(f"| `{model}` | {scenario} | {cellvals} |")
        w("")
        w("Averaged over persona cells with equal weight. Conditional by "
          "construction: `rescale` renormalises over {K, S} only.")
        w("")

    # ---- depth ----
    # The SAME frame plot 3 draws from. CONTRACTS.md §8: "Report depth
    # distributions per model" and "Exclude NO_COT_MODELS from any claim about
    # reasoning", because "pooling qwen3-30b-a3b's uniform L0/NON into a
    # cross-model depth figure would drag every bar toward L0 and the effect
    # would look like a finding". This table used to be built from the
    # unfiltered `cot` while plot 3 filtered — so the figure and the table in
    # the same out/ directory printed two different depth distributions for the
    # identically-labelled quantity, differing by up to 23 percentage points,
    # with nothing to tell a reader which was which. The per-model table below
    # is where the excluded models belong, and it reports them with a
    # `visible cot?` column.
    codeable = cot[~cot["model"].isin(list(contracts.NO_COT_MODELS))]
    excluded = [m for m in models
                if m in contracts.NO_COT_MODELS and m in set(cot["model"])]

    w("## Depth distribution")
    w("")
    w("Shares are per-cell, pooled with cells weighted equally and one vote per "
      "model. Every cot response has a depth code; an unparseable annotation is "
      "`NON`.")
    w("")
    if excluded:
        w("**Excluded from every pooled figure in this section: "
          + ", ".join(f"`{m}`" for m in excluded)
          + "** — `contracts.NO_COT_MODELS`, which emit no visible reasoning to "
            "code (CONTRACTS.md §8). Their codes appear in the per-model table "
            "below and nowhere else. Plot 3 excludes the same models, so the "
            "figure and this table are the same quantity.")
        w("")
    if codeable.empty:
        w("_No model in this run emits codeable reasoning, so there is no pooled "
          "depth distribution._")
        w("")
    else:
        w("| grouping | " + " | ".join(contracts.DEPTH_CODES) + " |")
        w("|---|" + "---|" * len(contracts.DEPTH_CODES))
        for persona in [p for p in contracts.PERSONAS if p in set(codeable["persona"])]:
            sub = codeable[codeable["persona"] == persona]
            vals = []
            for code in contracts.DEPTH_CODES:
                r = bootstrap_ci(cell_values(sub, success=(code,), column="depth"),
                                 unit=("model",), n_boot=n_boot, seed=seed)
                vals.append(f"{r.point:.3f}" if np.isfinite(r.point) else "n/a")
            w(f"| persona={persona} | " + " | ".join(vals) + " |")
        w("")
    w("### Per model (never pooled — contracts.NO_COT_MODELS emit no visible "
      "reasoning, so their codes carry no information)")
    w("")
    w("| model | visible cot? | " + " | ".join(contracts.DEPTH_CODES) + " |")
    w("|---|---|" + "---|" * len(contracts.DEPTH_CODES))
    for model in models:
        sub = cot[cot["model"] == model]
        if sub.empty:
            continue
        has_cot = "no" if model in contracts.NO_COT_MODELS else "yes"
        counts = sub["depth"].value_counts()
        vals = [f"{counts.get(code, 0) / len(sub):.3f}" for code in contracts.DEPTH_CODES]
        w(f"| `{model}` | {has_cot} | " + " | ".join(vals) + " |")
    w("")
    w("### Flags")
    w("")
    # Same exclusion, same reason: both flags are read off the reasoning text
    # (PLAN.md §5 — "asserts there is no rational basis", "suspects a trick"),
    # so a model that emits no reasoning cannot raise either and would drag both
    # pooled rates toward zero.
    if excluded:
        w(f"Pooled over codeable models only; "
          + ", ".join(f"`{m}`" for m in excluded)
          + " excluded, as above — both flags are read off reasoning text.")
        w("")
    if codeable.empty:
        w("_No codeable model in this run._")
    else:
        for flag in ("flag_no_basis", "flag_rigged"):
            cells = _flag_cells(codeable, flag)
            r = bootstrap_ci(cells, unit=("model",), n_boot=n_boot, seed=seed)
            w(f"- `{flag}`: {r.fmt(3)} pooled over {r.n_units} models.")
        w("")
        w("Per model, so a model with no visible reasoning is visible as such "
          "rather than averaged away:")
        w("")
        w("| model | visible cot? | flag_no_basis | flag_rigged |")
        w("|---|---|---|---|")
        for model in models:
            sub = cot[cot["model"] == model]
            if sub.empty:
                continue
            has_cot = "no" if model in contracts.NO_COT_MODELS else "yes"
            w(f"| `{model}` | {has_cot} | "
              f"{sub['flag_no_basis'].mean():.3f} | {sub['flag_rigged'].mean():.3f} |")
    w("")

    w("## Figures")
    w("")
    w("| file | denominator |")
    w("|---|---|")
    w("| `keep_vs_temperature.png` | conditional — matches the analytic curve |")
    w("| `flip_rate.png` | bars unconditional; flip rate conditional |")
    w("| `depth_by_persona.png` | unconditional |")
    w("| `nonbinary_by_temperature.png` | unconditional |")
    w("| `answers_by_role.png` | unconditional — other answers shown, not dropped |")
    w("| `keep_by_temperature.png` | conditional — same denominator as plot 1 |")
    w("")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def _flag_cells(cot: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Cell-level rate of a boolean annotation flag, unconditional."""
    tmp = cot.assign(**{"_flag": np.where(cot[flag], "TRUE", "FALSE")})
    tmp = tmp.drop(columns=[c for c in _SAMPLE_LEVEL_TELLS if c in tmp.columns])
    tmp["answer"] = tmp["_flag"]
    return cell_values(tmp, success=("TRUE",))


# ------------------------------------------------------------------- run ----

def run(*, config: str | Path, raw: str | Path, annotated: str | Path,
        out: str | Path, n_boot: int = N_BOOT_DEFAULT,
        seed: int | None = None) -> dict:
    """Load, estimate, plot, write. Returns the paths and the two reports."""
    cfg = load_config(config)
    temps = [float(t) for t in cfg.get("temperatures", [])]
    seed = int(cfg.get("seed", 0)) if seed is None else int(seed)

    raw_records = read_jsonl(raw)
    annotated_records = read_jsonl(annotated)
    cot, join = load_cot(raw_records, annotated_records)
    if cot.empty:
        raise DataIntegrityError(
            f"no cot records survived the join of {raw} × {annotated}. "
            f"{'; '.join(join.lines())}"
        )

    direct_report = curve_mod.direct_mode_report(raw_records)
    direct = curve_mod.direct_curve(raw_records, temps)

    # The union, not just the CoT side. A model whose direct records all extract
    # cleanly but whose CoT rows are missing — an annotator that failed on one
    # model, or plotting between the direct grid and the CoT grid, which PLAN.md
    # §9 runs in that order — would otherwise lose its overlay without ever
    # appearing in `direct_report.dropped_models`, so summary.md would say the
    # curve exists in one table and omit it from the next. CONTRACTS.md §8:
    # "draw the overlay where it exists, say so on the figure."
    models = model_order(cfg, set(cot["model"]) | set(direct["model"]))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_keep_vs_temperature(cot, direct, models, temps, direct_report,
                                 out_dir, n_boot=n_boot, seed=seed),
        plot_flip_rate(cot, models, out_dir, n_boot=n_boot, seed=seed),
        plot_depth_by_persona(cot, out_dir, n_boot=n_boot, seed=seed),
        plot_nonbinary_by_temperature(cot, temps, out_dir, n_boot=n_boot, seed=seed),
        plot_answers_by_role(cot, models, out_dir),
        plot_keep_by_temperature(cot, models, temps, out_dir),
    ]
    summary = write_summary(out_dir / "summary.md", cfg=cfg, cot=cot,
                            direct=direct, models=models, temps=temps,
                            join=join, direct_report=direct_report,
                            n_boot=n_boot, seed=seed)

    return {"plots": paths, "summary": summary, "cot": cot, "direct": direct,
            "join": join, "direct_report": direct_report, "models": models,
            "overlay": getattr(plot_keep_vs_temperature, "last_overlay_report", {})}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.analyze",
        description="Aggregate and plot the two-bottle grid (PLAN.md §8.5).")
    ap.add_argument("--config", default=str(contracts.CONFIG_PATH))
    ap.add_argument("--raw", default=str(contracts.RAW_PATH))
    ap.add_argument("--annotated", default=str(contracts.ANNOTATED_PATH))
    ap.add_argument("--out", default=str(contracts.OUT_DIR))
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT,
                    help="bootstrap iterations (PLAN.md §8.5 asks for 1000)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config `seed` for the bootstrap RNG")
    args = ap.parse_args(argv)

    result = run(config=args.config, raw=args.raw, annotated=args.annotated,
                 out=args.out, n_boot=args.n_boot, seed=args.seed)

    for line in result["join"].lines():
        print(line)
    for line in result["direct_report"].lines():
        print(line)
    for p in result["plots"]:
        print(f"wrote {p}")
    print(f"wrote {result['summary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
