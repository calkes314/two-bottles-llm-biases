"""Flatten raw.jsonl + annotated.jsonl into one analysis-ready table.

`analyze.py` builds its DataFrames internally and writes only figures and
`summary.md`, so answering "what did this model say in that cell" means
re-joining two JSONL files every time. This module writes the join out once:

    python -m src.export                      # -> out/cells.csv
    python -m src.export --playground         # -> out/playground_cells.csv

One row per call. `raw.jsonl` is ~15 MB at full grid size, a third of it the
same three prompts repeated 5,625 times; the CSV drops the duplicated prompt
text and the raw API envelope, landing near 1-2 MB — small enough to load
whole, grep, or hand to pandas.

Owned by the scaffold session. It reads the other sessions' output files and
imports their helpers; it does not modify them.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from src import contracts as c

COLUMNS = [
    "run_id", "experiment", "mode", "model", "model_returned", "tier",
    "scenario", "persona", "temperature", "replicate", "provider",
    "answer", "depth", "flag_no_basis", "flag_rigged",
    "finish_reason", "out_tokens", "prompt_tokens", "cost",
    "p_keep_direct", "logprob_status", "annotator_model", "parse_ok",
    "ts", "response_text",
]


def read_jsonl(path: str | Path) -> list[dict]:
    """Tolerant reader: a truncated final line (killed mid-write) is skipped
    rather than raising, because that is a normal state for an append-only
    file that someone Ctrl-C'd."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def response_text(record: dict) -> str:
    """The assistant's visible text. Falls back to `reasoning` ONLY to make a
    thinking model's output inspectable in the UI — never for depth coding,
    which reads `content` alone (CONTRACTS.md §8)."""
    try:
        message = record["response"]["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return ""
    return message.get("content") or message.get("reasoning") or ""


def _choice(record: dict) -> dict:
    try:
        return record["response"]["choices"][0]
    except (KeyError, IndexError, TypeError):
        return {}


def flatten(raw: dict, annotation: dict | None) -> dict:
    """One row: the factor columns, the outcome, and enough provenance to
    audit it. `provider` is the server that answered, never what was routed."""
    usage = raw.get("usage") or {}
    choice = _choice(raw)
    row = {
        "run_id": raw.get("run_id"),
        "experiment": raw.get("experiment"),
        "mode": raw.get("mode"),
        "model": raw.get("model"),
        "model_returned": raw.get("model_returned"),
        "tier": raw.get("tier"),
        "scenario": raw.get("scenario"),
        "persona": raw.get("persona"),
        "temperature": raw.get("temperature"),
        "replicate": raw.get("replicate"),
        "provider": raw.get("provider"),
        "finish_reason": choice.get("finish_reason"),
        "out_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cost": usage.get("cost"),
        "ts": raw.get("ts"),
        "response_text": response_text(raw),
        "answer": None, "depth": None,
        "flag_no_basis": None, "flag_rigged": None,
        "annotator_model": None, "parse_ok": None,
        "p_keep_direct": None, "logprob_status": None,
    }
    if annotation:
        row.update(
            answer=annotation.get("answer"),
            depth=annotation.get("depth"),
            flag_no_basis=annotation.get("flag_no_basis"),
            flag_rigged=annotation.get("flag_rigged"),
            annotator_model=annotation.get("annotator_model"),
            parse_ok=annotation.get("parse_ok"),
        )
    if raw.get("mode") in ("direct", c.DIRECT_VALIDATION_MODE):
        logprobs, status = c.answer_logprobs(choice)
        row["logprob_status"] = status
        if logprobs:
            import math
            total = sum(math.exp(v) for v in logprobs.values())
            if total > 0:
                row["p_keep_direct"] = round(math.exp(logprobs["K"]) / total, 6)
    return row


def build_rows(raw_path: str | Path = c.RAW_PATH,
               annotated_path: str | Path = c.ANNOTATED_PATH) -> list[dict]:
    annotations = {a["run_id"]: a for a in read_jsonl(annotated_path) if a.get("run_id")}
    return [flatten(r, annotations.get(r.get("run_id"))) for r in read_jsonl(raw_path)]


def write_csv(rows: list[dict], out_path: str | Path = c.CELLS_CSV) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", default=None)
    parser.add_argument("--annotated", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--playground", action="store_true",
                        help="export data/playground.jsonl instead of the grid")
    args = parser.parse_args(argv)

    if args.playground:
        raw = args.raw or c.PLAYGROUND_PATH
        ann = args.annotated or c.PLAYGROUND_ANNOTATED_PATH
        out = args.out or (c.OUT_DIR / "playground_cells.csv")
    else:
        raw = args.raw or c.RAW_PATH
        ann = args.annotated or c.ANNOTATED_PATH
        out = args.out or c.CELLS_CSV

    rows = build_rows(raw, ann)
    if not rows:
        print(f"no records in {raw}", file=sys.stderr)
        return 1
    path = write_csv(rows, out)
    coded = sum(1 for r in rows if r["answer"])
    print(f"{len(rows)} rows -> {path}  ({coded} annotated, "
          f"{len(rows) - coded} not)  {path.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
