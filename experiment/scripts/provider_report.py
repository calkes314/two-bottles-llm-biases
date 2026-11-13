#!/usr/bin/env python3
"""Which provider served each cell, and which cells are contaminated by a mix.

    python -m scripts.provider_report              # data/raw.jsonl
    python -m scripts.provider_report --strict     # exit 1 if any cell is split

A cell is (model x scenario x persona x temperature x mode). Its replicates are
only draws from one distribution if one provider served all of them. Providers
differ measurably — see contracts.COT_PROVIDER_PIN — so a split cell is a
mixture, and its variance is inflated by an amount nothing downstream can
estimate.

Jobs built after the provider-pinning fix carry `provider_routing` for every
mode, so new data should show one provider per cell. This report exists for
data collected BEFORE that fix, and as a standing check that the pins are
actually being honoured.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from src import contracts as c
from src import export

CELL = ("model", "scenario", "persona", "temperature", "mode")


def cells(records: list[dict]) -> dict[tuple, collections.Counter]:
    out: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    for record in records:
        key = tuple(record.get(k) for k in CELL)
        out[key][record.get("provider")] += 1
    return out


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", default=str(c.RAW_PATH))
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 if any cell was served by more than one provider")
    args = parser.parse_args(argv)

    records = export.read_jsonl(Path(args.raw))
    if not records:
        print(f"no records in {args.raw}")
        return 0

    table = cells(records)
    split = {k: v for k, v in table.items() if len(v) > 1}
    pinned = sum(1 for r in records if r.get("provider_routing"))

    print(f"{len(records)} records · {len(table)} cells · "
          f"{pinned}/{len(records)} sent with a provider pin\n")

    per_model: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for record in records:
        per_model[f"{record.get('model')} [{record.get('mode')}]"][record.get("provider")] += 1
    print("providers used, per model and mode")
    for name, counter in sorted(per_model.items()):
        flag = "  <-- MIXED" if len(counter) > 1 else ""
        print(f"  {name:52s} {dict(counter)}{flag}")

    if not split:
        print(f"\nOK: every one of the {len(table)} cells was served by a single provider.")
        return 0

    print(f"\n!! {len(split)} of {len(table)} cells were served by MORE THAN ONE provider.")
    print("   Those cells are a mixture of distributions, not one distribution, so")
    print("   their replicates are not iid and their variance is inflated.")
    for key, counter in sorted(split.items(), key=lambda kv: -len(kv[1]))[:12]:
        model, scenario, persona, temperature, mode = key
        print(f"   {str(model).split('/')[-1]:26s} {scenario:15s} {persona:14s} "
              f"T={temperature!s:4s} {mode:6s} {dict(counter)}")
    if len(split) > 12:
        print(f"   … and {len(split) - 12} more")

    print("\n   Fix for new data: leave `pin_providers` at its default (true) so every")
    print("   mode pins. Fix for this data: re-collect the affected cells, or report")
    print("   provider as a factor and never pool across a split.")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(_main())
