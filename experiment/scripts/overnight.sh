#!/bin/bash
# Full budget_v1 run, in order, resumable. Every stage logs its own timing.
# Kill it any time: the runner resumes on run_id and never duplicates.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
say() { echo; echo "=== $(date -u +%H:%M:%SZ)  $*"; echo; }

say "1/6 direct grid (80 calls, provider-pinned)"
$PY -m src.runner --mode direct --yes || echo "!! direct stage returned $?"

say "2/6 contamination (20 calls)"
$PY -m src.runner --mode contamination --yes || echo "!! contamination returned $?"

say "3/6 cot grid (10,000 calls) — the long one"
$PY -m src.runner --mode cot --yes --progress-every 250 || echo "!! cot returned $?"

say "4/6 annotation (deepseek-v3.2)"
$PY -m src.annotate || echo "!! annotate returned $?"

say "5/6 audits"
$PY -m scripts.provider_report   || true
$PY -m scripts.extraction_check  || true

say "6/6 analysis + figures"
$PY -m src.analyze || echo "!! analyze returned $?"
$PY -m src.export  || true

say "DONE. Figures in out/, summary in out/summary.md"
