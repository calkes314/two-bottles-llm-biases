# bottles — a budget study of status quo bias in LLMs

Two people, two identical bottles, one poisoned; you may drink the one in your
hand or trade for the other. The game is symmetric and has no dominant answer,
so whatever a model does is driven by framing rather than by the payoff
structure. This harness runs that scenario across five cheap OpenRouter models
× five sampling temperatures × four scenario framings × four personas × two
answer modes, then codes each response for its answer
(`KEEP`/`SWITCH`/…) and for how many levels of opponent-modelling the reasoning
contains. The headline question is whether flipping *which option is the default
action* — keep-by-default versus trade-by-default, same game either way — flips
the model's answer, which would be status quo bias in a model, parallel to the
90/10 split found in humans. The secondary question is whether chain-of-thought
damps or amplifies sensitivity to sampling temperature. Full specification in
[`PLAN.md`](PLAN.md); the interface every stage builds against is
[`CONTRACTS.md`](CONTRACTS.md).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
export OPENROUTER_API_KEY=sk-or-v1-...      # or: cp .env.example .env && edit
```

Python 3.11+. `pip install -r requirements.txt` works just as well as `uv`.

## Running the stages

Every entry point runs as a module from the repo root, so `from src.contracts
import ...` resolves. The key comes from `OPENROUTER_API_KEY` or, failing that,
from `.env` — both stages use `contracts.load_api_key`, so you only have to put
it in one place.

```bash
# 0. Smoke test — one call to the cheapest model, verifies key + response shape
.venv/bin/python scripts/smoke.py

# 0b. Tests. Never make network calls: tests/conftest.py points the .env
#     fallback at a nonexistent path and the data paths at a tmpdir.
python -m pytest tests/ -q

# 1. Manifest — the job list and a dry-run cost estimate. Calls nothing.
python -m src.manifest --dry-run

# 2. Contamination check (PLAN.md §2) — 15 calls, then read the table
python -m src.runner --mode contamination --yes
python -m scripts.contamination_report --full

# 3. PILOT (PLAN.md §9 step 5 — "the one not to skip") — 40 calls, ~$0.003.
#    A sub-grid is a config file, not a set of flags: see CONTRACTS.md §12.
python -m src.runner   --config config/pilot.yaml --mode cot --yes
python -m src.annotate --config config/pilot.yaml
python -m scripts.pilot_report --config config/pilot.yaml   # all 40, in full

# 4. Direct grid — one call per (model, scenario, persona), provider-pinned.
#    The five temperature levels are computed from the logprobs, not sampled.
python -m src.runner --mode direct --yes

# 4b. Temperature validation (PLAN.md §8.3) — 200 sampled draws against the
#     analytic curve. A disagreement is the result, not a bug.
python -m scripts.direct_validation --dry-run
python -m scripts.direct_validation --yes

# 5. Full CoT grid — 10,000 calls. Resumable: kill it and rerun.
python -m src.runner --mode cot --dry-run     # check the number first
python -m src.runner --mode cot --yes

# 6. Annotate everything, then hand-code 60 for the depth kappa
python -m src.annotate
python -m src.annotate --export-for-handcoding 60 --out data/handcode.csv
python -m src.annotate --kappa data/handcode.csv    # after you fill it in

# 7. Four figures and summary.md into out/
python -m src.analyze

# 7c. Extraction audit — the annotator vs a regex, and the fights between them.
#     Only ~half of pilot responses ended with a bare answer word, and
#     non-compliance tracks the answer, so this is not cosmetic.
python -m scripts.extraction_check --full

# 7b. Provider audit — every cell should have been served by ONE provider.
#     Providers differ measurably, so a split cell is a mixture, not one
#     distribution. Pinning is on by default; this checks it held.
python -m scripts.provider_report --strict

# 8. Flat table — raw joined to annotated, one row per call, for ad-hoc
#    questions and for anything that wants a CSV rather than two JSONLs
python -m src.export                       # -> out/cells.csv

# 9. Local UI — ad-hoc playground plus a viewer for the grid
python -m src.webui                        # http://127.0.0.1:5000
```

### The local UI

`python -m src.webui` serves two tabs on 127.0.0.1.

**Playground** — pick a model, edit the scenario and role text freely, set a
temperature and a replicate count, run it. You get an answer chart and every
response in full. Use it to probe anything the pilot surprises you with,
without touching the grid.

**Experiment** — browse, filter and search the real grid (`raw.jsonl` joined to
`annotated.jsonl`) with the figures and `summary.md` beside it. This is the
comfortable way to do PLAN.md §9 step 5's hand-read of the pilot.

**Extra models.** The dropdown is grouped: the five grid models, then cheaper
extras, then frontier models (`claude-haiku-4.5`, `claude-sonnet-5`) for
checking whether a result is a property of the scenario or of small models. The
form prices a batch at its ceiling first, since these run to ~500× the grid's
cost per call. Extra models have no verified provider pin, and the form says so
rather than claiming one.

**Extra persona.** `survive_at_all_costs` is PLAN.md §10's deferred "not die at
all costs" goal arm: it leaves reasoning depth untouched but collapsed both
tested models to SWITCH 4/4. Playground-only — a goal arm is a different study
(CONTRACTS.md §14).

`adversary_bluff` started here too and was **promoted into the grid** on
2026-09-03, so it is now one of the four study personas (CONTRACTS.md §7).

**Getting deeper reasoning.** Max output tokens is a *ceiling*, not a target —
raising it alone changes nothing, because models stop when they are done
(gemma given 8,000 tokens still stopped at ~518). The lever that works is the
**"Demand explicit multi-level reasoning"** checkbox, which appends a measured
instruction before the answer line. Median visible reasoning, control →
instructed: gpt-4o-mini 409→2,662 chars, llama 467→2,966, granite 1,198→3,415,
gemma 2,698→3,915, and qwen3-30b **6→21,064**. Raise max output tokens to match
or it truncates. This is *not* what the grid sends: instructing depth measures
reasoning-when-told, a different question (CONTRACTS.md §13).

Three deliberate properties:

- The API key is read server-side and never reaches the browser.
- Playground calls are written to `data/playground.jsonl`, **never**
  `raw.jsonl`, with `pg-`-prefixed ids that hash the prompt text. The grid's
  `run_id` does not cover prompt text, so an ad-hoc call with an edited prompt
  would otherwise carry the same id as a real job — silently blocking its
  collection or contaminating it.
- Request building, retry and record shape come from `runner.py`, so the
  playground cannot drift from the experiment it informs.

It binds to localhost with no authentication. Do not expose it.

`--dry-run` on the manifest or the runner prices the pending jobs and calls
nothing. The runner is resumable on `run_id` and writes one flushed line per
successful call, so an interrupted run costs you nothing but the calls in
flight. **Move `data/raw.jsonl` aside if you change any prompt text** —
`run_id` does not cover the prompt, so resume would otherwise mix two wordings
under one id (CONTRACTS.md §2).

## Layout

```
config/budget.yaml               experiment grid, model ids, measured 2026-09-03
config/pilot.yaml                the 40-call pilot sub-grid (CONTRACTS.md §12)
scenarios/*.txt                  the three prompts + the contamination question
src/contracts.py                 shared enums, personas, run_id, messages, api key
src/manifest.py                  job list + dry-run cost
src/runner.py                    async OpenRouter calls, resume, retry
src/prices.py                    live price fetch with a 24h cache
src/annotate.py                  extraction + depth coding + kappa
src/rubric.py                    the annotation prompt
src/curve.py                     direct-mode temperature rescale
src/analyze.py                   aggregate, bootstrap, plot
src/export.py                    raw + annotated -> out/cells.csv
src/webui.py                     local playground + grid viewer (Flask)
src/webui_index.html             its single-page front end
scripts/smoke.py                 one hardcoded call
scripts/contamination_report.py  the §9 model x scenario table
scripts/direct_validation.py     the §8 temperature-validation sub-run
scripts/pilot_report.py          every response in full + seven pilot checks
scripts/provider_report.py       provider composition per cell; flags splits
scripts/extraction_check.py      annotator vs regex; lists what to hand-adjudicate
tests/                           456 tests; conftest.py blocks real spend
data/                            raw.jsonl, annotated.jsonl,
                                 playground.jsonl                  (gitignored)
out/                             figures + summary.md              (gitignored)
```

## Before you write code

Read `CONTRACTS.md`. It fixes the config schema, the job dict, both JSONL
record shapes, the two enums, the persona prompts and the direct-mode
convention, and it records three things that are true of the live API but not
of PLAN.md, all established by ~600 live probe calls: the model set has been
re-selected on measured behaviour rather than the catalogue (§11); OpenRouter's
advertised `supported_parameters` over-promises because it is a union across
providers, so direct-mode calls must pin their provider (§8); and one model in
the set emits no chain of thought at all and cannot be depth-coded (§8).
