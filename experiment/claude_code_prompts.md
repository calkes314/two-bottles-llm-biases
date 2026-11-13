# Claude Code prompts — budget version

## How to use these

```
Phase 0   Prompt 0 alone, in one tab. Wait for it to finish. (~15–20 min)
Phase 1   Prompts A, B, C in three tabs simultaneously. (~30–45 min)
Phase 2   Prompt D alone, once A–C have all committed. (~30 min + run time)
```

**The gate is between Phase 1 and Phase 2, not before Phase 1.** A, B and C
can start as soon as CONTRACTS.md exists, because everything still in flux
(model ids, max_tokens, provider pins, prompt wording) lives in
config/budget.yaml and src/contracts.py, which all three import rather than
restate. What must NOT start early is real data collection: `run_id` hashes
model|scenario|persona|mode|temp|replicate and does **not** cover the prompt
text, so if the scenario wording or max_tokens changes after you have started
collecting, resume will happily skip jobs and leave you with one raw.jsonl
mixing two different prompts under identical ids. A's 30-record resume check
is throwaway and fine. Prompt D step 4 is the real gate.

Before starting, put the plan where Claude Code can read it:

```bash
mkdir bottles && cd bottles
cp ~/Downloads/plan_1_budget_version.md PLAN.md
export OPENROUTER_API_KEY=sk-or-...
claude
```

Every prompt starts with `ultrathink` and tells the tab which files it
owns. The file-ownership lines are what make parallel work safe — do not
remove them. Each tab commits only its own files.

Opus with ultrathink is a good fit for Prompt 0 and Prompt D, where the
judgment calls live. A, B, and C are well-specified enough that Sonnet
would do them fine if you want to save on the Claude Code side too.

---

## Prompt 0 — Scaffold and contracts (sequential, first)

```
ultrathink

You are setting up a small research harness. Read PLAN.md in full before
doing anything else — it is the specification and it is authoritative.

Your job in this session is ONLY the scaffold and the shared contracts.
Three other sessions will build the runner, annotator, and analysis in
parallel afterwards, so the contracts you write are what let them work
independently. Get them right and be explicit; ambiguity here costs three
people time.

Do NOT write src/manifest.py, src/runner.py, src/annotate.py,
src/analyze.py, or src/curve.py. Those belong to other sessions.

DO write src/contracts.py: the executable half of CONTRACTS.md, holding the
enums, the persona prompts, the depth rubric, the direct-mode instruction,
run_id, the message builder, and the measured constants below. Three
sessions will import it, and anything they have to re-implement is
something they can get subtly wrong. Also write tests/test_scaffold.py
covering the scenario texts, run_id formatting and the config schema.

Do this:

1. Repo layout exactly as PLAN.md §7. Python 3.11+, uv or pip with a
   requirements.txt (httpx, pyyaml, numpy, pandas, matplotlib, pytest).
   .gitignore that excludes data/, out/, .env, and __pycache__.
   README.md with a one-paragraph description and the commands to run
   each stage (leave the commands as TODOs the other sessions will fill).

2. Write CONTRACTS.md. This is the most important file you produce. It
   must define, with a concrete JSON example for each:
   - the config schema (mirror config/budget.yaml, document every field
     and its type)
   - the job dict produced by the manifest, including how run_id is
     computed (sha1 of "model|scenario|persona|mode|temp|replicate") and
     the exact messages[] structure for direct vs cot mode
   - the raw.jsonl record: the job dict plus response, model_returned,
     provider, usage, and an ISO timestamp
   - the annotated.jsonl record: run_id plus answer, depth, flag_no_basis,
     flag_rigged, annotator_model, and raw_annotator_output
   - the two enums verbatim: answers
     [KEEP, SWITCH, RANDOMISE, THIRD_OPTION, REFUSE, UNPARSEABLE] and
     depth codes [L0, L1, L2, L3, MIX, NON]
   - the persona system prompts, verbatim from PLAN.md §3
   - the direct-mode convention: the prompt asks for a single letter K or
     S, max_tokens small, temperature always 1.0, logprobs on,
     top_logprobs 10. State that the temperature curve is computed from
     these logprobs, never sampled.
   Put a line at the top: "Other sessions build against this file. If you
   need to change it, change it here first and say so in your commit."

3. Write the three scenario files in scenarios/ verbatim from PLAN.md §2.
   Do not paraphrase or "improve" them — the wording is the manipulation.
   Also write scenarios/contamination_check.txt from §2.

4. Write config/budget.yaml from PLAN.md §7. Then fetch
   https://openrouter.ai/api/v1/models and verify every model id exists.
   If one does not, pick the closest equivalent in the same tier, update
   the config, and tell me what you changed and why.

   Do NOT trust the catalogue's logprobs claim. `supported_parameters` is
   a UNION over providers and over-promises in both directions: a model
   can advertise logprobs and return none, or return an array misaligned
   with its own emitted text. MEASURE it — one real direct-mode call per
   model per provider — and record the measurement, not the claim.
   Record a provider pin per model, exactly one provider each, because
   `only` is a permitted set rather than a preference order.

   Measure two more things while you are there, because neither is visible
   from the catalogue and both can invalidate a model:
     - does the CoT answer land on the LAST line, at temperature 1.6 as
       well as 0.8? Truncation at max_tokens destroys it and reads as
       disobedience in the data.
     - is there visible reasoning in message.content to depth-code?
       Thinking models hide it in message.reasoning and you pay for
       tokens the annotator never sees; terse models emit none at all.
   Report the measurements per model and tell me which models you would
   drop and why.

5. Write scripts/smoke.py: one hardcoded call to the cheapest model in the
   config with scenarios/default_keep.txt, no persona, temperature 0.8,
   cot mode. Print the response text, model_returned, provider, and usage.
   Run it. If it fails, fix the environment until it works. This verifies
   the API key, connectivity, and that the response shape matches what
   CONTRACTS.md claims — if it does not match, fix CONTRACTS.md.

6. git init, one commit: "scaffold, contracts, scenarios, config, smoke".

When you are done, print a summary: which model ids you changed, which
models support logprobs, and the smoke-test response. Then stop.
```

---

## Prompt A — Manifest and runner (parallel tab 1)

```
ultrathink

Read PLAN.md and CONTRACTS.md in full before writing anything. CONTRACTS.md
is binding; if you think it is wrong, stop and tell me rather than
diverging from it.

You own exactly these files and no others:
  src/manifest.py
  src/runner.py
  src/prices.py
  tests/test_manifest.py
  tests/test_runner.py
Two other sessions are building src/annotate.py and src/analyze.py right
now against the same CONTRACTS.md. Do not touch their files, scenarios/,
config/, or CONTRACTS.md.

Build:

src/prices.py
  - fetch_prices() -> dict[model_id, {prompt, completion}] in USD per token
    from https://openrouter.ai/api/v1/models. Cache to data/prices.json
    with a timestamp; refetch if older than 24h. Note data/ is gitignored,
    so the cache is local-only — that is intended, do not "fix" it.

src/manifest.py
  - build_jobs(config) -> list[job] per CONTRACTS.md. Direct-mode jobs:
    exactly one per (model, scenario, persona), temperature field set to
    1.0. CoT jobs: full cross product including temperatures and
    replicates. Contamination jobs: one per (model, scenario), persona
    none, mode "contamination" — 15 of them, CONTRACTS.md §9; they are
    part of the manifest even though Prompt D is what runs them. run_id
    per the contract. Deterministic ordering.
  - NEVER hardcode max_tokens. Take it from contracts.DIRECT_MAX_TOKENS,
    COT_MAX_TOKENS and CONTAMINATION_MAX_TOKENS. PLAN.md §8.2's snippet
    shows 1 and 900; both are superseded and 900 will silently truncate
    verbose models before they reach their final-line answer.
  - estimate_cost(jobs, prices) using the formula in PLAN.md §6. Use the
    MEASURED token counts in CONTRACTS.md §10, not §6's estimates: real
    input is ~200 tokens, not ~420. Output length is strongly
    model-dependent (a terse model returns 3 tokens, a verbose one runs to
    max_tokens), so return a per-model breakdown and a total, and do not
    report a single blended number as though it were reliable.
  - CLI: `python -m src.manifest --config config/budget.yaml --dry-run`
    prints job counts by mode and model, the cost breakdown, and the
    total. Nothing is called.

src/runner.py
  - Async, httpx, semaphore from config.max_concurrency, exponential
    backoff on 429/5xx, 5 attempts.
  - Before appending ANY record, call
    `contracts.is_usable_response(status_code, data)`. OpenRouter returns
    HTTP 200 with an `{"error": ...}` body and no `choices` when the
    upstream provider fails; `raise_for_status()` does not catch it, and
    writing it as a record marks that run_id done so resume never retries
    — a permanent silent hole in the grid. Treat a False as a retryable
    failure and write no line.
  - Append-only data/raw.jsonl, one record per call, per the contract.
    Writes serialised through an asyncio.Lock. Flush after every record.
  - Resume: on start, read existing run_ids from raw.jsonl and skip them.
  - Record model_returned and provider from the RESPONSE, not the request.
  - Direct mode: set logprobs=true, top_logprobs=10, temperature=1.0,
    max_tokens as CONTRACTS.md specifies. Store the full logprobs object.
  - Direct mode MUST send provider routing from
    `contracts.DIRECT_PROVIDER_PIN` as the request's `provider` field.
    Without it OpenRouter routes to a provider that silently returns no
    logprobs and the entire direct grid is wasted spend.
    `{"require_parameters": true}` does NOT work — it was measured routing
    to a provider that returned nothing. Only an explicit `only` list
    does. Emit direct jobs only for models the config marks
    logprobs: true. CONTRACTS.md §8 has the per-provider measurements.
  - CLI: `python -m src.runner --config config/budget.yaml [--limit N]
    [--mode direct|cot] [--models a,b]`. Always print the dry-run estimate
    and require a y/n confirmation before making calls unless --yes.
  - Progress line every 50 calls with running spend computed from usage
    and prices.

Tests, with pytest, no network:
  - manifest: job count equals the expected cross product; run_ids are
    unique; the same config produces byte-identical job lists twice.
  - runner: mock httpx; verify resume skips completed run_ids; verify a
    429 triggers a retry; verify concurrent writes produce valid JSONL
    (parse every line back).

Manual check I want you to actually perform: run
`python -m src.runner --config config/budget.yaml --limit 30 --mode cot --yes`,
kill it with Ctrl-C after ~10 records, rerun it, and confirm raw.jsonl
contains exactly 30 records with no duplicate run_ids. This costs about a
cent. Tell me the result.

Commit only your files. Commit message: "manifest, runner, prices".
```

---

## Prompt B — Annotator (parallel tab 2)

```
ultrathink

Read PLAN.md and CONTRACTS.md in full before writing anything. CONTRACTS.md
is binding; if you think it is wrong, stop and tell me rather than
diverging from it.

You own exactly these files and no others:
  src/annotate.py
  src/rubric.py
  tests/test_annotate.py
  tests/fixtures/raw_sample.jsonl
Two other sessions are building the runner and the analysis right now.
Do not touch their files, scenarios/, config/, or CONTRACTS.md.
data/raw.jsonl may not exist yet — you must be able to build and test
without it, using your own fixture.

Build:

tests/fixtures/raw_sample.jsonl
  - 12 hand-written records matching the raw.jsonl contract exactly.
    Include: a clear KEEP with L1 reasoning, a clear SWITCH with L2, an
    L3 regress, a MIX, a RANDOMISE (explicitly flips a coin despite the
    instruction), a THIRD_OPTION (refuses to drink), a REFUSE, a response
    that suspects the setup is rigged, a response that says there is no
    rational basis, one that is garbled/truncated, and two direct-mode
    records (single-token answer with logprobs) that the annotator must
    SKIP. Make the cot responses realistic — a few paragraphs each.

src/rubric.py
  - The annotation prompt template from PLAN.md §8.4. Import the rubric
    from `contracts.DEPTH_RUBRIC` rather than re-pasting §5's table: that
    constant already carries PLAN.md §5's `flag_no_basis` and
    `flag_rigged` definitions as well as the depth codes, and the §8.4
    JSON schema asks the annotator for both booleans without defining
    them. Paste it whole.
  - Takes response_text only. It must not receive or mention model name,
    persona, scenario, or temperature — the annotator is blind to
    condition. Assert this in a test.

src/annotate.py
  - Read raw.jsonl and annotate ONLY records with `mode == "cot"`. There
    are now four modes in the file — cot, direct, contamination and
    direct_validation (contracts.ALL_MODES) — and only cot is annotated.
    Whitelist cot; do not blacklist direct, or the two side runs leak in.
  - Skip run_ids already in annotated.jsonl (idempotent, same pattern as
    the runner). One line per run_id; if you deliberately re-annotate with
    a different model, write a NEW file rather than appending duplicates.
  - One call per cot record to config.annotator via OpenRouter,
    temperature 0.
  - Parse strict JSON. Strip markdown fences if present. Validate answer
    and depth against the enums in CONTRACTS.md. On any parse or
    validation failure: answer=UNPARSEABLE, depth=NON, and store the raw
    output so it can be reviewed. Never crash on one bad record.
  - Append to data/annotated.jsonl per the contract, one record per line.
  - CLI: `python -m src.annotate --config config/budget.yaml [--limit N]
    [--input path]`. --input lets me point it at the fixture.
  - Print at the end: count annotated, count UNPARSEABLE, and the
    UNPARSEABLE rate. If the rate is over 2%, print a warning.

  Also a hand-coding helper:
  - `python -m src.annotate --export-for-handcoding 60 --out data/handcode.csv`
    writes a stratified sample of 60 cot responses (stratify by model and
    scenario) to CSV with columns run_id, response_text, and empty
    answer/depth/flag_no_basis/flag_rigged columns for me to fill in.
  - `python -m src.annotate --kappa data/handcode.csv` reads my filled
    CSV, joins on run_id to annotated.jsonl, and prints Cohen's kappa for
    each of the four fields plus a confusion matrix for depth.

Tests, with pytest, no network:
  - the rubric prompt never contains model/persona/scenario/temperature
    strings from the record
  - direct-mode records are skipped
  - a mocked annotator returning valid JSON produces a valid record
  - a mocked annotator returning garbage produces UNPARSEABLE without
    raising
  - markdown-fenced JSON is parsed correctly
  - rerunning on the same input adds zero new records

Then actually run it once against the fixture with the real annotator
model (12 calls, well under a cent) and show me the 12 results. I want to
see whether the small model gets the depth codes right on the easy cases
before we trust it on 5,000 records.

Commit only your files. Commit message: "annotator, rubric, handcoding".
```

---

## Prompt C — Analysis and temperature curve (parallel tab 3)

```
ultrathink

Read PLAN.md and CONTRACTS.md in full before writing anything. CONTRACTS.md
is binding; if you think it is wrong, stop and tell me rather than
diverging from it.

You own exactly these files and no others:
  src/curve.py
  src/analyze.py
  tests/test_curve.py
  tests/test_analyze.py
  tests/fixtures/annotated_sample.jsonl
  tests/fixtures/direct_sample.jsonl
Two other sessions are building the runner and the annotator right now.
Do not touch their files, scenarios/, config/, or CONTRACTS.md.
data/ may be empty — build and test against your own fixtures.

Build:

src/curve.py
  - rescale(logprobs: dict[str, float], T: float) -> dict[str, float]
    implementing p_i(T) ∝ p_i(1)^(1/T), renormalised over the given
    tokens only. T=0 returns a one-hot on the argmax. Numerically stable
    (subtract max before exp).
  - extract_answer_logprobs(raw_record): this already exists as
    `contracts.answer_logprobs(choice)` — wrap it, do not reimplement it.
    It sums probabilities across token variants ("K", " K", "k") and
    returns (logprobs, status) where status is "ok", "no_logprobs" or
    "no_answer_tokens". Count and report each failure status. With the
    shipped config every model should come back "ok"; a failure is a
    pinning or provider problem, not a model limit, and is worth flagging
    loudly. Failures cluster by model and provider, never by condition, so
    they are not missing at random — drop the whole MODEL from the overlay
    rather than dropping individual cells, which would silently reweight
    the survivors.
  - direct_curve(raw_records, temps) -> DataFrame with columns
    model, scenario, persona, temperature, p_keep, p_switch.

tests/test_curve.py
  - T=1.0 returns the input distribution unchanged
  - T→0 (use 1e-6) returns one-hot on the argmax
  - T=100 is within 0.01 of uniform
  - a known two-token case computed by hand matches to 1e-9
  - leading-space token variants are merged

src/analyze.py
  - Load annotated.jsonl joined to raw.jsonl on run_id for cot records;
    load direct records via curve.py.
  - The unit of analysis is the (model × condition) cell, never the
    individual sample. For every rate you report, compute a 95% CI by
    cluster bootstrap over models (resample models with replacement,
    1000 iterations). Write a docstring explaining why pooling samples
    across models would be wrong; PLAN.md §8.5 says why.
  - Compute flip rate per CONTRACTS.md §8: for each (model, persona,
    temperature),

        flip_rate = keep_rate(default_keep) - keep_rate(default_switch)

    SIGNED, not absolute. The sign is the finding: positive means the
    model follows the default action in both framings (status quo bias);
    negative means it takes the action in both, which is the opposite
    behaviour and has already been observed in probing. An absolute value
    collapses the two into the same number. `keep_rate` here is
    conditional on {KEEP, SWITCH} — see the denominator note below. This
    is the headline number.
  - Four plots to out/, per PLAN.md §8.5, matplotlib, readable at
    single-column width:
    1. keep_vs_temperature.png — KEEP rate vs temperature, one line per
       model, cot mode, with the analytic direct-mode curve for the same
       model overlaid as a dashed line. One panel per scenario.
       MATCH THE DENOMINATORS. The analytic curve is conditional — it
       renormalises over {K, S} — so the cot series on this plot must be
       n(KEEP) / (n(KEEP) + n(SWITCH)), not the unconditional rate. Report
       the excluded fraction alongside; it is plot 4. Plots 2-4 use the
       unconditional denominator. State which is used in each caption.
       The dashed overlay exists for models in
       contracts.DIRECT_CURVE_MODELS — currently all five. Derive the set
       from that constant, never a literal count: it changes whenever a
       model is swapped. If one is ever missing, draw the overlay where it
       exists, say so on the figure, and do NOT drop that model's cot line
       to make the panels match.
    2. flip_rate.png — bar chart of default-keep vs default-switch KEEP
       rate per model, with CIs. Use the SAME conditional denominator as
       the flip_rate definition above, n(KEEP)/(n(KEEP)+n(SWITCH)), or the
       bars will not equal the number you are calling the headline. Show
       the signed flip rate, so a model biased toward the default and one
       biased toward acting point opposite ways.
    3. depth_by_persona.png — stacked bar of depth codes by persona,
       pooled over models with cells weighted equally, EXCLUDING every
       model in contracts.NO_COT_MODELS. Those models emit an answer with
       no reasoning attached (measured; it is a property of the model, not
       the prompt), so their depth codes are uniformly L0/NON and pooling
       them in would drag every bar toward L0 and look like a finding.
       State on the figure which models were excluded and why. Exclude
       them from any claim about reasoning, including PLAN.md's second
       question about CoT and temperature sensitivity — a model whose cot
       arm is three tokens has no cot arm.
    4. nonbinary_by_temperature.png — RANDOMISE, THIRD_OPTION, REFUSE,
       UNPARSEABLE rates vs temperature.
  - Write out/summary.md: a table of KEEP rate per (model, scenario) with
    CIs, the flip rates, the UNPARSEABLE rate, how many direct-mode
    records were unusable, and the depth distribution. Plain numbers, no
    interpretation — I will write the interpretation.
  - CLI: `python -m src.analyze --config config/budget.yaml
    [--raw path] [--annotated path] [--out out/]`

tests/fixtures/
  - annotated_sample.jsonl and direct_sample.jsonl: synthetic but
    contract-conformant data for 3 models × 3 scenarios × 2 personas ×
    3 temperatures × 8 replicates, with a deliberate built-in flip effect
    for one model and none for another, so I can eyeball whether the
    plots show what the fixture encodes.

tests/test_analyze.py
  - the bootstrap CI for a cell with all-KEEP is [1, 1]
  - flip rate for the fixture model with the built-in effect is large and
    for the null model is near zero
  - all four plots and summary.md are produced from the fixtures
  - the analysis raises a clear error if it detects it is being asked to
    pool across models without clustering

Run the analysis against the fixtures and show me the four plots.

Commit only your files. Commit message: "curve, analysis, fixtures".
```

---

## Prompt D — Integration and pilot (sequential, after A–C)

```
ultrathink

Read PLAN.md, CONTRACTS.md, and the git log. Three sessions have just
built the runner, annotator, and analysis in parallel. Your job is to
integrate, pilot, and then run the real thing — with a hard stop for me
in the middle.

Step 1 — Integration.
  Run the full test suite. Fix any failures. Then run the three stages
  end-to-end against the fixtures: annotate on tests/fixtures/raw_sample
  .jsonl, analyze on the fixture outputs. If the pieces do not fit — a
  field name mismatch, a shape the analysis did not expect — fix it at
  the seam and update CONTRACTS.md to match reality. Commit
  "integration fixes".

Step 2 — Contamination check.
  Run scenarios/contamination_check.txt once per model per scenario
  (15 calls). Print a table: model × scenario → what it named. Tell me
  whether the isomorph is recognised. If it is, stop and tell me; do not
  proceed.

Step 3 — Pilot. STOP AFTER THIS STEP.
  Run: 2 models (one small, one mid) × scenarios default_keep and
  default_switch × persona none × temperature 0.8 × cot × 10 replicates
  = 40 calls. Then annotate them.
  Print ALL 40 responses in full, grouped by model and scenario, each
  followed by its annotation. Do not summarise them. I am going to read
  every one.
  Then print: UNPARSEABLE count, answer distribution per cell, and
  anything you noticed — models weaselling out of the binary, the
  final-line format being ignored, the default_switch wording being
  misread, annotator errors on obvious cases.
  Then STOP and wait for me. Do not run the full grid until I say so.

Step 4 — Only when I say go.
  a. Direct grid: `python -m src.runner --mode direct --yes`. With the
     shipped config this is exactly 45 calls — all five models return
     usable logprobs when pinned — but take the count from the manifest's
     dry run rather than hardcoding it, since it drops if a model is ever
     swapped for one without logprobs.
     Report how many direct records had usable answer logprobs, broken
     down by contracts.answer_logprobs status AND by provider. The
     expected result is 45/45. Anything less is a provider-pinning
     failure, not a model limit — every configured model is in
     contracts.DIRECT_CURVE_MODELS. Stop and tell me rather than
     proceeding on a partial direct grid.
  b. Temperature validation: run the `direct_validation` sub-run exactly
     as CONTRACTS.md §8 specifies it — openai/gpt-4o-mini, default_keep,
     persona none, T=0.4 and T=1.2, replicate 0..99 at each, 200 calls.
     Use mode "direct_validation" and give each draw a REAL temperature
     and replicate index; if you reuse direct mode's temperature=1.0 and
     replicate=0 all 100 draws collapse onto one run_id and resume keeps
     only the first. Compare observed KEEP proportions to what
     curve.rescale predicts from the T=1.0 logprobs. Report both numbers
     side by side. This is a reportable methods check, and a disagreement
     is itself the result — do not "fix" it.
  c. Full cot grid. Print the dry-run cost, confirm it is under $6, then
     run with --yes. Progress every 50 calls. Watch the per-model output
     token counts in the first 100 calls: a model whose median output is
     under ~50 tokens is not doing chain-of-thought at all and cannot be
     depth-coded, and a model running to max_tokens on most calls is
     being truncated before it reaches its final-line answer. Either is
     worth stopping for.
  d. Annotate everything. Report UNPARSEABLE rate.
  e. Export 60 for hand-coding to data/handcode.csv and tell me where it
     is. I will fill it in later.
  f. Run analysis. Open the four plots and summary.md and tell me what
     they show — numbers only, no interpretation.

Commit after each substep. Final commit "budget_v1 full run".
```

---

## Notes

**If a tab wants to change CONTRACTS.md:** it should stop and tell you,
not do it. Then you decide, edit the contract yourself, and tell the other
tabs to re-read it. This is the one coordination point and it is worth
being strict about.

**If Prompt 0's model-ID check changed models:** fine, the config is the
source of truth and all tabs read from it. Model ids are data that the
manifest iterates over, not code, so swapping one changes zero lines in A,
B or C. The same is true of everything else that moved after Phase 1
started — max_tokens, provider pins, prompt wording — because all of it
lives in config/budget.yaml and src/contracts.py, which every tab imports
rather than restates. That is the whole point of contracts.py: import from
it, never re-implement it.

**Expect the OpenRouter catalogue to be wrong.** `supported_parameters` is
a union over providers and over-promises: a model can advertise logprobs
and still return none, or return an array misaligned with the emitted
text. Anything the harness depends on gets verified with a real call, not
read off the catalogue. CONTRACTS.md §8 records what was measured.

**Same working tree or separate?** Same tree is fine because file
ownership is disjoint. If you want belt-and-braces, `git worktree add`
one directory per tab and merge after; but for a project this size that
is more ceremony than it saves.

**The pilot stop in Prompt D is the point of the whole evening.** Forty
responses, read by you, before the real spend. Everything after it is
mechanical.
