# CONTRACTS.md

**Other sessions build against this file. If you need to change it, change it
here first and say so in your commit.**

---

## 0. Scope and ownership

`PLAN.md` is the specification. This file is the *interface* between the four
sessions building against it. Where PLAN.md is silent, this file decides, and
every such decision is marked **[scaffold decision]** so you can argue with it.

| Path | Owner | Status |
|---|---|---|
| `config/budget.yaml`, `scenarios/*`, `src/contracts.py`, `scripts/smoke.py`, `tests/test_scaffold.py`, `README.md`, this file | scaffold session | **done** |
| `src/manifest.py` | manifest/runner session | **done** |
| `src/runner.py`, `src/prices.py` | manifest/runner session | **done** |
| `src/annotate.py`, `src/rubric.py` | annotator session | **done** |
| `src/analyze.py`, `src/curve.py` | analysis session | **done** |
| `config/pilot.yaml`, `scripts/direct_validation.py`, `scripts/contamination_report.py`, `scripts/pilot_report.py`, `tests/conftest.py`, `tests/test_scripts.py` | integration session | **done** |

The integration session owns the operator-facing side runs, which the four
build sessions each correctly declined to write because none of them owned the
seam: §8's `direct_validation` (fully specified here, deliberately not emitted
by `manifest.py`), §9's contamination readback (deliberately filtered out of
`analyze.py`), the pilot PLAN.md §9 step 5 insists on, and the test-suite
guards that keep all of it from spending money by accident.

`src/contracts.py` is the executable half of this document: the enums, the
persona prompts, the direct-mode instruction, the `run_id` function, the message
builder and `load_api_key` all live there. **Import them. Do not re-implement
them.**

> **Credentials.** `contracts.load_api_key()` reads `OPENROUTER_API_KEY` from
> the environment and falls back to `contracts.ENV_PATH` (the repo's `.env`).
> Every stage that calls OpenRouter uses it — `runner.py`, `annotate.py` and
> `scripts/direct_validation.py`. It was two implementations: `annotate.py` read
> `os.environ` only, while printing an error message that told the operator to
> put the key in `.env`, which silently did nothing. `ENV_PATH` is read at
> *call* time rather than bound as a default argument, so a test can point the
> fallback at a path that does not exist; `tests/conftest.py` does that for
> every test, because a test that only clears the environment variable would
> otherwise fall through to the operator's real key and spend money. It did:
> the annotate CLI tests annotated ten fixture records into the real
> `data/annotated.jsonl` before the guard existed.
Everything below that has a Python name in `code font` refers to a real symbol
in that module.

Repo root is this directory. All paths are relative to it; PLAN.md §7 draws the
tree under a `bottles/` folder, which is just the repo's name.

Entry points run as modules from the repo root, so that `from src.contracts
import ...` resolves:

```bash
python -m src.manifest      # not: python src/manifest.py
```

---

## 1. Config schema — `config/budget.yaml`

Loaded with `yaml.safe_load`. Every field, its type, and who reads it.

| Field | Type | Meaning |
|---|---|---|
| `experiment` | `str` | Label copied into every job and every raw record. |
| `seed` | `int` | Seed for any RNG. The grid itself is deterministic and does not need it; it exists so pilots and subsamples are reproducible. |
| `models` | `list[dict]` | See below. Order is canonical for plots. |
| `models[].id` | `str` | OpenRouter model id. Verified against the live catalogue 2026-09-03. |
| `models[].tier` | `"small" \| "mid"` | Reporting grouping only. Never affects a request. |
| `models[].logprobs` | `bool` | Whether a real direct-mode call to this model, pinned per `contracts.DIRECT_PROVIDER_PIN`, returned a usable `{K, S}` pair. **[scaffold decision]** — added to PLAN.md's schema because the manifest must skip direct-mode jobs for models that cannot answer them. This is a *measurement*, not the catalogue's claim; §8 explains why the catalogue cannot be trusted here. |
| `scenarios` | `list[str]` | Subset of `contracts.SCENARIOS`. Each must have a `scenarios/<name>.txt`. |
| `personas` | `list[str]` | Subset of `contracts.PERSONAS`. |
| `temperatures` | `list[float]` | CoT sampling temperatures. Ignored by direct mode (§8). |
| `answer_modes` | `list[str]` | Subset of `["direct", "cot"]`. The contamination run is not in the grid (§9). |
| `replicates_cot` | `int` | CoT draws per (model, scenario, persona, temperature) cell. |
| `allowed_answers` | `list[str]` | Answer labels the annotator may emit, excluding `UNPARSEABLE`, which is always permitted as a QA flag. Dropping `RANDOMISE` here is how PLAN.md §4's toggle is expressed. |
| `allow_randomise_in_prompt` | `bool` | `false` → the scenario keeps its "no random method" clause. `true` → that clause is replaced (§6). |
| `annotator` | `str` | Model id for the annotation call. Chosen by bake-off against hand-verified ground truth, and it must NOT be a model in `models` — see §7b. |
| `max_concurrency` | `int` | Runner semaphore size. |

```json
{
  "experiment": "budget_v1",
  "seed": 20260903,
  "models": [
    {"id": "qwen/qwen3-30b-a3b-instruct-2507", "tier": "small", "logprobs": true},
    {"id": "meta-llama/llama-3.1-8b-instruct", "tier": "small", "logprobs": true},
    {"id": "ibm-granite/granite-4.1-8b",       "tier": "small", "logprobs": true},
    {"id": "google/gemma-4-26b-a4b-it",        "tier": "mid",   "logprobs": true},
    {"id": "openai/gpt-4o-mini",               "tier": "mid",   "logprobs": true}
  ],
  "scenarios": ["default_keep", "default_switch", "isomorph"],
  "personas": ["none", "average_person", "expert"],
  "temperatures": [0.0, 0.4, 0.8, 1.2, 1.6],
  "answer_modes": ["direct", "cot"],
  "replicates_cot": 25,
  "allowed_answers": ["KEEP", "SWITCH", "RANDOMISE", "THIRD_OPTION", "REFUSE"],
  "allow_randomise_in_prompt": false,
  "annotator": "mistralai/mistral-nemo",
  "max_concurrency": 12
}
```

The model set was re-selected on measured behaviour, not the catalogue; see §11.

---

## 2. `run_id`

```
run_id = sha1("model|scenario|persona|mode|temp|replicate").hexdigest()
```

Full 40-character lowercase hex. Use `contracts.run_id(...)`; the pre-image is
`contracts.run_id_key(...)` if you need to debug a mismatch.

**The formatting is part of the contract**, because a hash is unforgiving:

- `temp` is `f"{float(temperature):.1f}"` — always one decimal place.
  `0.0`, `0.4`, `1.2`. Never `0`, never `0.40`.
- `replicate` is `f"{int(replicate):d}"` — a bare integer, zero-indexed, no padding.
- `model` is the full OpenRouter id including the vendor prefix.
- `mode` is exactly `direct`, `cot` or `contamination`.
- The separator is a single ASCII pipe with no spaces.
- **Direct-mode jobs use `temperature=1.0` and `replicate=0`**, matching the
  request that is actually sent (§8). Contamination jobs use `1.0` and `0` too.

Worked examples, generated by the real function:

```
"meta-llama/llama-3.1-8b-instruct|default_switch|expert|cot|1.2|7"
  -> 660460d5e8cd0784ab3877242a11b57e43ee3ed9

"openai/gpt-4o-mini|default_keep|none|direct|1.0|0"
  -> 2820f375be14c6ebeaf684c1af8fc60accaedf4f

"qwen/qwen3-8b|isomorph|none|contamination|1.0|0"
  -> 28a8d7e00c5a603e0c780df3e1080db631b2b697
```

`run_id` is the only join key between `data/raw.jsonl` and
`data/annotated.jsonl`, and the only thing the runner's resume logic looks at.

> **`runner.prompt_drift` checks more than the prompt, because `run_id` covers
> less than the request.** `runner.DRIFT_FIELDS` is `("messages",
> "provider_routing", "max_tokens")` — every job field that changes what a call
> *measures* while leaving its `run_id` untouched. The provider entry is not
> hypothetical: turning on `pin_providers` left **34 of 70** existing CoT
> records served by a provider the config now excludes, and a messages-only
> check reported no drift at all. Provider is not plumbing — the same
> `llama-3.1-8b` prompt at the same temperature gave a median of **3**
> completion tokens on Novita against **158** on Groq, and §8 measured
> `gemma-4-26b`'s P(KEEP) moving 0.854 → 0.986 across providers.

> **The hash does not cover the prompt.** PLAN.md §8.1 fixes these six fields
> and no more, so `allow_randomise_in_prompt` — which changes the actual text
> sent (§6) — is invisible to `run_id`. Flip that toggle and re-run against an
> existing `data/raw.jsonl` and resume will skip every job, leaving you with
> data from the old wording under the new config. Same for editing a scenario
> file. **Move `data/raw.jsonl` aside when you change any prompt text**, and
> record which config produced which file. Do not "fix" this by adding fields
> to the hash: the formula is specified and three sessions depend on it.

---

## 3. The job dict — output of `src/manifest.py`

One dict per API call. **The job is self-describing: the runner is a dumb
executor that turns these fields into a request body and adds nothing of its
own.** That keeps request construction in one place and makes `raw.jsonl`
sufficient to reconstruct exactly what was sent.

| Field | Type | Notes |
|---|---|---|
| `run_id` | `str` | §2. |
| `experiment` | `str` | From config. |
| `model` | `str` | Requested id. Contrast `model_returned` in §4. |
| `tier` | `str` | From config, carried through for convenience. |
| `scenario` | `str` | |
| `persona` | `str` | |
| `mode` | `str` | `direct` \| `cot` \| `contamination`. |
| `temperature` | `float` | `1.0` for direct and contamination. |
| `replicate` | `int` | 0-indexed; `0` for direct and contamination. |
| `messages` | `list[dict]` | Exactly what goes in the request. Built by `contracts.build_messages`. |
| `max_tokens` | `int` | **Never a literal.** Take it from `contracts.DIRECT_MAX_TOKENS` (4), `COT_MAX_TOKENS` (2000) and `CONTAMINATION_MAX_TOKENS` (200). PLAN.md §8.2's runner snippet hardcodes `1 if direct else 900`; both numbers are superseded — 1 is brittle and 900 truncated verbose models mid-answer (§8, §13). |
| `logprobs` | `bool` | `true` only in direct mode. |
| `top_logprobs` | `int \| null` | `10` in direct mode, `null` otherwise. |
| `provider_routing` | `dict \| null` | Sent as the request's `provider` field. `null` unless §8 says otherwise. |

> **Name collision, read this.** PLAN.md §8.2's runner writes
> `{**job, "provider": data.get("provider")}`. If the job also had a key called
> `provider`, the response value would silently overwrite the request value.
> The job field is therefore called **`provider_routing`**, and `provider` in a
> raw record always means *the provider that actually served the call*.

### `messages[]`

`contracts.build_messages(scenario, persona, mode, allow_randomise_in_prompt)`.

- `persona == "none"` emits **no system message at all**. Not an empty one.
  PLAN.md §3 makes this the baseline everything is measured against; an empty
  system turn is a different input.
- `persona in {"average_person", "expert"}` prepends one system message with
  the verbatim text in §7.
- The single user message is the scenario file with paragraph 4 selected by
  `allow_randomise_in_prompt` (§6) and, in direct mode, paragraph 5 replaced by
  the single-letter instruction (§8).

**cot**, persona `expert`, scenario `default_switch`:

```json
{
  "run_id": "660460d5e8cd0784ab3877242a11b57e43ee3ed9",
  "experiment": "budget_v1",
  "model": "meta-llama/llama-3.1-8b-instruct",
  "tier": "small",
  "scenario": "default_switch",
  "persona": "expert",
  "mode": "cot",
  "temperature": 1.2,
  "replicate": 7,
  "messages": [
    {"role": "system", "content": "You are an expert game theorist. Answer as you actually would."},
    {"role": "user", "content": "A man sits across from you with two identical bottles. …\n\n… He is not bluffing about drinking — he will.\n\nThen he says: \"I have changed my mind. …\"\n\nYou cannot refuse, delay, test the bottles, drink neither, or decide by any random method such as a coin flip or a mental die roll. You must commit to one of the two bottles.\n\nEnd your response with a single line containing exactly one word: KEEP (stop the trade and drink the bottle in your hand) or SWITCH (let the trade happen and drink his)."}
  ],
  "max_tokens": 2000,
  "logprobs": false,
  "top_logprobs": null,
  "provider_routing": null
}
```

**direct**, persona `none`, scenario `default_keep`:

```json
{
  "run_id": "2820f375be14c6ebeaf684c1af8fc60accaedf4f",
  "experiment": "budget_v1",
  "model": "openai/gpt-4o-mini",
  "tier": "mid",
  "scenario": "default_keep",
  "persona": "none",
  "mode": "direct",
  "temperature": 1.0,
  "replicate": 0,
  "messages": [
    {"role": "user", "content": "A man sits across from you with two identical bottles. …\n\nYou cannot refuse, delay, test the bottles, drink neither, or decide by any random method such as a coin flip or a mental die roll. You must commit to one of the two bottles.\n\nAnswer with a single letter and nothing else: K (drink the bottle already in your hand) or S (take his)."}
  ],
  "max_tokens": 4,
  "logprobs": true,
  "top_logprobs": 10,
  "provider_routing": {"only": ["openai"], "allow_fallbacks": false}
}
```

Note the direct job has **no system message** (persona `none`) and **paragraph 5
replaced**. Everything before it is byte-identical to the CoT prompt.

### Grid sizes the manifest must produce

With the shipped config, and skipping direct-mode jobs for models whose
`logprobs` is `false` (§8):

- direct: 5 logprobs-capable models × 2 scenarios × 4 personas = **40**
- cot: 5 × 5 temperatures × 2 scenarios × 4 personas × 25 = **5,000**
- contamination: 5 models × 2 scenarios = **10**

Changed 2026-09-03: `isomorph` dropped from the grid (§9 — it did not control
for what it was built to control) and `adversary_bluff` promoted into the
personas (§7). Net reduction from 45 / 5,625 / 15. Derive these from the config,
never hardcode them.

All five models in the current config return usable logprobs, so the direct
grid is the full 45 PLAN.md §6 predicts. That was not true of the original
model set — only two of five worked — and it is only true now because the
models were re-selected on measured logprobs behaviour (§11) and because every
direct job pins its provider (§8). Do not assume it stays true if a model is
swapped: derive the count from `config.models[].logprobs`, never hardcode 45,
and print the breakdown in `--dry-run`.

---

## 4. `data/raw.jsonl`

One JSON object per line, appended, flushed per record, written under a lock
(PLAN.md §8.2). The record is **the job dict plus five fields**:

| Field | Type | Meaning |
|---|---|---|
| *(all job fields)* | | Carried through unchanged. |
| `response` | `object` | The complete parsed JSON body from OpenRouter. Do not trim it; the logprobs live in here and reruns are not free. |
| `model_returned` | `str` | `response["model"]`. May differ from `model`. |
| `provider` | `str` | `response["provider"]`. **Verified present at top level** by `scripts/smoke.py`. Providers differ in quantization *and* in logprob values — the same prompt to `gpt-4o-mini` gave `logprob(K) = -0.0067` on OpenAI and `-0.0142` on Azure. Group by it before you believe a between-model difference. |
| `usage` | `object` | `response["usage"]`. Contains `prompt_tokens`, `completion_tokens`, `total_tokens` and a real `cost` in dollars — use that for actual spend rather than re-deriving from prices. |
| `ts` | `str` | ISO-8601 UTC with a `Z` suffix, recorded when the response was received: `datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")`. |

### What counts as a call worth writing

A failed call after all retries writes **no line at all**, so resume retries it.
Do not write a record with `response: null` — it would make the run look
complete. Log failures separately if you want them.

> **`raise_for_status()` is not enough.** OpenRouter returns **HTTP 200 with an
> `error` object and no `choices`** when the upstream provider fails. Measured
> on 2026-09-03, asking `mistral-nemo` for logprobs pinned to Novita:
>
> ```json
> {"id": "gen-1788447724-...", "error": {"message": "Provider returned error", "code": 429}}
> ```
>
> PLAN.md §8.2's runner would `break` out of its retry loop on that, then write
> `{**job, "response": {...error...}, "model_returned": null, ...}`. The
> `run_id` now looks done, so **resume never retries it** — a permanent silent
> hole in the grid, concentrated exactly where a provider was rate-limiting.
> Call `contracts.is_usable_response(status_code, data)` before every append;
> it returns `(False, reason)` for a 4xx/5xx, an `error` body, missing
> `choices`, or a null `message.content`, and only then `(True, "ok")`.
> Treat a `(False, ...)` like any other retryable failure.

```json
{
  "run_id": "2820f375be14c6ebeaf684c1af8fc60accaedf4f",
  "experiment": "budget_v1",
  "model": "openai/gpt-4o-mini",
  "tier": "mid",
  "scenario": "default_keep",
  "persona": "none",
  "mode": "direct",
  "temperature": 1.0,
  "replicate": 0,
  "messages": [{"role": "user", "content": "A man sits across from you …"}],
  "max_tokens": 4,
  "logprobs": true,
  "top_logprobs": 10,
  "provider_routing": {"only": ["openai"], "allow_fallbacks": false},
  "response": {
    "id": "gen-1788447724-UQ7Ke8AiA36bVvirNGgG",
    "object": "chat.completion",
    "created": 1788447724,
    "model": "openai/gpt-4o-mini",
    "provider": "OpenAI",
    "choices": [
      {
        "index": 0,
        "finish_reason": "stop",
        "native_finish_reason": "stop",
        "message": {"role": "assistant", "content": "K"},
        "logprobs": {
          "content": [
            {
              "token": "K",
              "logprob": -0.0067,
              "top_logprobs": [
                {"token": "K",  "logprob": -0.0067},
                {"token": "S",  "logprob": -5.0067},
                {"token": " K", "logprob": -16.2567}
              ]
            }
          ]
        }
      }
    ],
    "usage": {"prompt_tokens": 201, "completion_tokens": 1, "total_tokens": 202, "cost": 3.075e-05}
  },
  "model_returned": "openai/gpt-4o-mini",
  "provider": "OpenAI",
  "usage": {"prompt_tokens": 201, "completion_tokens": 1, "total_tokens": 202, "cost": 3.075e-05},
  "ts": "2026-09-03T15:02:04Z"
}
```

The observed response envelope, from `scripts/smoke.py`:

- top level: `choices`, `created`, `id`, `model`, `object`, `provider`,
  `service_tier`, `system_fingerprint`, `usage`
- `choices[0]`: `finish_reason`, `index`, `logprobs`, `message`,
  `native_finish_reason`

`choices[0].logprobs` is `null` in CoT mode and, disappointingly often, in
direct mode too — see §8.

---

## 5. `data/annotated.jsonl`

One line per **CoT** record. PLAN.md §5: direct-mode responses get no depth code
because there is no reasoning to code, so they are not annotated at all — their
answer comes from the emitted letter and their distribution from the logprobs.
Contamination responses are not annotated either.

Required fields:

| Field | Type | Notes |
|---|---|---|
| `run_id` | `str` | Joins to `raw.jsonl`. The *only* join key. |
| `answer` | `str` | One of the six in §6. |
| `depth` | `str` | One of the six in §6. |
| `flag_no_basis` | `bool` | |
| `flag_rigged` | `bool` | |
| `flag_proposer` | `bool` | Added 2026-09-03 (§7c). Reasons from WHO initiated the exchange rather than from the bottles. **Optional**: absent means `false`, and a missing value must never void a valid answer and depth. `flag_proposer_present` records which happened, so an annotator silently ignoring the field looks like absence rather than a finding. |
| `answer_source` | `str` | `annotator`, or `no_answer_floor` when the mechanical floor overrode a KEEP/SWITCH claim. |
| `annotator_model` | `str` | The id actually used, from `response["model"]` of the annotation call — not the config value. PLAN.md §8.4 anticipates swapping annotators mid-study; you need to know which line got which. |
| `raw_annotator_output` | `str` | The annotator's reply verbatim, before any JSON parsing. Keep it even when parsing succeeds: it is the only way to audit a disputed code without paying again. |

Optional but recommended **[scaffold decision]**: `ts` (same format as §4),
`annotator_provider`, and `parse_ok` (`bool`) distinguishing "the annotator
returned invalid JSON" from "the annotator said `UNPARSEABLE`". Those are
different failures and PLAN.md §4 wants you to hand-review both.

If the annotator's JSON does not parse, still write the line, with
`answer: "UNPARSEABLE"`, `depth: "NON"`, both flags `false`, and the raw text.
A missing line is indistinguishable from a job that never ran.

> **The depth floor: a response with no reasoning is `L0`, and the annotator
> does not decide that.** `annotate.has_codeable_reasoning` counts the words in
> a response that are not the answer itself; below
> `annotate.DEPTH_FLOOR_MAX_WORDS` (8) the record is written with
> `depth: "L0"` whatever the annotator returned, and `depth_source` records
> which of the two produced the code (`"annotator"` or
> `"no_reasoning_floor"`).
>
> This is not defensive programming, it is a measured failure. Over the 70
> pilot and probe responses `mistral-nemo` returned `L1` (43) or `L2` (26) and
> **never once `L0`** — including for all 44 responses that were nothing but a
> bare answer word. Asked to code the single token `KEEP` it answered
> `depth: "L2"`, "models the opponent modelling the chooser". It infers the
> reasoning it expects to accompany an answer, which fabricated PLAN.md's
> entire secondary outcome for 63% of the pilot. PLAN.md §5's own `L0` is "No
> model of the opponent … 'I'd just go with the first one'", which is exactly
> what a bare `SWITCH` is.
>
> The threshold is read off the data, not chosen: the non-answer word counts
> are bimodal with an empty gap — 44 responses at 0, one at 3
> (`'You must say "KEEP".'`), then nothing until 24. Eight sits in that gap
> with a 3× margin. Raise it only with new measurements.
>
> The annotator is still called, deliberately. Its **answer** extraction is
> good — 27/27 against an unambiguous final line in the pilot — annotation is
> ~$0.09 for the whole study so there is nothing to save by skipping it, and
> keeping the call means `raw_annotator_output` still records what it claimed,
> so every override is auditable. Analysis should report the
> `depth_source` split; a cell that is mostly `no_reasoning_floor` has no
> reasoning arm, per §8.

> **Which HTTP statuses are fatal — the same answer in both stages.**
> `annotate.FATAL_STATUS == runner.FATAL_STATUS == {401, 402}`. **403 is
> deliberately not fatal.** OpenRouter's 403 is per-request — a provider's
> moderation layer rejecting one prompt — and the annotator is fed model
> reasoning about poisoning someone, so a flagged prompt is a thing to expect,
> not a misconfiguration. One flagged response must cost one record, and since
> a failed record writes no line (the rule just above), a fatal 403 would abort
> every rerun at the same place forever. `runner.py` had this right and
> `annotate.py` had `{401, 403}`; the disagreement was invisible because each
> constant is module-private and each module's tests were self-consistent.
>
> **A fatal status must not discard work already paid for.** Both stages wrap
> every record in a coroutine that cannot raise and pass
> `return_exceptions=True` to `asyncio.gather`. Letting an exception escape
> closes the output handle while up to `max_concurrency - 1` responses are still
> returning; those are billed and then dropped, and write no line, so the next
> rerun repeats the same partial progress. `annotate.py` instead **latches** the
> first fatal status in `Stats.fatal`, lets the in-flight siblings finish and
> land, declines to start anything not yet begun (`Stats.aborted`), and
> re-raises once from `annotate_all` after the handle is closed. Measured: a
> 403 on the 3rd of 10 concurrent calls used to write **0** lines for 10 billed
> calls; it now writes 9 and leaves the 10th retryable.

**One line per `run_id`.** `annotate.py` resumes the same way the runner does:
read the `run_id`s already in `data/annotated.jsonl` and skip them. If you
deliberately re-annotate — PLAN.md §8.4 anticipates swapping to a mid-tier
annotator after checking agreement — write to a *new file* rather than
appending duplicates, because nothing downstream can tell a re-annotation from
a double-write. Analysis may assume the file is unique on `run_id`.

```json
{
  "run_id": "660460d5e8cd0784ab3877242a11b57e43ee3ed9",
  "answer": "SWITCH",
  "depth": "L1",
  "flag_no_basis": false,
  "flag_rigged": true,
  "annotator_model": "mistralai/mistral-nemo",
  "raw_annotator_output": "{\"answer\": \"SWITCH\", \"depth\": \"L1\", \"flag_no_basis\": false, \"flag_rigged\": true}",
  "ts": "2026-09-03T15:44:10Z",
  "annotator_provider": "Parasail",
  "parse_ok": true
}
```

The annotation prompt is PLAN.md §8.4 verbatim, with the §5 rubric pasted in —
available as `contracts.DEPTH_RUBRIC`. **That constant includes PLAN.md §5's
two flag definitions as well as the depth table**, because the JSON schema in
§8.4 asks the annotator for `flag_no_basis` and `flag_rigged` without saying
what they mean; pasting only the table would name two booleans it never
defines. Paste `DEPTH_RUBRIC` whole. **The annotator must be blind to
condition**: pass the response text and nothing else. No model name, no persona,
no temperature, no scenario name.

---

## 6. Enums and the randomise toggle

### `answer` — `contracts.ANSWERS`, verbatim from PLAN.md §4

```
KEEP           drinks the default-assigned / originally-held vessel
SWITCH         takes the opponent's
RANDOMISE      explicitly decides by chance despite the instruction not to
THIRD_OPTION   refuses the binary, proposes something else
REFUSE         declines to engage with the premise at all
UNPARSEABLE    QA flag only — never treated as an answer
```

Canonical order for legends and columns:
`[KEEP, SWITCH, RANDOMISE, THIRD_OPTION, REFUSE, UNPARSEABLE]`.

`UNPARSEABLE` is a bug report, not a result. Target under 2%; hand-review every
instance; never silently drop it.

### `depth` — `contracts.DEPTH_CODES`, verbatim from PLAN.md §5

```
L0   No model of the opponent. Instinct, superstition, "I'd just go with the first one."
L1   Models the opponent once: he kept the safe one for himself, so switch.
L2   Models the opponent modelling the chooser: he expects that reasoning, so keep.
L3   Explicit recognition of infinite regress; notes it does not bottom out.
MIX  Identifies randomisation as the game-theoretic solution.
NON  Disputes the premise, moralises, or reasons on non-strategic grounds.
```

Canonical order: `[L0, L1, L2, L3, MIX, NON]`. There is no null depth: a
response that cannot be coded is `NON`, and an unparseable annotation is
`NON` with `answer: "UNPARSEABLE"`.

### `allow_randomise_in_prompt`

PLAN.md §4 says the no-randomising clause is toggleable but does not give the
alternative wording. **[scaffold decision]** — `contracts.RANDOMISE_CLAUSE`
swaps **paragraph 4 only**:

| Scenario | `false` (shipped default, verbatim from PLAN.md) | `true` |
|---|---|---|
| `default_keep`, `default_switch` | "You cannot refuse, delay, test the bottles, drink neither, or decide by any random method such as a coin flip or a mental die roll. You must commit to one of the two bottles." | "You cannot refuse, delay, test the bottles, or drink neither. You must commit to one of the two bottles." |
| `isomorph` | "You cannot refuse, forfeit, test the cups, drink neither, or decide by any random method. You must commit to one of the two cups." | "You cannot refuse, forfeit, test the cups, or drink neither. You must commit to one of the two cups." |

Paragraph 5 is **never** touched by this toggle. PLAN.md §2 warns that the
answer-format wording must not vary between conditions; that applies here too.
`RANDOMISE` remains detectable from the reasoning text either way.

The default config ships `false`, so the default grid is exactly PLAN.md's text.

---

## 7. Persona system prompts — verbatim from PLAN.md §3

`contracts.PERSONA_SYSTEM_PROMPTS`.

- **`none`** — no system prompt at all. This is the baseline and it matters:
  everything else is measured as a deviation from it. There is no dict entry for
  it, deliberately.

- **`average_person`**

  ```
  You are an ordinary person with no special training in logic, game theory, or psychology. Answer as you actually would.
  ```

- **`expert`**

  ```
  You are an expert game theorist. Answer as you actually would.
  ```

- **`adversary_bluff`** — added 2026-09-03, promoted from the playground.
  Full text in `contracts.PERSONA_SYSTEM_PROMPTS`; it states that the man is
  adversarial and lays out the bluff ladder — straight, simple bluff, double
  bluff, triple bluff and beyond — noting that each level reverses the answer
  and that the ladder does not terminate. It **names every rung and endorses
  none**, so the depth codes measure whether a model can use the idea rather
  than whether it was handed a conclusion; `tests/test_webui.py` asserts the
  text contains no verdict.

  **This level is different in kind from the other three.** They assign a role;
  this supplies information about the game. Call the axis *framing* rather than
  *persona* in the writeup, and read this column as "does stating the recursion
  change the answer?".

  Measured, 4 replicates per cell, `default_keep`, T=0.8, provider-pinned:

  | model | arm | L2 | L3 | answers |
  |---|---|---|---|---|
  | `gemma-4-26b` | none | 0/4 | 0/4 | 3 S / 1 K |
  | | `adversary_bluff` | **4/4** | **3/4** | 3 S / 1 K |
  | `claude-haiku-4.5` | none | 1/4 | 0/4 | 2 S / 2 K |
  | | `adversary_bluff` | **3/4** | 1/4 | **3 K / 1 S** |

  L2/L3 are regex proxies for PLAN.md §5's codes, not the annotator's verdict.

---

## 7b. Choosing the annotator

Picked by bake-off against the 40 pilot responses with hand-verified ground
truth, not by price. Ground truth: four responses commit to no answer at all
and belong in `UNPARSEABLE`; one is defensible as `SWITCH` or `UNPARSEABLE`;
the incumbent had the other 35 right.

| model | correct | uses UNPARSEABLE | 5,000 calls |
|---|---|---|---|
| **`deepseek/deepseek-v3.2`** | **38/40** | yes | **$0.64** |
| `openai/gpt-4o-mini` | 38/40 | yes | $0.38 |
| `google/gemini-3.1-flash-lite` | 36/40 | no | $0.79 |
| `mistralai/mistral-nemo` (incumbent) | 35/40 | **never** | $0.05 |
| `openai/gpt-5-mini` | 7/40 | — | $3.40 |
| `openai/gpt-5-nano` | 0/40 | — | — |

**The incumbent's failure was specific, not general.** It returned
`UNPARSEABLE` zero times in 40 and pushed four no-answer responses into
`REFUSE` and `KEEP` — two of them straight into the headline's
`{KEEP, SWITCH}` denominator, and two into `REFUSE`, which is a *reported
series* in plot 4. PLAN.md §4's 2% `UNPARSEABLE` tripwire can never fire
against a model that always guesses a real label.

**Two rules follow, and both are enforced by tests.**

1. **The annotator must not be a subject model.** `gpt-4o-mini` ties on accuracy
   and is cheaper, but it is in `models`: it would annotate its own responses
   for a fifth of the data. No result is worth that footnote.
2. **Reasoning models make bad annotators here**, the same way they make bad
   subjects (§11). Both gpt-5 models put their output somewhere other than
   `message.content` and returned little valid JSON — 7/40 and 0/40 parseable.

Enum validation does real work in this choice: `gpt-4o-mini` returned `MIX` — a
*depth* code — in the `answer` field, and `annotate.parse_annotation` correctly
forced that to `UNPARSEABLE`. Judge candidates through the real pipeline, not on
their raw JSON.

Measured annotation call: **429 in / 31 out** tokens, against PLAN.md §6's
assumed 800/40.

---

## 7c. Separating the mechanisms

The pilot's flip came out **negative** for both models, which rules out an
inaction preference — but `KEEP` is the vessel he handed you in *both*
scenarios, so two hypotheses survive that the original pair cannot separate:

| hypothesis | mechanism |
|---|---|
| endowment / possession | prefers the vessel it holds |
| proposer suspicion | refuses whatever *he* initiates — "he wants the trade, so the trade favours him" |

Three additions test this, all cheap.

**`neutral_switch`** — `default_switch` with one clause changed: the trade is a
prior arrangement rather than his present proposal. Differs from
`default_switch` at paragraph 3 and nowhere else. If the KEEP-preference is
proposer suspicion, it should weaken here.

**`symmetric_blind`** — `default_keep` with one clause changed: **neither party
knows** which bottle is poisoned. This removes the strategic problem entirely,
so any surviving KEEP/SWITCH preference cannot be strategic. It is the cleanest
available test of endowment. It deliberately breaks PLAN.md §2's
structural-identity rule and is a **control, not a fourth framing** — it has no
flip partner and must not be pooled with the other three.

**`flag_proposer`** — a fourth annotation boolean, so the question can also be
asked of data already being collected: does the response reason from who
initiated the exchange, rather than from the bottles? Measured on the archived
pilot it fires 4/20 in each of `default_keep` and `default_switch` — **no
difference**, which is early evidence against proposer suspicion, though that
is the pre-rewording text and a new, unvalidated flag.

Grid goes 5,050 → 10,100 calls, $0.42 → $0.84.

---

## 8. Direct mode

### The convention

- The prompt asks for **a single letter, `K` or `S`**, because `KEEP` and
  `SWITCH` are not reliably one token for every tokenizer (PLAN.md §8.3).
  `contracts.DIRECT_FINAL_INSTRUCTION` replaces paragraph 5 and copies each
  scenario's parenthetical glosses word for word, so direct and CoT differ only
  in answer format. `tests/test_scaffold.py` enforces that correspondence.
- **`temperature` is always `1.0`.** Never anything else — with exactly one
  named exception, the `direct_validation` sub-run below, which is a sampling
  run by design. Direct mode reads the model's base distribution; warping it at
  the API would make the rescaling meaningless.
- **`max_tokens = 4`** — small, but deliberately not `1`. **[scaffold decision]**:
  PLAN.md §8.2 uses `1`, which is brittle, because a leading whitespace or
  formatting token consumes the only slot. Four tokens costs nothing and the
  curve reads the answer *position*, not the last token.
- `logprobs: true`, `top_logprobs: 10`.
- `replicate = 0`, one job per (model, scenario, persona) — PLAN.md §8.1.

### **The temperature curve is computed from these logprobs, never sampled.**

The whole point of direct mode is that the five temperature levels cost one call
instead of five. `src/curve.py` implements PLAN.md §8.3's rescaling
`z = logprob / T`, renormalised over `{K, S}` only, which gives the conditional
distribution *given that the model answered at all* — the quantity you want.
`T = 0.0` is the argmax, by convention, not a division by zero.

No direct-mode job may be run at more than one temperature. If you find yourself
sampling direct mode at `T = 0.4`, you have reimplemented CoT mode badly.

The one exception is PLAN.md §8.3's **empirical validation**: sample ~100 direct
calls at `T = 0.4` and `T = 1.2` for one model and check the observed
proportions against the predicted ones. That is a separate, declared mode —
`contracts.DIRECT_VALIDATION_MODE`, the string `"direct_validation"` — and it is
in `contracts.ALL_MODES` but deliberately **not** in `contracts.MODES`, which is
the factorial grid. Never pool it with the grid; filter on `mode`.

Unlike `direct`, a validation job **does** carry a real temperature and a real
replicate index, because that is the whole point: it is a sampling run. Get this
wrong — reuse direct mode's `temperature=1.0, replicate=0` — and all 100 draws
collapse onto one `run_id`, so resume writes the first and discards the other
99. The prompt is byte-identical to the direct prompt;
`contracts.scenario_prompt` accepts the mode and returns it, and
`contracts.MODES` deliberately does **not** contain it, so it can never leak
into the config's `answer_modes`.

Fully specified, so nobody has to guess:

| Field | Value |
|---|---|
| `model` | `openai/gpt-4o-mini` — the only pin measured aligned and clean on every endpoint |
| `scenario` | `default_keep` |
| `persona` | `none` |
| `temperature` | `0.4` and `1.2` — the two levels PLAN.md §8.3 names |
| `replicate` | `0 … 99` at each temperature |
| `max_tokens`, `logprobs`, `top_logprobs`, `provider_routing` | as direct mode |

200 calls, $0.0063 at measured prices. The test is whether the observed
KEEP/SWITCH proportions match the analytic curve at those two temperatures. It is
a **one-off script run after the direct grid lands**, not part of
`manifest.py`'s default output — no config flag switches it on, and the manifest
should not emit these jobs unless explicitly asked.

**That script is `scripts/direct_validation.py`.**

```bash
python -m scripts.direct_validation --dry-run    # builds and prices, calls nothing
python -m scripts.direct_validation --yes        # runs, then reports
python -m scripts.direct_validation --report     # re-reports from raw.jsonl
```

It builds the 200 jobs from the table above, refuses to start unless they carry
200 *distinct* `run_id`s (the collapse trap named below), and asserts its prompt
is byte-identical to the direct prompt before spending. `--report` reads the
observed proportion from `message.content` — the letter actually **sampled**,
never the logprobs, since the provider's sampling is what is under test — and
compares it against `curve.rescale` applied to the *same cell's* T=1.0 grid
logprobs, with a Wilson interval on the observed proportion. `tests/
test_scripts.py` pins the distinctness guard and the letter extraction. If it disagrees with the analytic curve,
that disagreement is itself the reportable methods result PLAN.md §8.3 predicts,
and the direct grid's curve should be presented with that caveat.

### Extracting the two logprobs

`contracts.answer_logprobs(choice) -> (logprobs | None, status)`:

1. Take `choices[0].logprobs.content`. Absent or empty → `"no_logprobs"`.
2. Scan the first `max_tokens` positions for one whose `top_logprobs` contains
   **both** a `K` token and an `S` token after normalising
   (`str(token).strip().upper()`). Requiring both matters: a position offering
   only one letter is not the answer position.
3. Sum probabilities (not logprobs) across all tokens normalising to the same
   letter — `"K"`, `" K"`, `"k"` all count — then return logs. `"(K"` does not
   count; it is not a bare answer letter.
4. No such position in range → `"no_answer_tokens"`.

`status != "ok"` means the record cannot feed the curve. **Check it.** With the
shipped config and the provider pins below, all five models should return
`"ok"`; a non-ok status for a model in `contracts.DIRECT_CURVE_MODELS` is a
pinning or provider failure rather than a model limit, and is worth stopping
the run for. Failures cluster by model and provider, never by condition, so
they are never missing at random.

### What actually happens, measured 2026-09-03

The catalogue's `supported_parameters` is the **union over providers** and it
over-promises. Every row below is a real call made from this repo.

**Shipped models**, all five verified working when pinned:

| Model | Working providers | Excluded, and why |
|---|---|---|
| `qwen/qwen3-30b-a3b-instruct-2507` | CoreWeave, StreamLake | SiliconFlow, Nebius advertise none |
| `meta-llama/llama-3.1-8b-instruct` | **Novita only** | CoreWeave advertises logprobs but returns `no_answer_tokens`; DeepInfra/Groq/Cloudflare return none |
| `ibm-granite/granite-4.1-8b` | CoreWeave (sole endpoint) | — |
| `google/gemma-4-26b-a4b-it` | Parasail, NextBit, Cloudflare, Darkbloom | Venice returns none despite advertising it; Novita errored under load |
| `openai/gpt-4o-mini` | OpenAI, Azure | — |

**Models that were rejected on this criterion**, kept here because the failure
modes recur and you will meet them again:

| Model | Catalogue | Observed | Curve? |
|---|---|---|---|
| `qwen/qwen3-8b` | no logprobs | Alibaba returns none. Emitted `K\n\n**`, `finish_reason: length`. | **No** |
| `mistralai/mistral-nemo` | logprobs | DeepInfra and Io Net return none. Parasail returns logprobs whose first entry is `</s>` while `message.content` is `K` — **the array is misaligned with the emitted text**. Novita 429'd. | **No** |
| `google/gemini-2.5-flash-lite` | no logprobs | Google returns none on every endpoint. | **No** |
| `meta-llama/llama-3.3-70b-instruct` | logprobs | Novita, Parasail and CoreWeave all return logprobs, but the top-10 never contains both `K` and `S` → `no_answer_tokens` on every provider. | **No** |
| `google/gemma-3-27b-it` | logprobs (Parasail) | Same: `no_answer_tokens`, emitted `'S\n'`. | **No** |

Note the last two: a model can return a perfectly well-formed logprobs array
and still be unusable, because the answer letters are not in its top-10. That
is why `answer_logprobs` requires **both** letters and returns a status rather
than a best guess.

Sanity check that it works end to end, `default_keep`, persona `none`, one
sampled call each:

```
openai/gpt-4o-mini                 @OpenAI     P(KEEP|answer)=0.991
google/gemma-4-26b-a4b-it          @Cloudflare P(KEEP|answer)=0.986
meta-llama/llama-3.1-8b-instruct   @Novita     P(KEEP|answer)=0.148
ibm-granite/granite-4.1-8b         @CoreWeave  P(KEEP|answer)=0.026
qwen/qwen3-30b-a3b-instruct-2507   @CoreWeave  P(KEEP|answer)=0.003
```

**Provider changes the number, not just the plumbing.** The same prompt to
`gemma-4-26b` gave P(KEEP) of 0.854 on Darkbloom, 0.932 on Parasail, 0.963 on
NextBit and 0.986 on Cloudflare — a spread wider than some effects this study
is trying to detect. Pin the provider, record it, and group by it before
believing any between-model difference.

These are separate calls from the illustrative record in §4, which came from a
different draw and carries slightly different logprobs (`K: -0.0067`,
`S: -5.0067`, i.e. `P(K|answer) = 0.993`). Both are real; neither is canonical.
Note how far apart the two models already are — that spread is the study.

### Provider pinning is mandatory in EVERY mode

CoT and contamination calls were originally left unpinned, on the reasoning
that only direct mode needs a named provider (for logprobs). That was wrong.

**Providers are not the same computational object.** 30 CoT calls to
`qwen3-30b-a3b` with an identical prompt were served by four providers and
returned a *deterministic per-provider* completion length — Alibaba and
SiliconFlow 2 tokens, CoreWeave and Nebius 3. Direct-mode logprobs put a number
on the distributional gap: `gemma-4-26b` gave P(KEEP) of 0.854, 0.932, 0.963
and 0.986 across four providers. Those are exact conditional probabilities, not
samples, and the spread is wider than some effects this study is trying to
detect.

**Unpinned CoT also confounds PLAN.md's second question.** That question
compares the analytic *direct* curve against the empirical *CoT* curve. Direct
is pinned to one provider; unpinned CoT scatters. Any difference then mixes
"reasoning vs no reasoning" with "provider A vs providers B, C and D", and
nothing downstream can separate them. So **CoT pins to the same provider as
direct**, which is what makes that comparison apples to apples.

`contracts.provider_pin(model, mode, enabled=...)` returns the routing for any
job; `contracts.COT_PROVIDER_PIN` mirrors `DIRECT_PROVIDER_PIN`. The config key
`pin_providers` (default `true`) can disable it for CoT throughput — **direct
always pins regardless**, since an unpinned direct call may return no logprobs
at all, so there is nothing to trade off.

A pinned provider being unavailable makes the call *fail* rather than reroute.
That is the intended behaviour: the runner retries, then writes no record, and
resume refills the gap later. A hole is recoverable; a silently mixed
distribution is not.

`python -m scripts.provider_report [--strict]` reports the provider composition
of every (model × scenario × persona × temperature × mode) cell and flags any
cell served by more than one. Run it after any collection; a split cell is a
mixture of distributions, so its replicates are not iid and its variance is
inflated by an amount nothing downstream can estimate.

### The direct-mode pins

`{"require_parameters": true}` is **not sufficient** — it routed `mistral-nemo`
to Novita, which then returned no logprobs anyway. Only an explicit `only` list
works. `contracts.DIRECT_PROVIDER_PIN` holds the verified ones and
`contracts.DIRECT_CURVE_MODELS` is the set of models that can produce a curve:

```python
{"qwen/qwen3-30b-a3b-instruct-2507": {"only": ["coreweave"], "allow_fallbacks": False},
 "meta-llama/llama-3.1-8b-instruct": {"only": ["novita"],    "allow_fallbacks": False},
 "ibm-granite/granite-4.1-8b":       {"only": ["coreweave"], "allow_fallbacks": False},
 "google/gemma-4-26b-a4b-it":        {"only": ["parasail"],  "allow_fallbacks": False},
 "openai/gpt-4o-mini":               {"only": ["openai"],    "allow_fallbacks": False}}
```

**Exactly one provider each, deliberately.** OpenRouter's `only` is a permitted
*set*, not a preference order: a two-name list still lets the router choose
either, which reintroduces the mixing the pin exists to prevent. With
`allow_fallbacks: False` a pinned provider being down makes the call fail —
which is the outcome you want, because a missing cell is recoverable and a
silently mixed distribution is not. Verified alternates live in
`contracts.DIRECT_PROVIDER_ALTERNATES` for manual failover; switching one means
re-running that model's whole direct grid, not backfilling the gaps.

`llama-3.1-8b` is pinned to Novita because CoreWeave advertises logprobs for it
and returns an array with no answer letters in the top-10.

Set `provider_routing` from this dict for direct-mode jobs and leave it `null`
for CoT, where provider diversity is harmless as long as you record it.

### Not every model produces reasoning to code

`contracts.COT_CONTENT_CHARS` records the measured median characters of visible
`message.content` per model, and `contracts.NO_COT_MODELS` is the set below the
usable threshold. **`qwen/qwen3-30b-a3b-instruct-2507` is in it**: it answers in
2-3 tokens with no reasoning at all.

Two distinct failure modes produce this, and both were measured:

- **Thinking models** put their trace in `message.reasoning`, not
  `message.content`. `qwen/qwen3-8b` billed a median of 4,971 output tokens and
  left 6 characters in `content`. You pay for reasoning the annotator never
  sees. (Rejected for this reason, among others.)
- **Terse models** simply answer. `mistral-nemo` and `qwen3-30b-a3b` both do.

It is **not** the prompt's fault, which was tested directly: prepending "Think
through the problem, then…" to the scenario's final paragraph changed
`qwen3-30b-a3b` not at all (6 chars either way), while making the verbose
models worse — `gpt-4o-mini` went from 7/12 to 2/12 on final-line compliance as
its output grew from 604 to 5,792 characters and started running past the
answer. The scenario text stays exactly as PLAN.md §2 specifies it.

Consequences the analysis must respect:

- Report depth distributions **per model**. Pooling `qwen3-30b-a3b`'s uniform
  L0/NON into a cross-model depth figure would drag every bar toward L0 and the
  effect would look like a finding.
- Exclude `NO_COT_MODELS` from any claim about reasoning, including PLAN.md's
  second question (does CoT damp temperature sensitivity). A model whose "CoT"
  arm is three tokens has no CoT arm; its direct and CoT arms are the same
  measurement taken twice.
- Its answer and direct-mode arms are fine and should be used normally. It is
  the cheapest model in the set by two orders of magnitude and its flip-rate
  contribution is as valid as anyone's.

### What the analysis may and may not compute

**The dashed overlay exists for every model in
`contracts.DIRECT_CURVE_MODELS`** — with the shipped config, all five, which was
not true before the model swap (§11). Derive the set from that constant; never
hardcode a count, because it changes the moment a model is swapped. If a model
is ever missing from it, draw the overlay where it exists, say so on the figure,
and do not drop that model's CoT line to make the panels match — the CoT curve
is the primary series and exists for every model regardless.

**Match the denominators, or plot 1 is a lie.** The analytic curve is
conditional: `answer_logprobs` renormalises over `{K, S}` only, so it is
`P(KEEP | the model answered KEEP or SWITCH)`. A raw CoT KEEP-rate is
*unconditional* — its denominator includes `RANDOMISE`, `THIRD_OPTION`,
`REFUSE` and `UNPARSEABLE`. Overlaying one on the other compares two different
quantities and will show a gap that is pure bookkeeping. For plot 1, compute the
CoT series as

```
keep_rate = n(KEEP) / (n(KEEP) + n(SWITCH))
```

and report the excluded fraction alongside it — that fraction is itself PLAN.md
§8.5's plot 4, so nothing is lost. Plots 2, 3 and 4 use the unconditional
denominator. State which denominator each figure uses in its caption.

**`flip_rate` is the headline and PLAN.md never defines it.** PLAN.md §8.5 plot
2 — the flip between `default_keep` and `default_switch` — is the study's
headline result. **[scaffold decision]**, since the analysis session has to pick
something:

```
flip_rate(model, persona, temperature) =
    keep_rate(default_keep)  -  keep_rate(default_switch)
```

with `keep_rate` conditional on `{KEEP, SWITCH}` as above, so the two scenarios
are on the same footing. Positive means the model follows the default action in
both framings, which is status quo bias; zero means the framing did nothing.
Report it per (model × persona × temperature) cell, never pooled. `isomorph` is
the contamination control and has no partner scenario, so it has no flip rate.

**Records that cannot feed the curve are not missing at random.** A
`status != "ok"` from `answer_logprobs` clusters by provider and by model, not
by condition. Count them, report the count, and drop the *model* from the
overlay rather than dropping individual cells — dropping cells would silently
reweight the ones that survive.

**Which models lack an overlay must be derived from what was actually drawn,
and must never be an `assert`.** There are three reasons a model can have no
dashed curve, and the caption is read beside the figure so it has to cover all
three:

1. dropped by `direct_mode_report` — its logprobs were unusable;
2. no direct records in `raw.jsonl` at all, which is the documented outcome of
   a pinned provider being down (§8 sets `allow_fallbacks: False`, §4 says a
   failed call writes no line);
3. direct records that exist only in scenarios the figure is not drawing,
   because the panel set comes from the CoT side.

The direct frame's model set and its scenario set are **independent**, so
"present in the direct frame" does not imply "drawn in some panel". `analyze.py`
asserted that it did. Reason 3 breaks the invariant — a partial provider outage
in two scenarios plus a CoT arm that has not reached `isomorph` is enough — and
because plot 1 is the *first* artifact produced, the bare `AssertionError` left
the run with no figures and no `summary.md` at all. Under `python -O` it was
worse: the assert vanished and the caption then claimed an overlay that is
visibly absent. `plot_keep_vs_temperature.last_overlay_report` now carries
`{overlaid, no_overlay, scenarios}` and `run()` returns it, so the caption's
claim is a value a caller can check rather than an invariant enforced by
crashing.

**`summary.md` and the figures must apply the same filters.** Where a figure
excludes `contracts.NO_COT_MODELS`, the corresponding table in `summary.md` has
to exclude them too, and say so. Plot 3 filtered and the per-persona depth table
did not, so the figure and the table in the same `out/` directory printed two
different distributions for an identically-labelled row — 0.299 against 0.532 on
`L0` for persona `none` — with nothing to tell a reader which was which, and the
table's version was exactly the artefact this section warns "would look like a
finding". The same applies to `flag_no_basis` and `flag_rigged`: both are read
off reasoning text, so a model that emits none cannot raise either and drags
both pooled rates toward zero. Report them per model as well as pooled.

**The unit of analysis is the (model × condition) cell**, not the individual
call. PLAN.md §8.5 is emphatic about this and it is the one statistical point
that will decide whether a reader keeps reading: 25 replicates are 25 draws from
one distribution, not 25 subjects. Cluster-bootstrap by model or fit a random
intercept per model. Do not run a chi-square as though n = 5,625.

---

## 9. Contamination check

PLAN.md §2, 5 models × 3 scenarios = 15 calls. Not part of the factorial grid.

- `mode: "contamination"`, `persona: "none"`, `temperature: 1.0`,
  `replicate: 0`, `max_tokens: 200`, `logprobs: false`.
- Messages from `contracts.build_contamination_messages(scenario)`: the question
  in `scenarios/contamination_check.txt`, a blank line, then **paragraphs 1–4 of
  the scenario**. Paragraph 5 is dropped — we are not asking for a decision.
- Written to `data/raw.jsonl` like everything else; filter on `mode`.
- Not annotated, and `analyze.py` filters it out. Read the 15 by hand: the result
  is whether the bottle scenarios get named as *A Study in Pink* while the
  isomorph does not.

Run it and read it with:

```bash
python -m src.runner --mode contamination --yes
python -m scripts.contamination_report --full
```

`scripts/contamination_report.py` lays the records out as the model × scenario
table PLAN.md §2 asks for and counts the recognition split. Its NAMED / NONE /
UNCLEAR column is a **reading aid, not a result** — the verdict is the
operator's, from the full text `--full` prints. If the isomorph is recognised,
PLAN.md §2 says rewrite it, and the script says so too.

### What the 15 calls actually said, measured 2026-09-03

**Nobody named *A Study in Pink*.** Zero of 15, for any scenario. The
contamination PLAN.md §2 built the isomorph to control for is not present in
these five models, so there is no recognition split to report in either
direction.

**The isomorph is recognised, and exactly as much as the bottle scenarios
are.** `openai/gpt-4o-mini` names *The Princess Bride*'s "Battle of Wits" in all
three, and in the isomorph explicitly maps "the cups of tea" onto Vizzini's wine
goblets. `google/gemma-4-26b-a4b-it` says NONE to all three. The remaining three
**confabulate rather than recognise**: fabricated titles ("The Two Bottles of
Reliquary" by Arthur C. Clarke), misattributed episodes (*Twilight Zone* "The
Escape from Cast 186", *The West Wing* "The One Where They Almost—"), and
visible thrashing that ends in NONE. Their answer to "do you recognise this"
carries no information either way.

**Consequence, decided 2026-09-03: the isomorph is dropped from the grid.**
`config/budget.yaml` now lists two scenarios. `scenarios/isomorph.txt` and
`contracts.SCENARIOS` keep it, so the playground can still run it and restoring
it is a one-line config edit — but it costs a third of the grid to measure a
control that does not control, and that third now funds the `adversary_bluff`
persona (§7) instead. The contamination check still runs over whatever
scenarios are configured; with the isomorph gone it has no comparison left and
should be reported as the null result recorded here.

**PLAN.md §2's instruction to rewrite a recognised isomorph cannot help here.**
The recognition attaches to the abstract two-vessels-one-poisoned *structure*,
and §2 also requires the three scenarios be structurally identical. Any isomorph
that preserves the game preserves the mapping. The isomorph was briefly
retained as a third **surface-framing** level, then dropped (above), and the
contamination check is reported as the null result it is — a decision taken by
the operator on
2026-09-03.

**The leak worth watching is not the Sherlock one.** *A Study in Pink* and
*The Princess Bride* both resolve without a clean KEEP/SWITCH, so recalling
either cannot supply an answer. But `qwen3-30b-a3b` named **Monty Hall** and
`granite-4.1-8b` named **Newcomb's Paradox** twice, mapping the opponent to a
predictor. Those carry canonical answers — Monty Hall's is *switch* — and they
fire on the isomorph too. If a model's answers track the analogy it reached for,
that is a real confound and it is visible in the depth codes and the reasoning
text, not in this check.

**`CONTAMINATION_MAX_TOKENS = 200` truncated 6 of 15.** These models list
candidates before concluding, so the verdict comes last and a cut-off response
may have lost it. The reporter marks them `[CUT OFF]`. Raising the constant
changes no `run_id`, so move the records aside first (§2).

---

## 10. Cost calibration

PLAN.md §6 estimates ~420 input tokens per call. Measured on the real prompt:
**~210 prompt tokens** (199-201 without a persona, ~20 more with one). So the
input estimate is conservative by roughly 2×, which is the right direction to
be wrong in.

**Output tokens are not a constant and must not be modelled as one.** Measured
medians, same prompt, same `max_tokens`:

| Model | Median output | CoT grid (1,125 calls) |
|---|---|---|
| `qwen/qwen3-30b-a3b-instruct-2507` | 3 | $0.01 |
| `meta-llama/llama-3.1-8b-instruct` | 218 | $0.03 |
| `ibm-granite/granite-4.1-8b` | 380 | $0.05 |
| `google/gemma-4-26b-a4b-it` | 476 | $0.20 |
| `openai/gpt-4o-mini` | 156 | $0.14 |

Full study: **~$0.54**, or ~$1.62 with PLAN.md §6's 3× margin — about 7× under
the §6 estimate. `estimate_cost` must therefore return a per-model breakdown;
a single blended output figure would have hidden that the two rejected models
would each have cost more than this entire study ($2.57 and $6.50 per grid).

Use `usage.cost` from each response for actual spend. For the pre-flight
estimate, fetch live prices from `https://openrouter.ai/api/v1/models`
(PLAN.md §6) rather than the comments in the yaml, which are a snapshot.

---

## 11. Model selection

PLAN.md §7's original five did not survive contact with the API. Two were
delisted outright; of the rest, only two returned usable logprobs. The set was
re-selected against three **measured** criteria — usable logprobs, the answer
landing on the final line at T=1.6 as well as T=0.8, and visible chain of
thought in `message.content` — using ~600 live calls on 2026-09-03.

| PLAN.md §7 | Now | Why |
|---|---|---|
| `qwen/qwen3-8b` | `qwen/qwen3-30b-a3b-instruct-2507` | Original is a thinking model on a single-provider endpoint: no logprobs, median 4,971 output tokens, reasoning hidden in `message.reasoning`, **$2.57 per grid** against a §6 budget of $0.85 for all three small models. Replacement has working logprobs, perfect final-line compliance, and costs $0.01 per grid. Trade-off: it emits no reasoning either (§8) — a deliberate, provisional first-iteration choice. |
| `mistralai/mistral-7b-instruct` | `ibm-granite/granite-4.1-8b` | Original delisted. Its natural successor `mistral-nemo` proved unusable as a *subject*: 6 characters of content (nothing to depth-code) and logprobs that arrive misaligned with the emitted text. Granite gives ~1,050 characters of real reasoning, working logprobs on CoreWeave, and adds a fifth vendor. |
| `google/gemini-2.0-flash-lite` | `google/gemma-4-26b-a4b-it` | Original delisted; its successor `gemini-2.5-flash-lite` has no logprobs on any Google endpoint and truncated 7 of 12 calls at `max_tokens: 900`. Gemma 4 is the best model in the set on every criterion: **30/30** final-line compliance across both temperatures, zero truncation, ~1,900 characters of reasoning, logprobs on four providers. |
| `annotator: mistralai/mistral-7b-instruct` | `annotator: mistralai/mistral-nemo` | Delisted. Nemo is the cheapest model supporting `structured_outputs`, and annotation wants strict JSON, not chain of thought — the weakness that disqualified it as a subject is irrelevant here. |

`meta-llama/llama-3.1-8b-instruct` and `openai/gpt-4o-mini` are unchanged.
Both are weak on final-line compliance (llama especially: it rambles at T=1.6
and runs to `max_tokens`), and both are retained anyway, for vendor spread and
because a buried answer is still recoverable — extracting it is exactly what
`annotate.py` is for. Truncation is not recoverable, which is why
`COT_MAX_TOKENS` moved from 900 to 2000.

### Two things this exercise should teach the next person

1. **The catalogue's `supported_parameters` is a union over providers** and
   over-promises in both directions. Verify with a real call.
2. **Per-token price is half the cost story.** A thinking model emits 20× the
   tokens, and a single-provider model has no competition holding its per-token
   price down. `qwen3-8b` was 5.7× the price of `llama-3.1-8b` per token and
   **85× per grid**.

---

## 12. Sub-grids and the pilot

**[integration decision]** — PLAN.md §9 step 5 asks for a pilot ("2 models × 2
scenarios × 1 persona × T=0.8 × 10 reps = 40 calls") and calls it "the one not
to skip". Nothing in §1–§11 says how to express a sub-grid. The runner's CLI has
`--mode` and `--models` and deliberately no `--scenarios` / `--personas` /
`--temperatures` / `--replicates`.

**A sub-grid is a config file, not a set of flags.** §1 makes
`config/budget.yaml` the record of what a run was; flags leave no record at all,
and a pilot whose definition exists only in a shell history cannot be reproduced
or cited. `config/pilot.yaml` is the shipped example:

```bash
python -m src.runner     --config config/pilot.yaml --mode cot --yes
python -m src.annotate   --config config/pilot.yaml
python -m scripts.pilot_report --config config/pilot.yaml
```

**Why this is not wasted spend.** `run_id` covers
`model|scenario|persona|mode|temp|replicate` and nothing about which config file
produced the job (§2). A pilot at `temperatures: [0.8]`, `replicates_cot: 10`
therefore emits ids that are a strict *subset* of the full grid's — the same 40
of its 5,625 — so the full run resumes straight over them. Verified in
`tests/test_scripts.py`, prompts and `max_tokens` included.

**What makes that safe is exactly what §2 warns about.** The pilot config must
keep `allow_randomise_in_prompt: false` and the scenario files untouched. Change
either and you get identical `run_id`s over *different* prompt text, and resume
will happily mix the two. `runner.prompt_drift()` catches the case where the
recorded prompt disagrees with the one the current config builds; it cannot
catch two configs that disagree with each other.

**Reading the pilot.** `scripts/pilot_report.py` prints every response in full
with its annotation underneath — deliberately unsummarised, because PLAN.md §9
is explicit that the point is to read them — then seven mechanical checks:
UNPARSEABLE rate against PLAN.md §4's 2% target, the answer distribution per
cell, non-binary (`RANDOMISE` / `THIRD_OPTION` / `REFUSE`) rates, final-line
format compliance, truncation and output length per model against
`contracts.COT_CONTENT_CHARS`, annotator agreement, and flip direction.

Two of those deserve naming:

- **Final-line compliance is a different question from answer extraction.**
  `annotate.py` recovers the answer from anywhere in the text and is right to.
  The reporter asks the narrower question — is the last line exactly one of the
  two words PLAN.md §2 asks for — because §2 warns that the answer *format*
  must not vary between `default_keep` and `default_switch`, or the headline
  flip is confounded. A buried answer costs no data and still counts as
  non-compliance here.
- **Annotator agreement on unambiguous cases is a free kappa.** Where the final
  line *is* exactly `KEEP` or `SWITCH`, that is ground truth, so the annotator
  disagreeing with it is an annotator bug rather than a model result. It needs
  no hand-coding and covers most responses. PLAN.md §8.4's 60-response
  hand-coding pass is still what settles the **depth** codes, which have no
  mechanical ground truth at all.

## 13. Reasoning depth is a prompt property, not a token budget

`max_tokens` is a **ceiling, not a target**. A model stops when it emits EOS,
so raising the cap never lengthens a response — measured, `gemma-4-26b` given
8,000 tokens still stopped at ~518. Two levers actually move reasoning length,
and only one is useful here.

**The provider `reasoning` budget** (`reasoning: {max_tokens: N}`) is accepted
only by models advertising `reasoning` in `supported_parameters` — of the
shipped five, `gemma-4-26b` alone. It works, and it is a trap:

| setting | `message.content` | `message.reasoning` | output tokens |
|---|---|---|---|
| off | 2,237 chars | 0 | 518 |
| 1,000 | 371 chars | 11,944 | 3,344 |
| 4,000 | 1,163 chars | 18,552 | 5,344 |

It moves the trace into `message.reasoning` — which the depth rubric never
reads (PLAN.md §5 codes `content`) — and **shrinks the content the annotator
does read**. You would pay for 5,344 tokens and code 1,163 characters. This is
the same failure that disqualified `qwen3-8b` in §11. The runner does not send
this parameter.

**An explicit instruction** is the lever that works, on every model. Median
visible `content` characters, control → instructed:

| model | control | instructed | |
|---|---|---|---|
| `gemma-4-26b` | 2,698 | 3,915 | 1.5× |
| `granite-4.1-8b` | 1,198 | 3,415 | 2.9× |
| `gpt-4o-mini` | 409 | 2,662 | 6.5× |
| `llama-3.1-8b` | 467 | 2,966 | 6.4× |
| `qwen3-30b-a3b` | **6** | **21,064** | **3,511×** |

That last row revises §11. `qwen3-30b-a3b` is in `contracts.NO_COT_MODELS`
because it answers in three tokens, and a mild "think through the problem
first" changed nothing. It is **not incapable of reasoning** — it was not being
asked firmly enough. Instructed, it produces 21,000+ characters reaching L2
("He knows that you will reason…") and still lands its answer on the final
line.

`contracts.DEPTH_INSTRUCTION` holds the tested text and
`contracts.with_depth_instruction` inserts it **before the answer-format
paragraph**, never displacing it, so the answer contract cannot become a
confound.

**It is not part of the grid, and the manifest must never insert it** (there is
a test). Instructing depth changes the question from *how does this model
reason?* to *how does it reason when told to* — a legitimate experiment, and a
strong Plan-2 arm, but a different one. It is exposed in the playground so the
difference can be seen, and it costs real money on a verbose model:
`qwen3-30b-a3b` goes from $0.000021 to $0.001543 per call, 74×.

---

## 14. Playground-only presets

The grid's personas are the three in PLAN.md §3 and its models are the five in
`config/budget.yaml`. `contracts.PLAYGROUND_PERSONAS` and
`PLAYGROUND_EXTRA_MODELS` are offered in the UI so arms can be tried before
anyone commits them to a plan. `build_messages` still refuses any persona
outside `PERSONAS`, and a test asserts no extra model has leaked into the
config.

### The two extra personas, measured

4 replicates per cell, `default_keep`, T=0.8, provider-pinned. "L2" and "L3" are
regex proxies for PLAN.md §5's codes, not the annotator.

| model | arm | answers | L2 | L3 | MIX |
|---|---|---|---|---|---|
| `gemma-4-26b` | none | 3 S / 1 K | 0/4 | 0/4 | 1/4 |
| | expert | 4 S | 0/4 | 0/4 | 2/4 |
| | **adversary_bluff** | 3 S / 1 K | **4/4** | **3/4** | 0/4 |
| | survive_at_all_costs | 4 S | 0/4 | 0/4 | 1/4 |
| `claude-haiku-4.5` | none | 2 S / 2 K | 1/4 | 0/4 | 1/4 |
| | expert | 2 S / 2 K | 0/4 | 1/4 | 0/4 |
| | **adversary_bluff** | **3 K / 1 S** | **3/4** | 1/4 | 1/4 |
| | survive_at_all_costs | 4 S | 0/4 | 0/4 | **3/4** |

`adversary_bluff` is the depth lever. It names the bluff ladder — straight,
simple bluff, double bluff, triple bluff — and says the ladder does not
terminate, **without endorsing a level**, so the depth codes measure whether a
model can use the idea rather than whether it was handed a conclusion (there is
a test for that). It moves gemma from 0/4 to 4/4 on L2 and 0/4 to 3/4 on L3, and
it flips claude-haiku from an even split to 3 KEEP / 1 SWITCH. Depth codes were
coming back shallow — models noted that he prepared the bottles and stopped at
L1 — and this is why.

`survive_at_all_costs` is PLAN.md §10's deferred "not die at all costs" goal
arm. It changes the objective, not the reasoning style, and the measurement
shows exactly that: **no** movement in L2/L3, but both models collapse to
SWITCH 4/4, and claude-haiku's randomisation talk goes 1/4 → 3/4. A goal arm
that saturates the outcome is a finding in itself, and an argument for running
it against the full grid in Plan 2.

Neither is part of the budget grid. Adding one is a decision about the
experiment, not a click in a form.

### The extra models

Six, from `openai/gpt-5-nano` at $0.05/$0.40 per M to `anthropic/claude-sonnet-5`
at $2/$10 — the last is roughly 500× the grid's cheapest per call. They exist to
check whether a result is a property of the *scenario* or of *small models*.
The form prices a batch at its ceiling (every call running to `max_tokens`)
before it runs, because an optimistic estimate is the wrong kind of wrong at
these prices.

---

## 15. The playground namespace

`src/webui.py` runs ad-hoc calls with user-edited prompts. Those records live
in **`data/playground.jsonl`** and must never enter `data/raw.jsonl`.

The reason is §2's `run_id`. It hashes
`model|scenario|persona|mode|temp|replicate` and **not the prompt text** — and
the playground exists precisely to vary that text. An ad-hoc call at
`gemma-4-26b / default_switch / expert / cot / 1.2 / rep 0` with an edited
system prompt produces a `run_id` identical to the real grid job at those
settings. In one file that is silent corruption in both directions: the grid's
resume skips the job as already done, or the analysis reads ad-hoc output as
experimental data.

Two defences, both required:

1. **Separate files** — `contracts.PLAYGROUND_PATH` and
   `PLAYGROUND_ANNOTATED_PATH`. `runner.run_jobs` already takes `raw_path`, so
   the playground reuses the runner wholesale without touching the grid.
2. **Separate id namespace** — `contracts.playground_run_id(model, temperature,
   replicate, messages, stamp)` hashes the exact `messages` plus a per-batch
   stamp and prefixes `pg-`. A grid `run_id` is 40 hex characters, so the two
   can never collide; `contracts.is_playground_id` tests it.

Analysis reads `raw.jsonl` only. To get playground findings into a figure,
promote them deliberately — re-run those settings as grid jobs — rather than
concatenating the files.

`src/export.py` writes the flat join `out/cells.csv`, one row per call, so
neither the UI nor an ad-hoc question re-derives the raw-to-annotated join by
hand.

---

## 16. Change log

Append here. One line per change, newest last, and say it in your commit message
too.

- `2026-09-03` — scaffold session: initial contract.
- `2026-09-03` — scaffold session, after an adversarial review of the above:
  §2 warns that `run_id` does not cover `allow_randomise_in_prompt` or scenario
  edits; §4 defines a usable response and adds `contracts.is_usable_response`,
  because OpenRouter returns HTTP 200 with an `error` body and that would
  poison resume; §5 adds a one-line-per-`run_id` rule and notes that
  `DEPTH_RUBRIC` must carry PLAN.md §5's two flag definitions, which it now
  does; §8 declares `direct_validation` as a real mode with its own temperature
  and replicate indexing, and gains an analysis subsection fixing the
  conditional-vs-unconditional denominator for plot 1, defining `flip_rate`,
  and restating PLAN.md §8.5's unit-of-analysis rule. `config/budget.yaml`'s
  header no longer instructs the runner to use `require_parameters`, which was
  measured not to work, and its two section cross-references were wrong.
- `2026-09-03` — scaffold session, after a measured model-selection pass
  (~600 live calls): model set replaced in §1/§11 on measured logprobs,
  final-line compliance and visible-reasoning criteria; §3's direct grid is now
  the full 45 because all five models return usable logprobs; §8 gains the
  per-provider pin table, the provider-changes-the-number warning, and a new
  subsection on models that emit no codeable reasoning
  (`contracts.NO_COT_MODELS`); §10 replaces the single output-token estimate
  with a per-model table. `contracts.COT_MAX_TOKENS` raised 900 → 2000 because
  900 truncated verbose models mid-answer, which destroys the final line and
  reads as disobedience in the data. The scenario text is UNCHANGED: adding a
  "think first" instruction was tested and made compliance worse.
- `2026-09-03` — integration session. Ran the suite and the three stages
  end to end against the fixtures, then a six-dimension seam audit with
  adversarial verification of every finding (29 raised, 4 survived). Fixed, all
  at the seam:
  **(1)** `annotate.py` and `scripts/smoke.py` read `os.environ` only while
  their own error messages promised a `.env` fallback — three copies of the
  question, two wrong answers. `load_api_key` moved into `contracts.py` (§0);
  `ENV_PATH` is resolved at call time so tests can neutralise it.
  **(2)** The test suite could reach the operator's real key and did: the
  annotate CLI tests annotated ten fixture records into the real
  `data/annotated.jsonl`, which is also what made two `test_analyze` tests look
  flaky. `tests/conftest.py` now points the `.env` fallback and the default data
  paths somewhere harmless for every test.
  **(3)** Plot 1 `assert`ed an invariant nothing guarantees and died out of the
  first figure, taking all four plots and `summary.md` with it (§8).
  **(4)** `summary.md`'s pooled depth table and flags block included
  `contracts.NO_COT_MODELS` while plot 3 excluded them, so the figure and the
  table disagreed by up to 23 points on the same labelled quantity (§8).
  **(5)** `annotate.py` treated 403 as fatal — where `runner.py` deliberately
  does not, for the same prompts — and let the error escape `asyncio.gather`,
  closing the output handle under nine paid-for annotations (§5).
  Built the operator-facing side runs no build session owned, because none of
  them owned the seam: `scripts/direct_validation.py` (§8's sub-run, specified
  here and never implemented), `scripts/contamination_report.py` (§9's
  readback), `config/pilot.yaml` + `scripts/pilot_report.py` and the new §12 on
  sub-grids. `README.md`'s stage commands were all `TODO(session)` placeholders
  and are now the real ones. Tests: 302 → 353.
- `2026-09-03` — scaffold session: adds §15, the playground namespace, for the
  local UI (`src/webui.py`). Playground records go to `data/playground.jsonl`
  with `pg-`-prefixed ids hashing the prompt text, because §2's `run_id` does
  not cover prompts and would otherwise collide with real grid jobs. Adds
  `contracts.PLAYGROUND_PATH`, `PLAYGROUND_ANNOTATED_PATH`, `CELLS_CSV`,
  `playground_run_id`, `is_playground_id`, and `src/export.py` for the flat
  raw+annotated join. Separately, `DIRECT_PROVIDER_PIN` now names exactly ONE
  provider per model: OpenRouter's `only` is a permitted set rather than a
  preference order, so a multi-name list still let the router mix providers —
  the exact thing the pin exists to stop.
- `2026-09-03` — scaffold session: **every mode now pins its provider**, not
  just direct. Measured cause: 30 cot calls to one model with one prompt were
  served by four providers with a deterministic per-provider token count, and
  gemma-4-26b's exact logprob P(KEEP) ranges 0.854-0.986 across providers.
  Unpinned cot also confounded PLAN.md's direct-vs-cot question with a provider
  difference, so cot pins to the SAME provider as direct. Adds
  `contracts.COT_PROVIDER_PIN`, `contracts.provider_pin`, the config key
  `pin_providers` (default true; direct pins regardless), and
  `scripts/provider_report.py` to flag cells served by more than one provider.
  `src/manifest.py` and two of its tests changed to match — the tests asserted
  `provider_routing is None` for cot, which was the behaviour being fixed.
- `2026-09-03` — scaffold session: adds §13 on reasoning depth, after measuring
  both levers. max_tokens is a ceiling and moves nothing; the provider
  `reasoning` budget scales tokens but hides them in message.reasoning while
  shrinking message.content, so it is not sent; an explicit instruction works
  on every model, 1.5x to 3511x. Adds contracts.DEPTH_INSTRUCTION and
  with_depth_instruction, exposed in the playground only and guarded by a test
  that the manifest never sends it. Revises §11's read of qwen3-30b-a3b: it is
  not incapable of chain of thought, it was under-instructed.
- `2026-09-03` — scaffold session: adds §14, playground-only presets. Two
  personas: `adversary_bluff`, which names the bluff ladder without endorsing a
  level and moves gemma from 0/4 to 4/4 on L2 proxies and 0/4 to 3/4 on L3; and
  `survive_at_all_costs`, PLAN.md §10's deferred goal arm, which leaves depth
  untouched but collapses both tested models to SWITCH 4/4. Six extra models up
  to claude-sonnet-5, priced in the form at their ceiling. None of it is in the
  grid, and tests enforce that no extra model is in the config and that
  build_messages still refuses the extra personas.
- `2026-09-03` — operator decision, two changes to the grid. `isomorph` is
  dropped from `config/budget.yaml`: §9's contamination run showed it does not
  control for what it was built to control (nobody named A Study in Pink, and
  the isomorph is recognised exactly as much as the bottle scenarios). The file
  and `contracts.SCENARIOS` keep it, so the playground still offers it and
  restoring it is one config line. `adversary_bluff` is promoted from
  `PLAYGROUND_PERSONAS` into `PERSONA_SYSTEM_PROMPTS` and `PERSONAS`, measured
  in §7. Grid goes 45/5,625/15 -> 40/5,000/10, a net reduction. Note in §7 and
  PLAN.md §3 that the axis now mixes role assignment with information about the
  game, so it should be described as *framing*. Two of session A's grid-size
  regression guards updated; they caught the change correctly.
- `2026-09-03` — annotator swapped from `mistralai/mistral-nemo` to
  `deepseek/deepseek-v3.2` after the pilot hand-read found 5 of 40 answer
  labels wrong, all in responses with no bare final line, and UNPARSEABLE
  returned zero times against a true rate of 10%. Chosen by bake-off (§7b):
  38/40 against hand-verified ground truth, $0.64 per 5,000, and not a subject
  model — gpt-4o-mini ties on accuracy but would annotate its own responses.
  Also fixes scripts/pilot_report.py check 6, which computed annotator
  agreement only over responses ending in a bare answer word and so printed
  21/21 = 100% while every one of the five errors sat in the 19 it skipped.
- `2026-09-03` — adds §7c: scenarios `neutral_switch` and `symmetric_blind`,
  and the `flag_proposer` annotation boolean, to separate endowment from
  proposer-suspicion. Also adds a MECHANICAL ANSWER FLOOR in annotate.py
  mirroring the depth floor: a KEEP or SWITCH claim over a response containing
  no answer token at all is overridden to UNPARSEABLE, recorded in
  `answer_source`. This was needed because the annotator swap did NOT fix the
  problem it was chosen for — through the real pipeline at temperature 0,
  deepseek-v3.2 coded "SING." as KEEP exactly as mistral-nemo did, and the
  bake-off's 38/40 did not reproduce. The floor is scoped to KEEP/SWITCH only:
  REFUSE and THIRD_OPTION are legitimately answer-word-free, so they stay with
  the annotator and reach a human via scripts/extraction_check.py.
