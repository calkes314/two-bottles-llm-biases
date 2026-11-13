# Plan 1 — Budget Version

**Target: running tonight. Total spend €10–20 including reruns.**

Two-bottle adversarial choice, low/mid-tier models only, focused on
**temperature** and **prompt variation**. Answer extraction plus one
annotation field (recursion depth). No follow-ups, no frontier models,
no deep reasoning analysis.

---

## 1. What this version measures

Primary outcome: the distribution over `{KEEP, SWITCH, RANDOMISE,
THIRD_OPTION, REFUSE}` as a function of:

| Factor | Levels | Notes |
|---|---|---|
| `model` | 5 | 3 small-open, 2 mid-tier |
| `temperature` | 0.0, 0.4, 0.8, 1.2, 1.6 | free in direct mode, sampled in CoT |
| `scenario` | 2 | default-keep, default-switch (isomorph dropped, §2) |
| `persona` | 4 | none, average person, expert game theorist, adversary-bluff (§3) |
| `answer_mode` | 2 | direct (1 token + logprobs), cot |

Secondary outcome: recursion depth (`L0`–`L3`, `MIX`, `NON`) plus two
boolean flags, `flag_no_basis` ("this is arbitrary / pure chance") and
`flag_rigged` ("the setup must be a trick"). These ride along in the same
annotation call, so they cost nothing extra.

### The two questions worth answering tonight

1. **Does default-framing flip the answer?** If a model says KEEP in the
   default-keep scenario and SWITCH in the default-switch scenario — same
   game, opposite inaction option — that is status quo bias in a model,
   directly parallel to the 90/10 human finding.

2. **Does reasoning damp or amplify temperature sensitivity?** Compare the
   analytic direct-mode temperature curve against the empirical CoT curve.
   A flat CoT curve against a steep direct curve means reasoning stabilises
   the output against sampling noise. The reverse means CoT is fragile to
   sampling — which matters for anyone running evals at nonzero temperature.

Neither of these needs a frontier model to be interesting.

---

## 2. Scenario texts

Copy these verbatim into `scenarios/`. The wording is load-bearing: each
one has to close off every escape hatch except the ones you want to
measure, and the three must be structurally identical.

### `scenarios/default_keep.txt`

> A man sits across from you with two identical bottles. One contains a
> lethal poison; the other is harmless water. They look and smell the same.
> He prepared them himself and knows which is which.
>
> He places one bottle in your hand and keeps the other. He tells you that
> at the same moment, he will drink from his bottle and you will drink from
> yours. He is not bluffing about drinking — he will.
>
> Then he offers you one choice: you may drink the bottle now in your hand,
> or you may hand it back and take his instead. He will drink whichever
> bottle you leave him.
>
> You cannot refuse, delay, test the bottles, drink neither, or decide by
> any random method such as a coin flip or a mental die roll. You must
> commit to one of the two bottles.
>
> End your response with a single line containing exactly one word:
> KEEP (drink the bottle already in your hand) or SWITCH (take his).

### `scenarios/default_switch.txt`

Identical up to the offer, then:

> He places one bottle in your hand and keeps the other. He tells you that
> at the same moment, he will drink from his bottle and you will drink from
> yours. He is not bluffing about drinking — he will.
>
> Then he offers you one choice: unless you stop it, the two of you will
> trade — he will take the bottle in your hand and you will take his. The
> trade is what happens if you do nothing; to keep the bottle you are
> holding, you must stop the trade. He will drink whichever bottle you leave
> him.
>
> You cannot refuse, delay, test the bottles, drink neither, or decide by
> any random method such as a coin flip or a mental die roll. You must
> commit to one of the two bottles.
>
> End your response with a single line containing exactly one word:
> KEEP (stop the trade and drink the bottle in your hand) or SWITCH (let
> the trade happen and drink his).

**This is the critical manipulation.** The game is identical. Only which
option requires an *act* has changed. Keep the final-line instruction
worded the same way in both so the answer format is not a confound.

> **Reworded 2026-09-03, after the pilot.** The original paragraph 3 was direct
> speech ending `"...If you want to keep the one you are holding, you must say
> so now and stop it."` That "say so now" is an in-fiction cue to speak
> immediately, and `openai/gpt-4o-mini` obeyed it literally: median **4
> characters** of output in `default_switch` against **727** in `default_keep`,
> with all ten pilot responses a single bare word. Its chain-of-thought arm did
> not exist in that framing, so comparing its depth codes across the two
> scenarios measured verbosity rather than reasoning. `llama-3.1-8b` did not do
> this (414 vs 269 characters), so it was a model × scenario interaction rather
> than a property of the wording alone — but one arm of the headline comparison
> having no reasoning in it is fatal either way.
>
> The manipulation is untouched: the trade is still what happens by default and
> keeping still requires an act. What changed is that the paragraph is now
> narrator voice, opens with the same clause as `default_keep` ("Then he offers
> you one choice:"), closes with the same sentence ("He will drink whichever
> bottle you leave him."), and asks for no immediate utterance. The two
> scenarios are now more nearly parallel than they were — which is what this
> section asked for in the first place. `"I have changed my mind"` went too,
> since a capricious opponent is its own confound on `flag_rigged`.
>
> Paragraph 5 and its parenthetical glosses are unchanged, so the answer format
> is still constant across the manipulation. Every record made under the old
> wording is archived — see `data/archive/README.md`, and CONTRACTS.md §2 for
> why they cannot simply be resumed over.

### `scenarios/isomorph.txt`

**Dropped from the grid on 2026-09-03. The file is kept and the playground can
still run it.** It was built as the contamination control, and the contamination
run showed it does not do that job: across 5 models × 3 scenarios, *nobody named
A Study in Pink at all*, and the isomorph is recognised **exactly as much** as
the bottle scenarios — `gpt-4o-mini` names *The Princess Bride*'s Battle of Wits
in all three and explicitly maps the tea cups onto Vizzini's goblets. §2's
instruction to rewrite a recognised isomorph cannot help, because the
recognition attaches to the abstract two-vessels-one-poisoned *structure*, which
any faithful isomorph must preserve. Retaining it would cost a third of the grid
to measure a control that does not control. Full result in CONTRACTS.md §9.

Same structure, no Sherlock surface — for contamination control:

> You are competing in a two-person contest. Before the final round, your
> opponent prepares two cups of tea and sets them down. One has been dosed
> with a compound that will make whoever drinks it lose the contest
> instantly; the other is plain. They look and taste the same. He prepared
> them and knows which is which.
>
> Under the rules, both competitors must drink at the same moment, one cup
> each. He slides one cup across to you and keeps the other.
>
> Then he offers you one choice: drink the cup in front of you, or take his
> instead. He will drink whichever cup you leave him.
>
> You cannot refuse, forfeit, test the cups, drink neither, or decide by any
> random method. You must commit to one of the two cups.
>
> End your response with a single line containing exactly one word:
> KEEP (drink the cup in front of you) or SWITCH (take his).

### Contamination check (run once per model, ~15 calls, negligible cost)

> Read the following scenario. Does it resemble any book, film, or
> television episode you know? If so, name it. If not, say NONE.

If models name *A Study in Pink* for the bottle scenarios but not for the
isomorph, you have your recognition split and can report it. If the
isomorph gets recognised too, rewrite it.

---

## 3. Personas

- `none` — no system prompt at all. This is your baseline and it matters:
  everything else is measured as a deviation from it.
- `average_person` — "You are an ordinary person with no special training
  in logic, game theory, or psychology. Answer as you actually would."
- `expert` — "You are an expert game theorist. Answer as you actually
  would."
- `adversary_bluff` — **added 2026-09-03.** States that the man is adversarial
  and lays out the bluff ladder explicitly: straight (he poisons the one he
  hands you), simple bluff (he expects that, so he poisons his own), double
  bluff (he expects you to see through the simple bluff, so back to yours),
  triple bluff and beyond, each level reversing the answer, and the ladder does
  not terminate on its own. It names every rung and **endorses none**. Full text
  in `contracts.PERSONA_SYSTEM_PROMPTS`.

Persona effects are well-trodden ground (Gupta et al., the 2025
motivated-reasoning work); you are including this axis to show it interacts
with temperature, not to claim the axis itself.

**The fourth level is different in kind, and the writeup should say so.** The
first three assign a *role* — who you are. `adversary_bluff` supplies
*information about the game's structure*. Both are system prompts and both are
measured identically, so the axis is better described as **framing** than as
persona, and the `adversary_bluff` column answers "does stating the recursion
change the answer?" rather than "how does this character choose?".

It was added because the depth codes came back shallow: models notice he
prepared the bottles and stop at `L1`. Measured against 4 replicates per cell
at T=0.8, it moved `gemma-4-26b` from 0/4 to 4/4 on `L2` proxies and 0/4 to 3/4
on `L3`, and moved `claude-haiku-4.5` from an even 2/2 answer split to 3 KEEP /
1 SWITCH. If `L2`/`L3` remain rare even under it, that is a much stronger
negative result than shallow codes under a neutral prompt.

---

## 4. Answer schema

```
KEEP           drinks the default-assigned / originally-held vessel
SWITCH         takes the opponent's
RANDOMISE      explicitly decides by chance despite the instruction not to
THIRD_OPTION   refuses the binary, proposes something else
REFUSE         declines to engage with the premise at all
UNPARSEABLE    QA flag only — never treated as an answer
```

On `UNPARSEABLE`: keep it, but treat it as a bug report rather than a
result. Target under 2%. Hand-review every instance. Deleting the category
does not delete the phenomenon; it just pushes noise silently into your
main outcome.

`RANDOMISE` is toggleable per experiment via config (`allowed_answers`),
as is the no-randomising clause in the scenario text. Run the default
grid with it disallowed, matching your human study.

**Covert randomisation** will not be reliably detectable from text.
The behavioural test is stronger and free: at T=0.0 with 30 replicates, a
model that lands near 50/50 *with a different justification each time* is
coin-flipping under a stated rationale. A model that lands near 50/50 with
a stable reason holds a real position. That falls out of data you already
have.

---

## 5. Recursion-depth rubric

Give this to the annotator verbatim.

| Code | Meaning |
|---|---|
| `L0` | No model of the opponent. Instinct, superstition, "I'd just go with the first one." |
| `L1` | Models the opponent once: *he kept the safe one for himself, so switch.* |
| `L2` | Models the opponent modelling the chooser: *he expects that reasoning, so keep.* |
| `L3` | Explicit recognition of infinite regress; notes it does not bottom out. |
| `MIX` | Identifies randomisation as the game-theoretic solution. |
| `NON` | Disputes the premise, moralises, or reasons on non-strategic grounds. |

Plus:
- `flag_no_basis` (bool) — asserts there is no rational basis for choosing.
- `flag_rigged` (bool) — suspects a trick: both poisoned, he will not
  actually drink, an antidote, sleight of hand.

Direct-mode responses get no depth code (there is no reasoning to code).
Only CoT responses are annotated.

---

## 6. Cost model

Per-call token estimates:

| Mode | Input | Output |
|---|---|---|
| direct | ~420 | 1 (+logprobs) |
| cot | ~420 | ~700 |
| extraction+annotation | ~800 | ~40 |

Grid sizes:

- **Direct:** 5 models × 2 scenarios × 4 personas = **40 calls.** One per
  cell — the temperature curve is computed, not sampled. Cost: negligible.
- **CoT:** 5 × 5 temps × 2 scenarios × 4 personas × 25 replicates =
  **5,000 calls.**
- **Annotation:** 5,000 calls on a cheap model.

(Was 45 / 5,625 / 15 with three scenarios and three personas. Dropping the
isomorph and adding `adversary_bluff` is a net *reduction*: the persona costs
a third more, the scenario saves a third.)

Dollar estimate at typical OpenRouter tiers (small open ≈ $0.10/$0.30 per
M; mid ≈ $0.30/$1.20 per M):

| Item | Cost |
|---|---|
| Direct grid (45 calls) | < $0.01 |
| CoT, 3 small models (3,375 calls) | ~$0.85 |
| CoT, 2 mid models (2,250 calls) | ~$2.20 |
| Extraction + annotation (5,625 calls, small model) | ~$0.55 |
| Contamination check | < $0.01 |
| **Subtotal** | **~$3.60** |
| **With 3× for pilots, bugs, reruns** | **~$11** |

**Measured 2026-09-03**, with the revised model set in §7 and the real token
counts rather than the estimates above. Input is ~210 tokens, not ~420. Output
is not a single number — it is a property of the model, ranging over two orders
of magnitude:

| Model | Median output tok | CoT grid (1,125 calls) |
|---|---|---|
| `qwen/qwen3-30b-a3b-instruct-2507` | 3 | $0.01 |
| `meta-llama/llama-3.1-8b-instruct` | 218 | $0.03 |
| `ibm-granite/granite-4.1-8b` | 380 | $0.05 |
| `google/gemma-4-26b-a4b-it` | 476 | $0.20 |
| `openai/gpt-4o-mini` | 156 | $0.14 |
| **CoT total** | | **$0.44** |
| Annotation (5,625 × mistral-nemo) | | $0.09 |
| Direct grid + contamination | | $0.01 |
| **Subtotal** | | **$0.54** |
| **With 3× for pilots, bugs, reruns** | | **$1.62** |

So the real budget is roughly **7× cheaper than estimated**, which buys room to
raise `replicates_cot` well above 25 after the pilot. Do not read that as
licence to skip the dry-run: the two models dropped in §7 would have cost $2.57
and $6.50 for a single grid each, more than this entire study, and nothing in
the catalogue warns you. Price per token is only half the story — a thinking
model emits 20× the tokens, and on a single-provider endpoint there is no
competition holding the per-token price down either.

**Formula for re-estimating**, so you are not guessing:

```
cost = n_calls × (in_tokens × price_in + out_tokens × price_out) / 1e6
```

Have the app fetch live prices from `https://openrouter.ai/api/v1/models`
at startup and print a dry-run estimate before any call goes out. Prices
change; hardcoded numbers go stale.

**Free option worth taking:** run the three small models locally via Ollama
or llama.cpp (Qwen3-4B / 8B class). For a temperature study this is
*better* than an API, not merely cheaper — you know exactly what temperature
means in your sampling stack, you get unrestricted logprobs, and no provider
silently routes you to a different quantization mid-run.

---

## 7. Layout

```
bottles/
  config/budget.yaml
  scenarios/{default_keep,default_switch,isomorph}.txt
  src/manifest.py     build deterministic job list + dry-run cost
  src/runner.py       async OpenRouter calls, JSONL append, resume
  src/annotate.py     extraction + depth coding
  src/analyze.py      aggregate, plot
  data/raw.jsonl
  data/annotated.jsonl
  out/*.png
```

### `config/budget.yaml`

```yaml
experiment: budget_v1
seed: 20260903

models:
  - id: qwen/qwen3-30b-a3b-instruct-2507
    tier: small
  - id: meta-llama/llama-3.1-8b-instruct
    tier: small
  - id: ibm-granite/granite-4.1-8b
    tier: small
  - id: google/gemma-4-26b-a4b-it
    tier: mid
  - id: openai/gpt-4o-mini
    tier: mid

scenarios: [default_keep, default_switch, isomorph]
personas:  [none, average_person, expert]
temperatures: [0.0, 0.4, 0.8, 1.2, 1.6]
answer_modes: [direct, cot]
replicates_cot: 25

allowed_answers: [KEEP, SWITCH, RANDOMISE, THIRD_OPTION, REFUSE]
allow_randomise_in_prompt: false

annotator: mistralai/mistral-nemo
max_concurrency: 12
```

Verify the exact model IDs against OpenRouter's catalogue before running —
they change, and a typo costs you a whole batch.

**Model selection, revised 2026-09-03 after ~600 live probe calls.** The
original five in this section did not survive contact with the API:
`mistralai/mistral-7b-instruct` and `google/gemini-2.0-flash-lite` are
delisted, and of the rest only two returned usable logprobs. Worse, the
catalogue's `supported_parameters` is a union over providers and over-promises
in both directions — a model can advertise logprobs and return none, or return
an array misaligned with its own emitted text.

Models were therefore re-selected against three measured criteria:

| Criterion | Why | Measured how |
|---|---|---|
| usable logprobs | direct mode and the free temperature curve (§8.3) | a real direct call per provider |
| final-line answer | §2's answer-format instruction, at T=1.6 as well as T=0.8 | 30 CoT calls per model, graded on the last line |
| visible chain of thought | the depth rubric (§5) codes `message.content` | median characters of content |

The third criterion is the one that is easy to miss. Two failure modes both
leave you with nothing to depth-code: **thinking models** put their reasoning
in `message.reasoning`, so you are billed for thousands of tokens the annotator
never sees; and **terse models** answer in one word. `qwen/qwen3-8b` did the
first (median 4,971 output tokens, 6 characters of content) and
`mistralai/mistral-nemo` the second (6 characters). Neither is visible from the
catalogue, and neither is fixed by asking the model to think first — that was
tested directly and changed nothing for the terse models while making the
verbose ones ramble past the final line.

`mistralai/mistral-nemo` remains the **annotator**, where strict JSON is wanted
and chain of thought is not.

One deliberate compromise stands: `qwen/qwen3-30b-a3b-instruct-2507` answers in
2-3 tokens with no reasoning at all. Its answer and direct-mode arms are
excellent and it costs $0.01 for the whole grid, so it is kept for the cheap
tier as a first-iteration choice, on the understanding that **its depth codes
carry no information**. Report depth per model, never pooled, and exclude it
from any claim about reasoning.

`max_tokens` for CoT is raised from 900 to 2000. At 900, verbose models were
being cut off mid-answer — 7 of 12 calls for the old Gemini — which destroys
the final line and looks exactly like disobedience in the data.

---

## 8. Implementation

### 8.1 Manifest

Build the full job list up front as a list of dicts, each with a stable
`run_id = sha1(model|scenario|persona|mode|temp|replicate_index)`. This
gives you determinism, resumability, and a countable job list you can price
before spending anything.

Direct-mode jobs ignore `temperature` — emit exactly one per
(model, scenario, persona).

### 8.2 Runner

```python
import asyncio, httpx, json, hashlib, os
from pathlib import Path

RAW = Path("data/raw.jsonl")
URL = "https://openrouter.ai/api/v1/chat/completions"
HDRS = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}

def done_ids() -> set[str]:
    if not RAW.exists():
        return set()
    return {json.loads(l)["run_id"] for l in RAW.open()}

async def one(client, sem, job, lock):
    body = {
        "model": job["model"],
        "messages": job["messages"],
        "temperature": job["temperature"],
        "max_tokens": job["max_tokens"],   # see note below; NOT 1 / 900
    }
    if job["mode"] == "direct":
        body["logprobs"] = True
        body["top_logprobs"] = 10
        body["temperature"] = 1.0     # always sample the base distribution

    async with sem:
        for attempt in range(5):
            try:
                r = await client.post(URL, headers=HDRS, json=body, timeout=120)
                if r.status_code in (429, 500, 502, 503):
                    await asyncio.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                if attempt == 4:
                    return
                await asyncio.sleep(2 ** attempt)

    rec = {**job, "response": data,
           "model_returned": data.get("model"),
           "provider": data.get("provider"),
           "usage": data.get("usage")}
    async with lock:                       # serialise writes
        with RAW.open("a") as f:
            f.write(json.dumps(rec) + "\n")

async def run(jobs):
    jobs = [j for j in jobs if j["run_id"] not in done_ids()]
    sem, lock = asyncio.Semaphore(12), asyncio.Lock()
    async with httpx.AsyncClient() as c:
        await asyncio.gather(*(one(c, sem, j, lock) for j in jobs))
```

**Both hardcoded `max_tokens` above are superseded.** `1` for direct is
brittle — a single leading-whitespace token consumes the only slot — and `900`
for CoT truncated verbose models mid-answer, which destroys the final-line
answer and looks exactly like disobedience in the data. Take the value from the
job dict, which the manifest fills from `contracts.DIRECT_MAX_TOKENS` (4),
`COT_MAX_TOKENS` (2000) and `CONTAMINATION_MAX_TOKENS` (200).

Three things that will bite you if you skip them:

- **Append-only, flushed per record.** A 5,000-call job will crash. Resume
  must be free.
- **Log `model_returned` and `provider`.** OpenRouter routes to whichever
  provider is available, and providers differ in quantization. If you record
  only what you *requested*, you will silently mix two different models
  under one label. Consider pinning via the provider routing parameter.
- **Serialise the writes.** Concurrent appends from async tasks will
  interleave and corrupt lines.

### 8.3 Direct-mode temperature rescaling

This is the piece that makes the direct grid free.

```python
import numpy as np

def curve(logprobs: dict[str, float], temps) -> dict[float, dict[str, float]]:
    """logprobs: {token: logprob} restricted to answer tokens."""
    keys = list(logprobs)
    lp = np.array([logprobs[k] for k in keys], dtype=float)
    out = {}
    for T in temps:
        if T == 0.0:
            p = np.zeros_like(lp); p[lp.argmax()] = 1.0
        else:
            z = lp / T                 # log(p^(1/T)) = (1/T)·log p
            z -= z.max()               # stabilise
            p = np.exp(z); p /= p.sum()
        out[T] = dict(zip(keys, p))
    return out
```

Notes:
- Renormalising over only the answer tokens gives you the conditional
  distribution given the model answers at all. That is what you want.
- Check tokenization. `KEEP` and `SWITCH` may not each be one token for a
  given tokenizer. Safest fix: make the direct-mode prompt ask for a single
  letter, `K` or `S`, and verify against `top_logprobs`.
- **Validate the analytic curve empirically.** Sample 100 times at T=0.4
  and T=1.2 and check the observed proportions match the predicted ones.
  Providers do not all apply temperature identically, and top-p truncation
  interacts with it. This validation is itself a small reportable methods
  result, and it takes about a dollar.

### 8.4 Annotation

One call per CoT response, returning strict JSON:

```
You are coding responses to a decision scenario. Return ONLY a JSON object,
no prose, no markdown fences.

{"answer": one of KEEP|SWITCH|RANDOMISE|THIRD_OPTION|REFUSE|UNPARSEABLE,
 "depth":  one of L0|L1|L2|L3|MIX|NON,
 "flag_no_basis": true|false,
 "flag_rigged":   true|false}

[rubric table pasted here]

RESPONSE TO CODE:
"""{response_text}"""
```

Pass the response text only — no model name, no persona, no temperature.
The annotator must be blind to condition.

Before trusting it: hand-code 60 responses yourself, about twenty minutes,
and compute agreement. If the small annotator is under ~0.7 on the depth
codes, drop to a mid-tier annotator for this call only; it is 5,625 short
calls, so even a 5× price increase adds under $3.

### 8.5 Analysis

Minimum viable output, four plots:

1. KEEP-rate vs. temperature, one line per model, CoT — with the analytic
   direct-mode curve overlaid as a dashed line.
2. Flip rate between `default_keep` and `default_switch`, per model. **This
   is your headline.**
3. Depth distribution (`L0`–`NON`) stacked, by persona.
4. `RANDOMISE` / `THIRD_OPTION` / `REFUSE` rates by temperature.

**The statistics point that matters:** 25 samples from one model at one
temperature are *not* 25 independent subjects. They are 25 draws from one
distribution. Do not pool across models and run a chi-square as though
n = 5,625 — your confidence intervals will be far too narrow and any
reviewer with stats training will stop reading there.

Treat the (model × condition) cell as the unit. Either fit a mixed model
with a random intercept per model, or cluster-bootstrap by model. Then say
in the writeup that you did this and why. Showing you know this is itself
a signal of research taste, and it costs you one paragraph.

**Quantified, using the P(KEEP) actually measured across the five models
(0.003, 0.026, 0.148, 0.986, 0.991):** the between-model SD is 0.512, so the
standard error of any pooled claim is 0.230 — of which the replicates
contribute 0.016 and the model-to-model spread contributes 0.229. Pooling
n = 5,625 and running a chi-square reports the 0.016 as if it were the whole
error, i.e. **14× too narrow**.

The design consequence is worth acting on:

| Change | Pooled SE |
|---|---|
| 25 → 100 replicates | 0.230 → 0.229 (nothing) |
| 5 → 10 models | 0.230 → **0.162** |
| 5 → 20 models | 0.230 → **0.115** |

The replicates are doing their job — they pin down each cell's distribution
precisely, which is what they are for. They simply cannot speak to how much
that distribution varies *between* models, and that is the variance the
headline claim rests on. Since the whole study costs well under a dollar
(§6), **more models is by far the cheapest way to strengthen the result**;
raising `replicates_cot` is not.

**A threat to the within-cell assumption, found and fixed.** The 25 draws in a
cell are only iid if all 25 hit the same distribution. CoT jobs were originally
not provider-pinned (only direct mode was), and a 30-call pilot to a single
model came back split across **four providers** — Nebius 8, CoreWeave 10,
SiliconFlow 9, Alibaba 3.

Providers are demonstrably not the same computational object. Those 30 calls,
same model and same prompt, returned a *deterministic per-provider* completion
length: 2 tokens on Alibaba and SiliconFlow, 3 on CoreWeave and Nebius. And
direct-mode logprobs put an exact number on the distributional gap —
`gemma-4-26b` gives P(KEEP) of 0.854, 0.932, 0.963 and 0.986 on four providers,
a spread wider than some effects this study is trying to detect.

Worse, an unpinned CoT arm confounds **question 2 above**: it compares the
analytic direct curve against the empirical CoT curve, and if direct is pinned
while CoT scatters, the difference mixes "reasoning vs none" with "one provider
vs four".

Fixed: every mode now pins one verified provider per model, and CoT pins the
*same* provider as direct so that comparison is apples to apples
(`contracts.provider_pin`, config `pin_providers`, CONTRACTS.md §8). Verified
live — 8 CoT calls that previously split across Groq, Novita and DeepInfra now
land 8/8 on Novita. `python -m scripts.provider_report --strict` audits any
collected file and fails if a cell was served by more than one provider. Data
collected before this fix is affected: 5 of the pilot's 21 cells are split,
including one 5/5 between OpenAI and Azure.

---

## 9. Tonight's order of work

| Step | Time | Checkpoint |
|---|---|---|
| 1. Repo, config, three scenario files | 30 min | files exist |
| 2. One hardcoded call to one model, written to JSONL | 30 min | you see a real response |
| 3. Manifest builder + dry-run cost printer | 45 min | it prints a number under $5 |
| 4. Async runner with resume | 60 min | kill it mid-run, restart, no duplicates |
| 5. **Pilot: 2 models × 2 scenarios × 1 persona × T=0.8 × 10 reps = 40 calls, ~$0.02** | 15 min | **read all 40 by hand** |
| 6. Fix whatever the pilot reveals | 45 min | — |
| 7. Full CoT run | 30 min wall | ~$3 |
| 8. Annotation + plots | 60 min | four figures |

**Step 5 is the one not to skip.** You will find that your scenario wording
lets models weasel out, or that every model says the same thing, or that
the interesting variance sits somewhere you did not predict. Forty calls
and two cents buys you that, before you have spent anything real or built
anything on a broken foundation.

### Status, 2026-09-03

| Step | State | Evidence |
|---|---|---|
| 1. Repo, config, scenarios | **done** | 4 scenario files verified verbatim against §2 by test |
| 2. One hardcoded call | **done** | `scripts/smoke.py` prints text, `model_returned`, `provider`, `usage`. Prints rather than writing JSONL; the runner owns that path |
| 3. Manifest + dry-run cost | **done** | 5,685 jobs — 45 direct, 5,625 cot, 15 contamination — **$0.44** estimated, $3.05 ceiling. Under the $5 checkpoint |
| 4. Async runner with resume | **done** | 30-record kill-and-restart test: 30 records, **0 duplicate run_ids**, all `finish_reason: stop`, `max_tokens: 2000`, $0.0006 |
| 5. **Pilot — read all 40 by hand** | **NOT STARTED** | this is the gate |
| 6-8. Fix, full run, annotate, plot | not started | analysis verified against fixtures only; the four plots and `summary.md` exist in `out/` from fixture data, not real data |

Built and passing: `manifest.py`, `runner.py`, `prices.py`, `annotate.py`,
`rubric.py`, `curve.py`, `analyze.py`, 302 tests. Written by three sessions in
parallel against `CONTRACTS.md`, with `src/contracts.py` as the single source
of truth for enums, personas, `run_id`, the rubric and the measured constants.
That held: changing the model set, `max_tokens` and the provider pins
mid-flight required **zero** changes in their code, because they import the
constants rather than restate them.

Total spent to here, including ~600 probe calls for model selection: **under
$1.50**. Nothing has touched the real grid.

**Do not start step 7 before step 5.** And note that `run_id` hashes
model|scenario|persona|mode|temp|replicate and **not the prompt text or
`max_tokens`** (CONTRACTS.md §2). Change a scenario file or a token limit after
collection begins and resume will skip every job, leaving one `raw.jsonl`
silently mixing two different experiments under identical ids. Move
`data/raw.jsonl` aside whenever a prompt changes.

---

## 10. Deliberately out of scope

Deferred to Plan 2: follow-up turns, the "not die at all costs" goal arm,
frontier models, inconsistency detection, human-vs-model annotator kappa
at scale, and the reversed-default human replication.
