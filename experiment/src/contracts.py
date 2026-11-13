"""Shared vocabulary for the two-bottle experiment.

THIS MODULE IS THE EXECUTABLE HALF OF CONTRACTS.md. Import from it; do not
re-implement its constants or helpers in manifest.py / runner.py /
annotate.py / analyze.py / curve.py. If something here is wrong, change it
here (and in CONTRACTS.md) rather than working around it locally.

Owned by the scaffold session. Other sessions: read, import, do not edit
without updating CONTRACTS.md in the same commit.
"""

from __future__ import annotations

import copy
import hashlib
import re
import json
from pathlib import Path

# ---------------------------------------------------------------- paths ----

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "budget.yaml"
SCENARIO_DIR = ROOT / "scenarios"
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
RAW_PATH = DATA_DIR / "raw.jsonl"
ANNOTATED_PATH = DATA_DIR / "annotated.jsonl"
# Ad-hoc UI runs. NEVER the same file as RAW_PATH — see playground_run_id.
PLAYGROUND_PATH = DATA_DIR / "playground.jsonl"
PLAYGROUND_ANNOTATED_PATH = DATA_DIR / "playground_annotated.jsonl"
CELLS_CSV = OUT_DIR / "cells.csv"

ENV_PATH = ROOT / ".env"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# -------------------------------------------------------------- api key ----

def load_api_key(env_path: Path | None = None) -> str | None:
    """`OPENROUTER_API_KEY` from the environment, falling back to `.env`.

    Shared by `runner.py` and `annotate.py`. It lives here because both make
    OpenRouter calls and both used to answer this question differently:
    `annotate.py` read only `os.environ` while telling the operator in its own
    error message to "put it in .env", which silently did nothing. One loader,
    one behaviour. CONTRACTS.md §0's rule applies to this as much as to the
    enums: import it, do not re-implement it.
    """
    import os

    # ENV_PATH is read at call time, not bound as a default: a default would
    # freeze at import and tests could not point the fallback somewhere
    # harmless. They must be able to — with a real `.env` on disk, a test that
    # only clears the environment variable would otherwise fall through to the
    # operator's real key and spend money.
    if env_path is None:
        env_path = ENV_PATH
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        name, sep, value = line.partition("=")
        if sep and name.strip() == "OPENROUTER_API_KEY":
            return value.strip().strip("'\"") or None
    return None


# ---------------------------------------------------------------- enums ----

# PLAN.md §4. Order is canonical: use it for plot legends and table columns.
ANSWERS = ("KEEP", "SWITCH", "RANDOMISE", "THIRD_OPTION", "REFUSE", "UNPARSEABLE")

# PLAN.md §5. UNPARSEABLE has no depth analogue; a response that cannot be
# depth-coded is NON, not a missing value.
DEPTH_CODES = ("L0", "L1", "L2", "L3", "MIX", "NON")

# Order is canonical for plots. default_keep / default_switch / neutral_switch
# are the framing family and share a structure; symmetric_blind is a CONTROL
# that deliberately breaks it (neither party knows which bottle is poisoned) and
# has no flip partner; isomorph is retained but no longer in the grid (§9).
SCENARIOS = ("default_keep", "default_switch", "neutral_switch",
             "symmetric_blind", "isomorph")
PERSONAS = ("none", "average_person", "expert", "adversary_bluff")
MODES = ("direct", "cot")                        # the experimental grid
CONTAMINATION_MODE = "contamination"             # side run, PLAN.md §2
DIRECT_VALIDATION_MODE = "direct_validation"     # side run, PLAN.md §8.3
ALL_MODES = MODES + (CONTAMINATION_MODE, DIRECT_VALIDATION_MODE)

# The two side runs are NOT part of the factorial grid and must never be
# pooled with it. They use the same job dict, the same run_id function and the
# same raw.jsonl; filter on `mode`.

# -------------------------------------------------------------- personas ----

# PLAN.md §3, verbatim. `none` is deliberately absent: it means NO system
# message at all, not an empty one. Everything else is a deviation from it.
# NOTE ON THE AXIS. The first three are role assignments — they tell the model
# who it is. `adversary_bluff` is different in kind: it supplies INFORMATION
# about the game's structure. Both are system prompts and both are measured the
# same way, but the "persona" factor now mixes two manipulations, so describe
# the axis as *framing* rather than *persona* in any writeup, and read the
# adversary_bluff column as "does stating the recursion change the answer?"
# rather than as another role.
PERSONA_SYSTEM_PROMPTS = {
    "average_person": (
        "You are an ordinary person with no special training in logic, game "
        "theory, or psychology. Answer as you actually would."
    ),
    "expert": (
        "You are an expert game theorist. Answer as you actually would."
    ),
    # Promoted from the playground on 2026-09-03 after measuring it: it moved
    # gemma-4-26b from 0/4 to 4/4 on L2 reasoning proxies and 0/4 to 3/4 on L3,
    # and flipped claude-haiku-4.5 from an even split to 3 KEEP / 1 SWITCH.
    # It names every rung of the bluff ladder and endorses none, so the depth
    # codes measure whether a model can use the idea, not whether it was handed
    # a conclusion (tests/test_webui.py asserts the text contains no verdict).
    "adversary_bluff": (
        "The man across from you is adversarial. He prepared both bottles, he "
        "knows which one is poisoned, and he wants you to be the one who "
        "drinks it. He also knows that you will try to read him.\n\n"
        "So the question is not which bottle looks safe. It is what level of "
        "bluff he is playing on:\n\n"
        "- Straight: he does the obvious thing and poisons the bottle he hands "
        "you.\n"
        "- Simple bluff: he expects you to assume exactly that, so he poisons "
        "his own and hands you the safe one.\n"
        "- Double bluff: he expects you to have seen through the simple bluff "
        "and switch, so he goes back to poisoning the one he hands you.\n"
        "- Triple bluff and beyond: each further level reverses the answer "
        "again.\n\n"
        "Nothing in the setup tells you where he stopped, and the ladder does "
        "not terminate on its own. Your best chance of surviving is to work out "
        "which level he is playing on. Reason explicitly about that before you "
        "answer."
    ),
}

# ----------------------------------------------------------- depth rubric ----

# PLAN.md §5, verbatim. Paste into the annotator prompt exactly as-is.
DEPTH_RUBRIC = """\
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
- `flag_proposer` (bool) — reasons from WHO initiated the exchange rather than
  from the bottles: treats his having offered, proposed or set up the trade as
  itself evidence about which bottle is safe. "He wants the trade, so the trade
  must favour him" is this flag. Reasoning only about what he knows, or about
  what he predicts you will do, is NOT — that is the depth code's job."""

# ----------------------------------------------------------- direct mode ----

# Direct mode replaces each scenario's final paragraph with a single-letter
# request, because KEEP/SWITCH are not reliably one token for every tokenizer
# (PLAN.md §8.3). The parenthetical glosses are copied WORD FOR WORD from the
# CoT final paragraph so that direct and CoT differ only in answer format.
# tests/test_scaffold.py enforces that correspondence mechanically.
DIRECT_FINAL_INSTRUCTION = {
    "default_keep": (
        "Answer with a single letter and nothing else: "
        "K (drink the bottle already in your hand) or S (take his)."
    ),
    "default_switch": (
        "Answer with a single letter and nothing else: "
        "K (stop the trade and drink the bottle in your hand) or "
        "S (let the trade happen and drink his)."
    ),
    "isomorph": (
        "Answer with a single letter and nothing else: "
        "K (drink the cup in front of you) or S (take his)."
    ),
    # Same glosses as the scenario each is derived from, so the direct arm
    # differs from cot only in answer format for these too.
    "neutral_switch": (
        "Answer with a single letter and nothing else: "
        "K (stop the trade and drink the bottle in your hand) or "
        "S (let the trade happen and drink his)."
    ),
    "symmetric_blind": (
        "Answer with a single letter and nothing else: "
        "K (drink the bottle already in your hand) or S (take his)."
    ),
}

# Which single-token answers the temperature curve is renormalised over.
DIRECT_ANSWER_TOKENS = {"K": "KEEP", "S": "SWITCH"}

# Fixed direct-mode request parameters. See CONTRACTS.md §8.
DIRECT_TEMPERATURE = 1.0   # always: we want the base distribution, unwarped
DIRECT_MAX_TOKENS = 4      # small, but >1 so a leading-whitespace token cannot
                           # consume the only slot
DIRECT_TOP_LOGPROBS = 10
COT_MAX_TOKENS = 2000      # PLAN.md §8.2 uses 900; measured, that truncated
                           # 7/12 gemini calls and 5/12 llama calls mid-answer,
                           # destroying the final-line answer. 2000 gave zero
                           # truncation for every model in the shipped config.
CONTAMINATION_MAX_TOKENS = 200
CONTAMINATION_TEMPERATURE = 1.0

# Provider routing for direct mode. `require_parameters: true` is NOT enough:
# it routed mistral-nemo to Novita, which then returned no logprobs at all
# (measured 2026-09-03). Only an explicit `only` list is reliable.
# EXACTLY ONE provider per model. OpenRouter's `only` is a permitted SET, not
# a preference order, so a two-element list still lets the router pick either
# one and you are back to mixing distributions — the precise failure this pin
# exists to prevent (the same prompt to gemma-4-26b gave P(KEEP) from 0.854 to
# 0.986 across providers). One name, `allow_fallbacks: False`, no ambiguity.
# If a pinned provider is down the call FAILS, which is correct: a hole in the
# grid is recoverable, silently mixed distributions are not.
DIRECT_PROVIDER_PIN = {
    "qwen/qwen3-30b-a3b-instruct-2507": {"only": ["coreweave"], "allow_fallbacks": False},
    "meta-llama/llama-3.1-8b-instruct": {"only": ["novita"], "allow_fallbacks": False},
    "ibm-granite/granite-4.1-8b": {"only": ["coreweave"], "allow_fallbacks": False},
    "google/gemma-4-26b-a4b-it": {"only": ["parasail"], "allow_fallbacks": False},
    "openai/gpt-4o-mini": {"only": ["openai"], "allow_fallbacks": False},
}

# Verified-working alternates, for MANUAL failover only. Switching mid-study
# changes the measured distribution, so a switch means re-running that model's
# whole direct grid, not patching the missing cells.
DIRECT_PROVIDER_ALTERNATES = {
    "qwen/qwen3-30b-a3b-instruct-2507": ["streamlake"],
    "meta-llama/llama-3.1-8b-instruct": [],   # CoreWeave has no answer tokens
    "ibm-granite/granite-4.1-8b": [],         # sole endpoint
    "google/gemma-4-26b-a4b-it": ["nextbit", "cloudflare", "darkbloom"],
    "openai/gpt-4o-mini": ["azure"],
}

# ---------------------------------------------------------- provider pinning --

# CoT and contamination calls were originally left unpinned, on the reasoning
# that only direct mode needs a specific provider (for logprobs). That was
# wrong, for two measured reasons.
#
# 1. Providers are not the same computational object. 30 CoT calls to
#    qwen3-30b-a3b with an identical prompt were served by four providers and
#    returned a DETERMINISTIC per-provider token count — Alibaba and
#    SiliconFlow 2, CoreWeave and Nebius 3. Different tokenisation or
#    quantization, same nominal model. Direct-mode logprobs put numbers on the
#    distributional gap: gemma-4-26b gave P(KEEP) of 0.854, 0.932, 0.963 and
#    0.986 on four providers — exact conditional probabilities, not samples,
#    and a spread wider than some effects this study is trying to detect.
#
# 2. It confounds PLAN.md's second question. That question compares the
#    analytic DIRECT curve against the empirical COT curve. Direct is pinned
#    to one provider; unpinned CoT scatters across four. Any difference then
#    mixes "reasoning vs no reasoning" with "provider A vs providers B, C, D",
#    and the study cannot separate them after the fact.
#
# So every mode pins, and CoT pins to the SAME provider as direct wherever one
# exists, which is what makes the direct-vs-CoT comparison apples to apples.
# A pinned provider being unavailable makes the call fail rather than reroute:
# the runner retries, then writes nothing, and resume refills the gap later.
# A hole is recoverable; a silently mixed distribution is not.
COT_PROVIDER_PIN = {model: copy.deepcopy(pin) for model, pin in DIRECT_PROVIDER_PIN.items()}


def provider_pin(model: str, mode: str, *, enabled: bool = True) -> dict | None:
    """Provider routing for one job, or None to let OpenRouter choose.

    `enabled=False` (config `pin_providers: false`) restores the old
    free-routing behaviour for throughput, at the cost of the confound above.
    Direct mode ALWAYS pins regardless: without it the logprobs may simply not
    come back, so there is nothing to trade off.
    """
    if mode in ("direct", DIRECT_VALIDATION_MODE):
        pin = DIRECT_PROVIDER_PIN.get(model)
    elif not enabled:
        return None
    else:
        pin = COT_PROVIDER_PIN.get(model)
    # deepcopy, not dict(): a shallow copy shares the `only` LIST, so one job
    # appending to it would rewrite the pin every other job is using.
    return copy.deepcopy(pin) if pin else None


# Models whose direct-mode logprobs were verified usable end to end.
# Everything else gets a sampled letter but no temperature curve. With the
# shipped config this is all five, which it was not before the model swap.
DIRECT_CURVE_MODELS = frozenset(DIRECT_PROVIDER_PIN)

# Measured median characters of visible `message.content` in CoT mode. A model
# below ~120 emits an answer with no reasoning attached and CANNOT be
# depth-coded (PLAN.md §5); its depth codes will be uniformly L0/NON and its
# CoT arm is not really a CoT arm. This is a property of the model, not the
# prompt: adding "think through the problem first" to the scenario was measured
# and did NOT change it. Analysis must report depth results per model, never
# pooled, and exclude these from any CoT-vs-direct reasoning comparison.
COT_CONTENT_CHARS = {
    "qwen/qwen3-30b-a3b-instruct-2507": 6,      # NO chain of thought
    "meta-llama/llama-3.1-8b-instruct": 1344,
    "ibm-granite/granite-4.1-8b": 1056,
    "google/gemma-4-26b-a4b-it": 1934,
    "openai/gpt-4o-mini": 604,
}
NO_COT_MODELS = frozenset(m for m, n in COT_CONTENT_CHARS.items() if n < 120)

# --------------------------------------------- randomise-clause variants ----

# config `allow_randomise_in_prompt` toggles paragraph 4 only. The final
# instruction is NEVER touched, so answer format stays constant across the
# toggle (same reasoning as PLAN.md §2's warning about confounds).
# The `False` strings are verbatim from PLAN.md; the `True` strings are a
# scaffold decision (PLAN.md does not spell them out) — see CONTRACTS.md §6.
RANDOMISE_CLAUSE = {
    "default_keep": {
        False: (
            "You cannot refuse, delay, test the bottles, drink neither, or "
            "decide by any random method such as a coin flip or a mental die "
            "roll. You must commit to one of the two bottles."
        ),
        True: (
            "You cannot refuse, delay, test the bottles, or drink neither. "
            "You must commit to one of the two bottles."
        ),
    },
    "isomorph": {
        False: (
            "You cannot refuse, forfeit, test the cups, drink neither, or "
            "decide by any random method. You must commit to one of the two cups."
        ),
        True: (
            "You cannot refuse, forfeit, test the cups, or drink neither. "
            "You must commit to one of the two cups."
        ),
    },
}
# Same clause, separate object: these must never alias, or a future edit to
# one scenario would silently rewrite the other.
RANDOMISE_CLAUSE["default_switch"] = dict(RANDOMISE_CLAUSE["default_keep"])
# Both new scenarios are bottles, so they take the bottle wording verbatim.
RANDOMISE_CLAUSE["neutral_switch"] = dict(RANDOMISE_CLAUSE["default_keep"])
RANDOMISE_CLAUSE["symmetric_blind"] = dict(RANDOMISE_CLAUSE["default_keep"])

# ------------------------------------------------------------- scenarios ----

def load_scenario(name: str) -> str:
    """Full scenario text, exactly as stored on disk (5 paragraphs)."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; expected one of {SCENARIOS}")
    return (SCENARIO_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


def scenario_paragraphs(name: str) -> list[str]:
    """The 5 paragraphs. Index 3 is the randomise clause, 4 is the answer format."""
    paras = load_scenario(name).split("\n\n")
    if len(paras) != 5:
        raise ValueError(f"{name}.txt has {len(paras)} paragraphs, expected 5")
    return paras


def scenario_prompt(name: str, mode: str, allow_randomise_in_prompt: bool = False) -> str:
    """The user-message body for one (scenario, mode) pair."""
    if mode not in MODES and mode != DIRECT_VALIDATION_MODE:
        raise KeyError(f"unknown mode {mode!r}; expected one of "
                       f"{MODES + (DIRECT_VALIDATION_MODE,)}")
    paras = scenario_paragraphs(name)
    paras[3] = RANDOMISE_CLAUSE[name][bool(allow_randomise_in_prompt)]
    if mode in ("direct", DIRECT_VALIDATION_MODE):
        paras[4] = DIRECT_FINAL_INSTRUCTION[name]
    return "\n\n".join(paras)


def contamination_prompt(name: str) -> str:
    """Contamination check: the question, then the scenario minus its answer-format
    paragraph (we are not asking for a decision here)."""
    question = (SCENARIO_DIR / "contamination_check.txt").read_text(encoding="utf-8").strip()
    return question + "\n\n" + "\n\n".join(scenario_paragraphs(name)[:4])


# -------------------------------------------------------------- messages ----

def build_messages(
    scenario: str,
    persona: str,
    mode: str,
    allow_randomise_in_prompt: bool = False,
) -> list[dict]:
    """The exact `messages` array that goes into the job dict.

    persona == "none" emits NO system message. Do not substitute an empty
    system message: an empty system turn is not the same input.
    """
    if persona not in PERSONAS:
        raise KeyError(f"unknown persona {persona!r}; expected one of {PERSONAS}")
    messages: list[dict] = []
    if persona != "none":
        messages.append({"role": "system", "content": PERSONA_SYSTEM_PROMPTS[persona]})
    messages.append(
        {"role": "user", "content": scenario_prompt(scenario, mode, allow_randomise_in_prompt)}
    )
    return messages


# ---------------------------------------------------------------- run_id ----

def run_id_key(model: str, scenario: str, persona: str, mode: str,
               temperature: float, replicate: int) -> str:
    """The exact pre-image string that gets hashed. Formatting is part of the
    contract: temperature is always one decimal place, replicate is a bare int.
    """
    return f"{model}|{scenario}|{persona}|{mode}|{float(temperature):.1f}|{int(replicate):d}"


def run_id(model: str, scenario: str, persona: str, mode: str,
           temperature: float, replicate: int) -> str:
    """sha1 of run_id_key(...), full 40-char lowercase hex. PLAN.md §8.1.

    Direct-mode jobs must pass temperature=1.0 and replicate=0 so that the id
    matches the request that is actually sent.
    """
    key = run_id_key(model, scenario, persona, mode, temperature, replicate)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def build_contamination_messages(scenario: str) -> list[dict]:
    """Messages for the contamination check (PLAN.md §2). No system prompt,
    no answer-format paragraph, no persona axis."""
    return [{"role": "user", "content": contamination_prompt(scenario)}]


# ------------------------------------------- direct-mode logprob extraction ----

def answer_logprobs(choice: dict) -> tuple[dict[str, float] | None, str]:
    """Pull the {"K": logprob, "S": logprob} pair out of one API `choices[0]`.

    Returns (logprobs, status). status is "ok" or a reason the record cannot
    feed the temperature curve. CALLERS MUST CHECK STATUS. With the shipped
    config and DIRECT_PROVIDER_PIN applied every model should return "ok"; a
    non-ok status for a model in DIRECT_CURVE_MODELS is a pinning or provider
    failure, not a model limit, and is worth stopping for. Failures cluster by
    model and provider, never by condition, so they are not missing at random
    (CONTRACTS.md §8).

    Probabilities of every token normalising to the same letter (e.g. "K",
    " K", "k") are summed in probability space before being returned as a log.
    """
    import math

    lp = choice.get("logprobs")
    content = lp.get("content") if isinstance(lp, dict) else None
    if not content:
        return None, "no_logprobs"

    for position in content[:DIRECT_MAX_TOKENS]:
        acc: dict[str, float] = {}
        for entry in position.get("top_logprobs") or []:
            letter = str(entry.get("token", "")).strip().upper()
            if letter in DIRECT_ANSWER_TOKENS:
                acc[letter] = acc.get(letter, 0.0) + math.exp(entry["logprob"])
        # Both letters must be present: we renormalise over exactly {K, S}, and
        # a position offering only one of them is not the answer position.
        if len(acc) == 2:
            return {k: math.log(v) for k, v in acc.items()}, "ok"

    return None, "no_answer_tokens"


# ---------------------------------------------------- response validation ----

def is_usable_response(status_code: int, data: object) -> tuple[bool, str]:
    """Whether an OpenRouter reply may be written to raw.jsonl.

    OpenRouter returns **HTTP 200 with an `error` object and no `choices`** when
    the upstream provider fails — measured on 2026-09-03:

        {"id": "gen-...", "error": {"message": "Provider returned error", "code": 429}}

    `raise_for_status()` does not catch that. Writing it as a completed record
    poisons resume permanently, because the run_id then looks done and the
    call is never retried. Check this before every append.
    """
    if status_code >= 400:
        return False, f"http_{status_code}"
    if not isinstance(data, dict):
        return False, "not_an_object"
    if "error" in data:
        err = data.get("error")
        code = err.get("code") if isinstance(err, dict) else None
        return False, f"error_body_{code}"
    choices = data.get("choices")
    if not choices:
        return False, "no_choices"
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict) or message.get("content") is None:
        return False, "no_message_content"
    return True, "ok"


# ------------------------------------------------------------- playground ----

PLAYGROUND_PREFIX = "pg"


def playground_run_id(model: str, temperature: float, replicate: int,
                      messages: list[dict], stamp: str) -> str:
    """Identifier for one ad-hoc playground call.

    WHY THIS EXISTS. The grid's `run_id` hashes only
    model|scenario|persona|mode|temp|replicate — deliberately, since PLAN.md
    §8.1 fixes those six fields. It does NOT cover the prompt text. The
    playground's whole purpose is varying prompt and role text, so an ad-hoc
    call at, say, gemma/default_keep/expert/0.8/rep-0 with an edited system
    prompt would produce a `run_id` IDENTICAL to a real grid job's. In one
    file that is silent corruption in both directions: the real run's resume
    would skip that job as already done, or the analysis would read ad-hoc
    output as experimental data.

    Two defences, both required and neither sufficient alone:
      1. playground records are written to PLAYGROUND_PATH, never RAW_PATH;
      2. this id hashes the EXACT messages plus a per-run stamp, and carries a
         "pg-" prefix, so it can never equal a 40-hex grid run_id.
    """
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(
        f"{model}|{float(temperature):.4f}|{int(replicate):d}|{stamp}|{payload}".encode("utf-8")
    ).hexdigest()
    return f"{PLAYGROUND_PREFIX}-{digest}"


def is_playground_id(run_id: str) -> bool:
    return str(run_id).startswith(PLAYGROUND_PREFIX + "-")


# ------------------------------------------------------- reasoning depth ----

# `max_tokens` is a CEILING, not a target: the model stops when it emits EOS,
# so raising it never lengthens a response. Measured — gemma-4-26b given 8,000
# tokens still stopped at ~518. Two levers actually move reasoning length.
#
# LEVER 1, the provider `reasoning` budget (OpenRouter `reasoning: {max_tokens}`).
# Only models advertising `reasoning` in supported_parameters accept it; of the
# shipped five that is gemma-4-26b alone. It works, but it moves the trace into
# `message.reasoning` and SHRINKS `message.content`:
#     off        content 2237 chars   reasoning     0   out   518 tok
#     1000       content  371 chars   reasoning 11944   out 3344 tok
#     4000       content 1163 chars   reasoning 18552   out 5344 tok
# The annotator depth-codes `content` (PLAN.md §5), so this buys tokens the
# rubric never sees. Useful for reading in the playground; not for the grid.
#
# LEVER 2, an explicit instruction, is the one that works everywhere. Measured
# median `content` characters, control -> instructed:
#     gemma-4-26b     2698 ->  3915   (1.5x)
#     granite-4.1-8b  1198 ->  3415   (2.9x)
#     gpt-4o-mini      409 ->  2662   (6.5x)
#     llama-3.1-8b     467 ->  2966   (6.4x)
#     qwen3-30b-a3b      6 -> 21064   (3511x)
#
# That last row matters: qwen3-30b-a3b is in NO_COT_MODELS because it answers
# in three tokens, and a weak "think through the problem first" changed nothing.
# It is not incapable of reasoning — it was not being asked firmly enough.
#
# THIS TEXT IS NOT PART OF THE GRID. PLAN.md §2's wording is the manipulation,
# and instructing depth changes what is measured from "how does it reason?" to
# "how does it reason when told to". That is a legitimate experiment — and a
# good Plan-2 arm — but it is a different one, so this lives here for the
# playground and is never inserted by the manifest.
DEPTH_INSTRUCTION = (
    "Before your final line, reason explicitly and at length. Work through at "
    "least three levels: what he wants you to do, what he expects you to "
    "conclude, and what he expects you to conclude about his expectation. "
    "Consider the regress and say whether it terminates. Do not shortcut."
)


def with_depth_instruction(scenario_text: str) -> str:
    """Insert DEPTH_INSTRUCTION before the answer-format paragraph.

    Placed second-to-last on purpose: the final paragraph is the answer
    contract, and moving or rewording it would make the answer format a
    confound (PLAN.md §2).
    """
    paragraphs = scenario_text.split("\n\n")
    if len(paragraphs) < 2:
        return scenario_text + "\n\n" + DEPTH_INSTRUCTION
    return "\n\n".join(paragraphs[:-1] + [DEPTH_INSTRUCTION, paragraphs[-1]])


# ------------------------------------------------- playground-only presets ----
#
# NONE OF THIS IS THE GRID. The grid's three personas are PERSONA_SYSTEM_PROMPTS
# (PLAN.md §3) and its five models are config/budget.yaml. What follows is
# offered in the UI so arms can be tried before committing them to a plan, and
# `build_messages` still refuses any persona outside PERSONAS.

# `survive_at_all_costs` is PLAN.md §10's deferred "not die at all costs" goal
# arm, prototyped here. It changes the objective rather than the reasoning
# style: measured, it left L2/L3 untouched but collapsed both tested models to
# SWITCH 4/4. Still playground-only — a goal arm is a different study.
# (`adversary_bluff` was promoted into PERSONA_SYSTEM_PROMPTS on 2026-09-03.)
PLAYGROUND_PERSONAS = {
    "survive_at_all_costs": (
        "Your only goal is surviving. You must survive at all costs. Nothing "
        "else matters — not fairness, not dignity, not what he deserves, not "
        "what a reasonable person would do. Choose whichever bottle gives you "
        "the highest probability of living."
    ),
}

# Models offered in the playground beyond the five in config/budget.yaml, for
# checking whether a result is a property of the scenario or of small models.
# Prices are per million tokens, verified 2026-09-03. These are 10-500x the
# grid's models per call, which is why the UI prices a batch before running it
# and why none of them belong in budget.yaml without a deliberate decision.
PLAYGROUND_EXTRA_MODELS = [
    {"id": "openai/gpt-5-nano",            "tier": "extra", "in": 0.05, "out": 0.40},
    {"id": "openai/gpt-5-mini",            "tier": "extra", "in": 0.25, "out": 2.00},
    {"id": "google/gemini-3.1-flash-lite", "tier": "extra", "in": 0.25, "out": 1.50},
    {"id": "deepseek/deepseek-v3.2",       "tier": "extra", "in": 0.269, "out": 0.40},
    {"id": "anthropic/claude-haiku-4.5",   "tier": "frontier", "in": 1.00, "out": 5.00},
    {"id": "anthropic/claude-sonnet-5",    "tier": "frontier", "in": 2.00, "out": 10.00},
]


# ------------------------------------------------------ answer extraction ----

# Measured on the 40-call pilot: only 21 of 40 responses actually end with a
# bare answer word, and non-compliance is NOT random — in default_keep the
# conditional KEEP rate is 0.231 among compliant responses and 0.667 among
# non-compliant ones. Extraction is therefore not a tidying step. Parse the
# hard cases badly and you move the headline.
#
# This returns a METHOD alongside the answer rather than a bare label, so a
# caller can weigh a bare final line differently from a guess scraped out of
# prose. `analyze.py` should report the method mix; anything below "final_line"
# deserves the annotator's opinion too, and a disagreement deserves a human.
#
# It does NOT replace the annotator. The annotator reads the whole response and
# can tell "I would switch, but..." from "switch"; no regex can. Use both and
# treat disagreement as a flag (scripts/extraction_check.py).

_ANSWER_RE = re.compile(r"\b(KEEP|SWITCH)\b", re.IGNORECASE)
_TERMINAL_RE = re.compile(
    r"\b(?:answer|choice|decision|verdict)\s*(?:is|:)\s*\**\s*(KEEP|SWITCH)\b"
    r"|\b(?:I|you)\s+(?:would|will|should|must)\s+(KEEP|SWITCH)\b"
    r"|\b(?:therefore|so|thus|hence)[,:]?\s+\**(KEEP|SWITCH)\b",
    re.IGNORECASE)


def _strip_parens(text: str) -> str:
    """Remove parenthesised spans, innermost first.

    The scenario's own final line is `KEEP (drink the bottle already in your
    hand) or SWITCH (take his)`, so a response that echoes the instruction
    carries both words. The glosses are parenthesised; the answer words are
    not, which is why this alone does not disambiguate an echo — see
    `echoes_instruction`.
    """
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\([^()]*\)", "", text)
    return text


def echoes_instruction(text: str) -> bool:
    """True when the text repeats the scenario's own answer-format line.

    An echo puts KEEP and SWITCH in the output in the scenario's order without
    either being a choice, so a last-mention rule reads the echo's final word
    as the answer. Detect it and fall back rather than guess.
    """
    return bool(re.search(r"exactly one word\s*:?\s*\**\s*KEEP", text, re.IGNORECASE))


def extract_answer(text: str) -> tuple[str | None, str]:
    """(answer, method) for one response. Methods, most to least trustworthy:

    ``final_line``   the last non-empty line is a bare KEEP or SWITCH — what the
                     scenario asks for, and the only method that needs no
                     inference.
    ``terminal``     a closing declaration: "the answer is KEEP", "therefore,
                     SWITCH", "I would keep".
    ``last_caps``    the last ALL-CAPS mention. Models shout their conclusion
                     and lowercase the discussion, so case carries signal.
    ``last_mention`` the last mention outside parentheses. Weakest: the last
                     mention is often part of the reasoning rather than its
                     conclusion — "So the bottle you are holding is safe" ends
                     a KEEP argument whose last named word is SWITCH.
    ``none``         no answer word at all, or an unresolved instruction echo.
    """
    if not text or not text.strip():
        return None, "none"
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]

    bare = lines[-1].strip("*#>_`.!:;,\"' ").upper()
    if bare in ("KEEP", "SWITCH"):
        return bare, "final_line"

    tail = "\n".join(lines[-3:])
    matches = _TERMINAL_RE.findall(tail) or _TERMINAL_RE.findall(text)
    if matches:
        last = [g for g in matches[-1] if g]
        if last:
            return last[-1].upper(), "terminal"

    if echoes_instruction(text):
        # The echo supplies both words in a fixed order; anything after it is
        # what we want, and if there is nothing after it there is no answer.
        after = text[text.upper().rindex("EXACTLY ONE WORD"):]
        caps = re.findall(r"\b(KEEP|SWITCH)\b", after)
        if len(caps) > 2:
            return caps[-1].upper(), "last_caps"
        return None, "none"

    caps = re.findall(r"\b(KEEP|SWITCH)\b", text)
    if caps:
        return caps[-1].upper(), "last_caps"

    plain = _ANSWER_RE.findall(_strip_parens(text))
    if plain:
        return plain[-1].upper(), "last_mention"
    return None, "none"
