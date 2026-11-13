"""Answer extraction and depth coding — PLAN.md §8.4, CONTRACTS.md §5.

Reads `data/raw.jsonl`, sends **only the `cot` records** to the configured
annotator one at a time, and appends one line per `run_id` to
`data/annotated.jsonl`.

Three rules from the contract drive most of the code below.

*Whitelist `cot`, never blacklist `direct`.* `raw.jsonl` carries four modes
(`contracts.ALL_MODES`): the factorial grid's `direct` and `cot` plus the two
side runs, `contamination` and `direct_validation`. Only `cot` is annotated —
direct-mode answers come from the emitted letter and their distribution from
the logprobs, and neither side run is a result. A blacklist would have let both
side runs leak in the moment they were added.

*Never write a line you cannot stand behind, and never write one you can't.* A
parse failure still gets a line — `answer: UNPARSEABLE`, `depth: NON`, raw text
kept — because a missing line is indistinguishable from a job that never ran.
But a *call* failure gets no line at all, for the same reason the runner writes
none (CONTRACTS.md §4): the `run_id` would look done and resume would never
retry it. `contracts.is_usable_response` is the gate, because OpenRouter
answers HTTP 200 with an `error` body and no `choices` when a provider fails.

*One line per `run_id`.* Resume reads the ids already in the output and skips
them. Re-annotating with a different model goes to a *new file*: nothing
downstream can tell a re-annotation from a double-write, so mixing annotators
in one file raises rather than appends.

    python -m src.annotate --config config/budget.yaml [--limit N] [--input path]
    python -m src.annotate --export-for-handcoding 60 --out data/handcode.csv
    python -m src.annotate --kappa data/handcode.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Sequence

import yaml

from src import contracts as c
from src.rubric import build_annotation_prompt

# ------------------------------------------------------------- parameters ----

ANNOTATION_TEMPERATURE = 0       # PLAN.md §8.4: the coding must be reproducible
ANNOTATION_MAX_TOKENS = 200      # the reply is ~40 tokens; 200 leaves slack
MAX_ATTEMPTS = 5                 # same as the runner
BACKOFF_BASE = 2
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
#: The key or the credit is wrong: every remaining call fails identically, so
#: stop rather than burning 5,625 × 5 attempts to find out.
#: **403 is deliberately NOT here**, and used to be. It is per-request — a
#: provider's moderation layer rejecting one prompt — and this annotator is fed
#: model reasoning about poisoning someone, so a flagged prompt is exactly the
#: thing to expect. One flagged response must cost one record, not the pass.
#: `runner.py` reached the same conclusion for the same prompts (see its
#: FATAL_STATUS); the two constants disagreed, and annotate's answer was wrong.
#: 402 is added for the same symmetry: out of credit really is fatal.
FATAL_STATUS = frozenset({401, 402})
DEFAULT_CONCURRENCY = 12

UNPARSEABLE_WARN_RATE = 0.02     # PLAN.md §4's target
DEPTH_KAPPA_FLOOR = 0.7          # PLAN.md §8.4's annotator-swap threshold

HANDCODE_COLUMNS = ("run_id", "response_text", "answer", "depth",
                    "flag_no_basis", "flag_rigged", "flag_proposer")
# Characters Excel, Numbers and LibreOffice treat as the start of a formula.
FORMULA_PREFIXES = ("=", "+", "@", "-", "|")
#: Fields scored against a hand-coded CSV. `flag_proposer` is deliberately NOT
#: here: it was added after the pilot, so a CSV exported before it exists — or
#: one a hand-coder left blank because the column was not in their copy — must
#: still score. It is added per-file by `_label_columns` when the column is
#: actually present, so a fresh export does get it scored.
LABEL_COLUMNS = ("answer", "depth", "flag_no_basis", "flag_rigged")

#: Optional label columns: scored when the CSV has them, ignored when it does not.
OPTIONAL_LABEL_COLUMNS = ("flag_proposer",)


def _label_columns(fieldnames) -> tuple[str, ...]:
    """The label columns to score for one CSV: the required set, plus any
    optional column the file actually carries."""
    present = set(fieldnames or ())
    return LABEL_COLUMNS + tuple(c for c in OPTIONAL_LABEL_COLUMNS if c in present)

_TRUE_STRINGS = {"true", "yes", "y", "1", "t"}
_FALSE_STRINGS = {"false", "no", "n", "0", "f"}

_FENCE_OPEN = re.compile(r"^```[A-Za-z0-9_+-]*[ \t]*\r?\n?")
_FENCE_CLOSE = re.compile(r"\r?\n?[ \t]*```[ \t]*$")

Caller = Callable[[str], Awaitable[tuple[int, object]]]


class AnnotatorMismatch(RuntimeError):
    """The output file was written by a different annotator (CONTRACTS.md §5)."""


class FatalAnnotatorError(RuntimeError):
    """The API rejected us outright — retrying 5,625 times will not help."""


class HandCodingError(ValueError):
    """A hand-coded CSV cell is not a label from the enums."""


def warn(message: str) -> None:
    """stderr, but after stdout has caught up — otherwise a warning about the
    run jumps ahead of the header describing it."""
    sys.stdout.flush()
    print(message, file=sys.stderr, flush=True)


def utc_now() -> str:
    """CONTRACTS.md §4's timestamp format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------- io ----

def load_config(path: str | Path = c.CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> tuple[list[dict], list[int]]:
    """Every object in a JSONL file, plus the 1-indexed lines that would not
    parse. Bad lines are reported rather than raised: one corrupt line in
    `raw.jsonl` must not cost the other 5,624 annotations."""
    path = Path(path)
    if not path.exists():
        return [], []
    records, bad = [], []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad.append(lineno)
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                bad.append(lineno)
    return records, bad


def read_annotated_index(path: str | Path) -> tuple[set[str], Counter]:
    """The `run_id`s already annotated and which annotator produced them.

    One pass, because on the full study this file has 5,625 lines and both
    facts are needed before the first call goes out.
    """
    records, _ = read_jsonl(path)
    ids = {rec["run_id"] for rec in records if "run_id" in rec}
    # The REQUESTED id, falling back to the returned one for lines written
    # before that field existed. Guarding on the returned id would lock the
    # file the first time OpenRouter echoes a routed variant (CONTRACTS.md §4).
    models = Counter(rec.get("annotator_requested") or rec["annotator_model"]
                     for rec in records if rec.get("annotator_model"))
    return ids, models


def read_annotated_ids(path: str | Path) -> set[str]:
    return read_annotated_index(path)[0]


def cot_records(records: Iterable[dict]) -> list[dict]:
    """The records to annotate: `mode == "cot"` and nothing else.

    A whitelist. `direct`, `contamination` and `direct_validation` all live in
    the same file and none of them is annotated (CONTRACTS.md §5, §8, §9).
    """
    return [rec for rec in records if rec.get("mode") == "cot"]


def response_text(record: dict) -> str:
    """The subject model's visible reply — `choices[0].message.content`.

    Only `content`. A thinking model's `message.reasoning` is deliberately not
    consulted: PLAN.md §5 codes what the model actually emitted, and the
    shipped config excludes models that hide their reasoning
    (`contracts.NO_COT_MODELS` records the one that emits none at all).
    """
    try:
        content = record["response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


# ---------------------------------------------------------------- parsing ----

def strip_fences(text: str) -> str:
    """Drop one layer of markdown fencing, with or without a language tag.

    The prompt says "no markdown fences" and small models produce them anyway.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_OPEN.sub("", stripped)
        stripped = _FENCE_CLOSE.sub("", stripped)
    return stripped.strip()


def extract_json_object(text: str) -> str | None:
    """The first balanced `{...}` in `text`, or None.

    Brace-counting rather than a regex, so a nested object parses; string
    contents and escapes are skipped so a `}` inside a value cannot end it.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def coerce_bool(value: object) -> bool | None:
    """`true`, `"true"`, `"Yes"`, `1` → True. Anything undecidable → None.

    Lenient about the shape, strict about the meaning: a flag the annotator
    filled in with prose is a parse failure, not a False.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_STRINGS:
            return True
        if token in _FALSE_STRINGS:
            return False
    return None


def _normalise_label(value: object) -> str:
    return str(value).strip().upper() if value is not None else ""


def _clip(value: object, limit: int = 40) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


FAILED_FIELDS = {"answer": "UNPARSEABLE", "depth": "NON",
                 "flag_no_basis": False, "flag_rigged": False,
                 "flag_proposer": False, "flag_proposer_present": False}


def parse_annotation(raw_output: str,
                     allowed_answers: Sequence[str] | None = None,
                     ) -> tuple[dict, str | None]:
    """Parse one annotator reply into the four coded fields.

    Returns `(fields, error)`. `error` is None on success; otherwise `fields`
    is CONTRACTS.md §5's failure record — `UNPARSEABLE` / `NON` / both flags
    false — and `error` is a short reason for the hand-review queue.

    `allowed_answers` is the config's list (CONTRACTS.md §1): a label it
    excludes is not a permitted result, so `RANDOMISE` coming back under a
    config that dropped it is a failure, not a finding. `UNPARSEABLE` is always
    permitted, being a QA flag rather than an answer.
    """
    permitted = set(allowed_answers or c.ANSWERS) | {"UNPARSEABLE"}
    failure = dict(FAILED_FIELDS)

    blob = extract_json_object(strip_fences(raw_output or ""))
    if blob is None:
        return failure, "no_json_object"
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return failure, "invalid_json"
    if not isinstance(data, dict):
        return failure, "not_an_object"

    answer = _normalise_label(data.get("answer"))
    if answer not in c.ANSWERS:
        return failure, f"bad_answer:{_clip(data.get('answer'))}"
    if answer not in permitted:
        return failure, f"disallowed_answer:{answer}"

    depth = _normalise_label(data.get("depth"))
    if depth not in c.DEPTH_CODES:
        return failure, f"bad_depth:{_clip(data.get('depth'))}"

    flags = {}
    for name in ("flag_no_basis", "flag_rigged"):
        flag = coerce_bool(data.get(name))
        if flag is None:
            return failure, f"bad_{name}:{_clip(data.get(name))}"
        flags[name] = flag

    # `flag_proposer` was added after the pilot (CONTRACTS.md §5) and is
    # OPTIONAL on purpose. Requiring it would mean an annotator that omits one
    # secondary boolean discards a perfectly good answer and depth with it —
    # and it would retroactively invalidate every record coded before it
    # existed. Absent means False; only an explicitly unparseable value fails.
    # `flag_proposer_present` records which happened, so an annotator that
    # silently never emits it shows up as a column of False rather than as a
    # finding that nobody reasons about the proposer.
    present = "flag_proposer" in data
    proposer = coerce_bool(data.get("flag_proposer")) if present else False
    if proposer is None:
        return failure, f"bad_flag_proposer:{_clip(data.get('flag_proposer'))}"
    flags["flag_proposer"] = proposer
    flags["flag_proposer_present"] = present

    return {"answer": answer, "depth": depth, **flags}, None


#: A response with nothing in it but the answer word cannot carry a depth code.
#: PLAN.md §5's `L0` is "No model of the opponent. Instinct, superstition, 'I'd
#: just go with the first one'", which is exactly what a bare `SWITCH` is.
#:
#: Measured on the 40-call pilot plus 30 probe records (70 cot responses), the
#: distribution of non-answer word counts is bimodal with a wide empty gap:
#: 44 responses at **0**, one at **3** ('You must say "KEEP".'), and then
#: nothing until **24**. The threshold sits in that gap with a 3x margin to the
#: nearest response that does reason, so it is read off the data rather than
#: chosen. Raise it only with new measurements.
#:
#: This exists because the annotator does not do it. Given a response that is
#: the single token `KEEP`, `mistral-nemo` returned `depth: "L2"` — "models the
#: opponent modelling the chooser" — for all 70 pilot records it coded L1 (43)
#: or L2 (26) and never once L0, including on all 44 zero-reasoning responses.
#: It infers the reasoning it expects to accompany an answer. That silently
#: fabricated PLAN.md's entire secondary outcome for 63% of the pilot.
DEPTH_FLOOR_MAX_WORDS = 8

#: Words that are the answer itself rather than reasoning about it.
_ANSWER_WORDS = frozenset({"keep", "switch", "k", "s"})


def codeable_reasoning_words(text: str) -> int:
    """How many words the response contains that are not the answer itself."""
    return sum(1 for w in re.findall(r"[a-z']+", (text or "").lower())
               if w not in _ANSWER_WORDS)


def has_codeable_reasoning(text: str) -> bool:
    """Whether there is any reasoning in `text` for the depth rubric to code."""
    return codeable_reasoning_words(text) >= DEPTH_FLOOR_MAX_WORDS


def build_annotated_record(*, run_id: str, raw_output: str, annotator_model: str,
                           annotator_requested: str | None = None,
                           annotator_provider: str | None = None,
                           usage: dict | None = None, ts: str | None = None,
                           allowed_answers: Sequence[str] | None = None,
                           response_text: str | None = None) -> dict:
    """One `data/annotated.jsonl` line — CONTRACTS.md §5.

    `raw_annotator_output` is kept even when parsing succeeded: it is the only
    way to audit a disputed code without paying for the call again.
    `parse_ok` separates "the annotator returned invalid JSON" from "the
    annotator said UNPARSEABLE", which are different failures and both get
    hand-reviewed.

    `annotator_model` is the id the API *returned*, per §5. `annotator_requested`
    is the id we *asked* for, and the two can differ — §4 says so of
    `model_returned`. Both are recorded because the resume guard needs the
    requested one (a returned variant like `…-nemo:free` would otherwise never
    match the config again and lock the file) while the analysis needs the
    returned one to know which model produced a given code.
    """
    fields, error = parse_annotation(raw_output, allowed_answers)

    # The depth floor. Applied here rather than by skipping the annotation call,
    # deliberately: the annotator's ANSWER extraction is good (27/27 against an
    # unambiguous final line in the pilot) and there is no reason to replace it,
    # annotation costs ~$0.09 for the whole study so there is nothing to save,
    # and keeping the call means `raw_annotator_output` still records what the
    # annotator claimed. The override is therefore auditable: `depth_source`
    # says which of the two produced the code, and analysis can count them.
    depth_source = "annotator"
    if response_text is not None and not has_codeable_reasoning(response_text):
        if fields.get("depth") != "L0":
            depth_source = "no_reasoning_floor"
        fields = {**fields, "depth": "L0"}

    # ANSWER FLOOR, the mirror of the depth floor above. A response containing
    # no answer word at all cannot support KEEP or SWITCH, and both annotators
    # tried anyway: mistral-nemo AND deepseek-v3.2 both coded the response
    # "SING." as KEEP and "The man might have poisoned both bottles to kill
    # you." as REFUSE. That was measured through this pipeline, at temperature
    # 0, AFTER the annotator was swapped — a bake-off on raw JSON had suggested
    # deepseek handled these, and it did not reproduce here. Whether a string
    # contains an answer token is mechanical, so it is not left to a model.
    #
    # Scope: KEEP and SWITCH ONLY. Those are positive claims that a specific
    # choice was made, and with no answer token in the text the claim is simply
    # false. REFUSE and THIRD_OPTION are *legitimately* answer-word-free — a
    # genuine refusal names no bottle — so they stay with the annotator, which
    # is the only reader that can tell "I will not answer" from "I never got
    # round to answering". The pilot's two mislabelled REFUSEs are therefore
    # NOT caught by this floor; they are caught by
    # scripts/extraction_check.py, which flags them for a human.
    answer_source = "annotator"
    if response_text is not None and fields.get("answer") in ("KEEP", "SWITCH"):
        _, method = c.extract_answer(response_text)
        if method == "none":
            answer_source = "no_answer_floor"
            fields = {**fields, "answer": "UNPARSEABLE"}

    record = {
        "run_id": run_id,
        **fields,
        "depth_source": depth_source,
        "answer_source": answer_source,
        "annotator_model": annotator_model,
        "annotator_requested": annotator_requested or annotator_model,
        "raw_annotator_output": raw_output,
        "ts": ts or utc_now(),
        "annotator_provider": annotator_provider,
        "parse_ok": error is None,
        "parse_error": error,
    }
    if usage is not None:
        record["annotator_usage"] = usage
    return record


# ------------------------------------------------------------- the request ----

def annotation_request_body(model: str, prompt: str, *, json_mode: bool = False) -> dict:
    """The OpenRouter body for one annotation call.

    `temperature=0`: the same response must always get the same code, or the
    depth distribution picks up annotator noise on top of subject noise.
    `json_mode` is opt-in — `config.annotator` supports `response_format`
    (CONTRACTS.md §11) but a swapped-in annotator may not, and a 400 on every
    call is worse than the occasional fence.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": ANNOTATION_TEMPERATURE,
        "max_tokens": ANNOTATION_MAX_TOKENS,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def make_openrouter_caller(client, model: str, api_key: str, *,
                           json_mode: bool = False) -> Caller:
    """A `caller` that posts one annotation prompt and returns (status, json).

    The single network seam in this module; every test injects its own.
    """
    headers = {"Authorization": f"Bearer {api_key}"}

    async def caller(prompt: str) -> tuple[int, object]:
        response = await client.post(
            c.OPENROUTER_URL, headers=headers,
            json=annotation_request_body(model, prompt, json_mode=json_mode),
            timeout=120,
        )
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, None

    return caller


def _is_retryable(reason: str) -> bool:
    """Everything transient retries. A 400 or a 404 does not."""
    if reason.startswith("http_"):
        try:
            return int(reason[5:]) in RETRY_STATUS
        except ValueError:
            return False
    return True     # error bodies, missing choices, transport exceptions


# ------------------------------------------------------------------- stats ----

@dataclass
class Stats:
    input_records: int = 0
    bad_input_lines: int = 0
    by_mode: Counter = field(default_factory=Counter)
    cot: int = 0
    already_annotated: int = 0
    duplicate_inputs: int = 0
    skipped_no_run_id: int = 0
    beyond_limit: int = 0
    attempted: int = 0
    annotated: int = 0
    unparseable: int = 0        # answer == UNPARSEABLE, however it got there
    parse_failures: int = 0      # the subset where the reply was not valid JSON
    failed: int = 0
    aborted: int = 0             # never attempted: a fatal status was latched
    #: The first fatal status seen, latched rather than raised so that the
    #: in-flight siblings finish and their paid-for annotations reach disk.
    #: `annotate_all` re-raises it once, after the output handle is closed.
    fatal: str | None = None
    parse_errors: Counter = field(default_factory=Counter)
    failure_reasons: Counter = field(default_factory=Counter)
    answers: Counter = field(default_factory=Counter)
    depths: Counter = field(default_factory=Counter)
    annotator_models: Counter = field(default_factory=Counter)
    cost: float = 0.0

    def unparseable_rate(self) -> float:
        """Share of written annotations whose answer is UNPARSEABLE.

        PLAN.md §4 puts its 2% target on the *category*, not on invalid JSON:
        an annotator that returns well-formed `{"answer": "UNPARSEABLE"}` for
        every response has a 100% UNPARSEABLE rate, not 0%.
        """
        return self.unparseable / self.annotated if self.annotated else 0.0


# --------------------------------------------------------------- the run ----

async def annotate_all(records: Iterable[dict], *, caller: Caller,
                       out_path: str | Path, annotator: str,
                       allowed_answers: Sequence[str] | None = None,
                       concurrency: int = DEFAULT_CONCURRENCY,
                       limit: int | None = None,
                       sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                       clock: Callable[[], str] = utc_now,
                       json_mode: bool = False,
                       log: Callable[[str], None] | None = None,
                       progress_every: int = 200) -> Stats:
    """Annotate every `cot` record not already in `out_path`.

    Concurrency mirrors the runner: a semaphore of `concurrency`, writes
    serialised through an `asyncio.Lock` and flushed per record, so a kill
    mid-run costs nothing and every line stays independently parseable.
    """
    out_path = Path(out_path)
    records = list(records)
    stats = Stats(input_records=len(records))
    stats.by_mode.update(rec.get("mode", "<missing>") for rec in records)

    todo = cot_records(records)
    stats.cot = len(todo)

    done, existing_models = read_annotated_index(out_path)
    if existing_models and annotator not in existing_models:
        raise AnnotatorMismatch(
            f"{out_path} already holds annotations requested from "
            f"{sorted(existing_models)} but this run uses {annotator!r}. "
            "CONTRACTS.md §5: re-annotate into a NEW file — nothing downstream "
            "can tell a re-annotation from a double-write."
        )

    fresh, seen = [], set()
    for rec in todo:
        run_id = rec.get("run_id")
        if not run_id:
            stats.skipped_no_run_id += 1
            continue
        if run_id in done:
            stats.already_annotated += 1
            continue
        if run_id in seen:
            stats.duplicate_inputs += 1
            continue
        seen.add(run_id)
        fresh.append(rec)

    if limit is not None and len(fresh) > limit:
        stats.beyond_limit = len(fresh) - limit
        fresh = fresh[:limit]
    stats.attempted = len(fresh)
    if not fresh:
        return stats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()

    with out_path.open("a", encoding="utf-8") as handle:
        async def one(record: dict) -> None:
            """Never raises. `runner._run_one` wraps `_run_one_inner` for the
            same reason and says why: letting an exception escape into
            `asyncio.gather` closes the output handle while up to
            `concurrency - 1` other calls are still returning, so responses that
            have been paid for are dropped — and since a failed record writes no
            line, the next rerun reaches the same record and aborts in the same
            place forever."""
            try:
                await _one_inner(record)
            except FatalAnnotatorError as exc:
                # Latch it; do not propagate. The gather below finishes the
                # in-flight siblings, their annotations get written, and the
                # error is re-raised once from `annotate_all` — so the operator
                # still sees it and still stops, but nothing paid for is lost.
                if stats.fatal is None:
                    stats.fatal = str(exc)
                stats.failed += 1
                stats.failure_reasons["fatal_annotator_error"] += 1
            except Exception as exc:                   # never lose the handle
                stats.failed += 1
                stats.failure_reasons[f"internal_{type(exc).__name__}"] += 1

        async def _one_inner(record: dict) -> None:
            if stats.fatal is not None:
                # A fatal status has already been seen. Every remaining call
                # fails identically, so do not pay for them.
                stats.aborted += 1
                return
            prompt = build_annotation_prompt(response_text(record))
            data, reason = None, "unknown"
            async with semaphore:
                for attempt in range(MAX_ATTEMPTS):
                    try:
                        status, payload = await caller(prompt)
                    except Exception as exc:            # transport-level
                        status, payload = 0, None
                        reason = f"exception:{type(exc).__name__}"
                    else:
                        if status in FATAL_STATUS:
                            raise FatalAnnotatorError(
                                f"HTTP {status} from OpenRouter — check "
                                "OPENROUTER_API_KEY and model access.")
                        usable, reason = c.is_usable_response(status, payload)
                        if usable:
                            data = payload
                            break
                    if attempt == MAX_ATTEMPTS - 1 or not _is_retryable(reason):
                        break
                    await sleep(BACKOFF_BASE ** attempt)

            if data is None:
                # No line at all, exactly like the runner: a record here would
                # mark the run_id done and resume would never retry it.
                stats.failed += 1
                stats.failure_reasons[reason] += 1
                return

            raw = data["choices"][0]["message"]["content"]
            written = build_annotated_record(
                run_id=record["run_id"], raw_output=raw,
                response_text=response_text(record),
                annotator_model=data.get("model") or annotator,
                annotator_requested=annotator,
                annotator_provider=data.get("provider"),
                usage=data.get("usage"), ts=clock(),
                allowed_answers=allowed_answers,
            )
            async with lock:
                handle.write(json.dumps(written, ensure_ascii=False) + "\n")
                handle.flush()

            stats.annotated += 1
            stats.answers[written["answer"]] += 1
            stats.depths[written["depth"]] += 1
            stats.annotator_models[written["annotator_model"]] += 1
            if written["answer"] == "UNPARSEABLE":
                stats.unparseable += 1
            if not written["parse_ok"]:
                stats.parse_failures += 1
                stats.parse_errors[written["parse_error"]] += 1
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                stats.cost += float(usage["cost"])
            if log and progress_every and stats.annotated % progress_every == 0:
                log(f"  {stats.annotated}/{len(fresh)} annotated, "
                    f"${stats.cost:.4f} spent")

        # return_exceptions: belt and braces. `one` already swallows everything,
        # but a leak must never close the handle under tasks whose responses
        # have been paid for.
        await asyncio.gather(*(one(rec) for rec in fresh), return_exceptions=True)

    if stats.fatal is not None:
        # Raised only after the handle is closed and every in-flight annotation
        # is on disk. The CLI turns this into a non-zero exit; a rerun resumes
        # over the records that did land.
        raise FatalAnnotatorError(stats.fatal)
    return stats


def format_summary(stats: Stats) -> str:
    """PLAN.md §4: report the UNPARSEABLE count and rate, and warn above 2%."""
    rate = stats.unparseable_rate()
    out = [
        f"cot records in input : {stats.cot}",
        f"already annotated    : {stats.already_annotated}",
        f"annotated this run   : {stats.annotated}",
        f"UNPARSEABLE          : {stats.unparseable}  ({rate:.1%} of annotated)",
        f"  of which invalid JSON rather than an UNPARSEABLE coding: "
        f"{stats.parse_failures}",
    ]
    if stats.failed:
        out.append(f"failed calls         : {stats.failed} "
                   f"(no line written — rerun to retry) "
                   f"{dict(stats.failure_reasons)}")
    if stats.duplicate_inputs:
        out.append(f"duplicate run_ids in input, skipped: {stats.duplicate_inputs}")
    if stats.beyond_limit:
        out.append(f"left for a later run (--limit)     : {stats.beyond_limit}")
    if stats.skipped_no_run_id:
        out.append(f"cot records with no run_id, skipped: {stats.skipped_no_run_id}")
    if stats.cost:
        out.append(f"annotation spend     : ${stats.cost:.4f}")
    if stats.annotator_models:
        out.append(f"annotator(s) returned: {dict(stats.annotator_models)}")
    if stats.answers:
        out.append("answers : " + "  ".join(
            f"{label}={stats.answers.get(label, 0)}" for label in c.ANSWERS))
        out.append("depth   : " + "  ".join(
            f"{code}={stats.depths.get(code, 0)}" for code in c.DEPTH_CODES))
    if stats.parse_errors:
        out.append(f"parse failures: {dict(stats.parse_errors)}")
    if rate > UNPARSEABLE_WARN_RATE:
        out.append(
            f"WARNING: UNPARSEABLE rate {rate:.1%} is over PLAN.md §4's 2% target. "
            "Hand-review every instance before treating any of this as a result; "
            "an UNPARSEABLE is a bug report, not an answer.")
    return "\n".join(out)


# ------------------------------------------------- hand-coding export ----

def stratified_sample(records: Sequence[dict], n: int, seed: int) -> list[dict]:
    """`n` records spread as evenly as possible over (model × scenario).

    PLAN.md §8.4 wants agreement measured across the grid, not across whichever
    model happens to sit at the top of the file — one verbose model can
    otherwise own the whole sample. Strata are filled round-robin, so a stratum
    that runs dry hands its share to the others rather than shrinking the
    sample. Deterministic for a given seed.
    """
    strata: dict[tuple, list[dict]] = {}
    for record in records:
        key = (record.get("model"), record.get("scenario"))
        strata.setdefault(key, []).append(record)
    keys = sorted(strata, key=lambda k: tuple(str(part) for part in k))
    for key in keys:
        strata[key].sort(key=lambda rec: rec.get("run_id", ""))

    target = min(n, len(records))
    quota = {key: 0 for key in keys}
    allocated = 0
    while allocated < target:
        progressed = False
        for key in keys:
            if allocated == target:
                break
            if quota[key] < len(strata[key]):
                quota[key] += 1
                allocated += 1
                progressed = True
        if not progressed:
            break

    rng = random.Random(seed)
    chosen: list[dict] = []
    for key in keys:
        chosen.extend(rng.sample(strata[key], quota[key]))
    chosen.sort(key=lambda rec: rec.get("run_id", ""))
    return chosen


def export_for_handcoding(records: Iterable[dict], n: int, out_path: str | Path,
                          seed: int, restrict_ids: set[str] | None = None) -> list[dict]:
    """Write a stratified sample of CoT responses for hand-coding.

    Columns are `run_id`, `response_text`, then the four label columns left
    empty. `restrict_ids` narrows the pool to records that already have machine
    codes, so every row of the twenty minutes PLAN.md §8.4 budgets actually
    contributes to a kappa.
    """
    pool = cot_records(records)
    if restrict_ids is not None:
        pool = [rec for rec in pool if rec.get("run_id") in restrict_ids]
    chosen = stratified_sample(pool, n, seed)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    formula_rows = []
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(HANDCODE_COLUMNS)
        for record in chosen:
            text = response_text(record)
            if text[:1] in FORMULA_PREFIXES:
                formula_rows.append(record.get("run_id", ""))
            writer.writerow([record.get("run_id", ""), text, "", "", "", ""])
    if formula_rows:
        # Left byte-exact on purpose: prefixing a quote or a space would mean
        # hand-coding text the model did not emit. A named row is enough.
        warn(f"WARNING: {len(formula_rows)} response(s) start with a character a "
             f"spreadsheet evaluates as a formula, so the cell will not show the "
             f"text: {formula_rows[:5]}. Read those in the JSONL instead.")
    return chosen


# ------------------------------------------------------------------ kappa ----

def cohens_kappa(first: Sequence, second: Sequence) -> float:
    """Cohen's κ for two raters over the same items.

    `nan` when κ is undefined — no items, or both raters constant on the same
    single label, where chance agreement is already 1 and κ has no meaning.
    That is reported rather than rounded to 1.0, because "we always agreed
    because there was nothing to disagree about" is not reliability.
    """
    if len(first) != len(second):
        raise ValueError("kappa needs two equal-length label sequences")
    n = len(first)
    if n == 0:
        return math.nan
    observed = sum(1 for a, b in zip(first, second) if a == b) / n
    counts_a, counts_b = Counter(first), Counter(second)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n)
                   for label in set(counts_a) | set(counts_b))
    if math.isclose(expected, 1.0):
        return math.nan
    return (observed - expected) / (1.0 - expected)


def confusion_matrix(first: Sequence, second: Sequence,
                     labels: Sequence) -> dict:
    """Counts of (first, second) pairs, rows in `labels` order.

    Rows are the first rater (the human, by convention here), columns the
    second. Labels outside `labels` are appended rather than dropped.
    """
    seen = list(labels) + [label for label in dict.fromkeys(list(first) + list(second))
                           if label not in labels]
    matrix = {row: {col: 0 for col in seen} for row in seen}
    for a, b in zip(first, second):
        matrix[a][b] += 1
    return matrix


@dataclass
class KappaResult:
    n: int = 0
    unfilled: int = 0
    unjoined: list = field(default_factory=list)
    kappa: dict = field(default_factory=dict)
    observed: dict = field(default_factory=dict)
    confusion: dict = field(default_factory=dict)
    pairs: dict = field(default_factory=dict)
    annotator_models: Counter = field(default_factory=Counter)


def _hand_label(column: str, value: str, run_id: str):
    """One CSV cell as a label, or raise with enough context to fix the row."""
    if column == "answer":
        label = value.strip().upper()
        if label not in c.ANSWERS:
            raise HandCodingError(
                f"run_id {run_id}: answer {value!r} is not one of {list(c.ANSWERS)}")
        return label
    if column == "depth":
        label = value.strip().upper()
        if label not in c.DEPTH_CODES:
            raise HandCodingError(
                f"run_id {run_id}: depth {value!r} is not one of {list(c.DEPTH_CODES)}")
        return label
    flag = coerce_bool(value)
    if flag is None:
        raise HandCodingError(
            f"run_id {run_id}: {column} {value!r} is not a boolean "
            "(use TRUE/FALSE)")
    return flag


def kappa_report(csv_path: str | Path, annotated_path: str | Path) -> KappaResult:
    """Cohen's κ per field between a hand-coded CSV and `annotated.jsonl`.

    Joins on `run_id`, the only join key there is. Rows left entirely blank are
    counted as unfilled and skipped; a row filled in partially raises, because
    silently dropping half a judgement is worse than stopping.
    """
    csv_path, annotated_path = Path(csv_path), Path(annotated_path)
    machine_records, _ = read_jsonl(annotated_path)
    machine = {rec["run_id"]: rec for rec in machine_records if "run_id" in rec}

    result = KappaResult()
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        columns = _label_columns(csv.DictReader(handle).fieldnames)
    human_labels: dict[str, list] = {col: [] for col in columns}
    machine_labels: dict[str, list] = {col: [] for col in columns}

    # utf-8-sig, not utf-8: Excel's "CSV UTF-8" and Google Sheets both prepend a
    # byte-order mark, which would turn the first header into "\ufeffrun_id" and
    # fail the column check while the column is visibly there.
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [col for col in HANDCODE_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise HandCodingError(
                f"{csv_path} is missing column(s) {missing}; expected "
                f"{list(HANDCODE_COLUMNS)}")
        for row in reader:
            run_id = (row.get("run_id") or "").strip()
            if not run_id:
                continue
            cells = {col: (row.get(col) or "").strip() for col in columns}
            if not any(cells.values()):
                result.unfilled += 1
                continue
            # A blank REQUIRED column in an otherwise filled row is a
            # half-coded row and an error. A blank OPTIONAL column is not: the
            # export carries `flag_proposer` whether or not the hand-coder's
            # copy predates it, and refusing to score a 60-row CSV over one
            # unfilled boolean would throw away the whole hand-coding pass.
            blank = [col for col in LABEL_COLUMNS if not cells[col]]
            if blank:
                raise HandCodingError(
                    f"run_id {run_id}: column(s) {blank} left blank while others "
                    "are filled — complete the row or clear it entirely")
            coded = machine.get(run_id)
            if coded is None:
                result.unjoined.append(run_id)
                continue
            absent = [col for col in LABEL_COLUMNS if col not in coded]
            if absent:
                raise HandCodingError(
                    f"run_id {run_id}: annotated record is missing {absent}")
            for col in columns:
                # Optional column left blank, or absent from the annotation:
                # score neither side for this row rather than inventing a label.
                if col not in LABEL_COLUMNS and (not cells[col] or col not in coded):
                    continue
                human_labels[col].append(_hand_label(col, cells[col], run_id))
                value = coded[col]
                machine_labels[col].append(
                    value if col in ("answer", "depth") else bool(value))
            if coded.get("annotator_model"):
                result.annotator_models[coded["annotator_model"]] += 1
            result.n += 1

    for col in columns:
        result.kappa[col] = cohens_kappa(human_labels[col], machine_labels[col])
        pairs = list(zip(human_labels[col], machine_labels[col]))
        result.observed[col] = (
            sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else math.nan)
        result.pairs[col] = pairs
    result.confusion = confusion_matrix(
        human_labels["depth"], machine_labels["depth"], list(c.DEPTH_CODES))
    return result


def format_kappa_report(result: KappaResult) -> str:
    out = [f"joined on run_id: {result.n} rows"]
    if result.unfilled:
        out.append(f"unfilled rows skipped: {result.unfilled}")
    if result.unjoined:
        out.append(f"rows with no annotation to compare against: "
                   f"{len(result.unjoined)} {result.unjoined[:5]}")
    if result.annotator_models:
        out.append(f"machine codes from: {dict(result.annotator_models)}")
    out.append("")
    out.append(f"{'field':<16}{'kappa':>9}{'agreement':>12}")
    # Whatever this run actually scored, in canonical order: an optional column
    # is absent when the CSV predates it (see OPTIONAL_LABEL_COLUMNS).
    scored = [c for c in LABEL_COLUMNS + OPTIONAL_LABEL_COLUMNS if c in result.kappa]
    for col in scored:
        kappa, observed = result.kappa.get(col, math.nan), result.observed.get(col, math.nan)
        kappa_text = "  n/a" if math.isnan(kappa) else f"{kappa:.3f}"
        obs_text = "  n/a" if math.isnan(observed) else f"{observed:.1%}"
        out.append(f"{col:<16}{kappa_text:>9}{obs_text:>12}")
    if any(math.isnan(result.kappa.get(col, math.nan)) for col in LABEL_COLUMNS):
        out.append("  n/a: kappa undefined — both raters used a single label, so "
                   "chance agreement is already 1.")

    codes = list(result.confusion)
    out.append("")
    out.append("depth confusion matrix (rows = hand codes, columns = annotator)")
    out.append("      " + "".join(f"{code:>6}" for code in codes) + "     total")
    for row in codes:
        counts = result.confusion[row]
        out.append(f"{row:>5} " + "".join(f"{counts[col]:>6}" for col in codes)
                   + f"{sum(counts.values()):>10}")
    out.append("total " + "".join(
        f"{sum(result.confusion[row][col] for row in codes):>6}" for col in codes))

    depth_kappa = result.kappa.get("depth", math.nan)
    if not math.isnan(depth_kappa) and depth_kappa < DEPTH_KAPPA_FLOOR:
        out.append("")
        out.append(f"WARNING: depth kappa {depth_kappa:.3f} is under PLAN.md §8.4's "
                   f"{DEPTH_KAPPA_FLOOR} floor. Re-annotate with a mid-tier model for "
                   "this call only (5,625 short calls: even 5x is under $3), write it "
                   "to a NEW file per CONTRACTS.md §5, and compare again.")
    return "\n".join(out)


# -------------------------------------------------------------------- cli ----

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.annotate",
        description="Annotate cot responses (answer + depth) per PLAN.md §8.4.")
    parser.add_argument("--config", default=str(c.CONFIG_PATH),
                        help="config/budget.yaml (default: %(default)s)")
    parser.add_argument("--input", default=str(c.RAW_PATH),
                        help="raw JSONL to read (default: %(default)s)")
    parser.add_argument("--annotated", default=str(c.ANNOTATED_PATH),
                        help="annotated JSONL to append to and resume from "
                             "(default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None,
                        help="annotate at most N records this run")
    parser.add_argument("--annotator", default=None,
                        help="override config.annotator (write to a NEW "
                             "--annotated file if you do)")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="override config.max_concurrency")
    parser.add_argument("--json-mode", action="store_true",
                        help="send response_format=json_object (annotator must "
                             "support it)")
    parser.add_argument("--export-for-handcoding", type=int, default=None,
                        metavar="N", help="write N cot responses to --out as CSV")
    parser.add_argument("--out", default=None,
                        help="CSV path for --export-for-handcoding")
    parser.add_argument("--kappa", default=None, metavar="CSV",
                        help="score a filled hand-coding CSV against --annotated")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.config)
    input_path, annotated_path = Path(args.input), Path(args.annotated)

    if args.kappa:
        result = kappa_report(args.kappa, annotated_path)
        print(f"hand codes : {args.kappa}")
        print(f"machine    : {annotated_path}")
        print()
        print(format_kappa_report(result))
        return 0

    records, bad_lines = read_jsonl(input_path)
    if not records:
        warn(f"{input_path} has no records. Run the runner first, or pass "
             f"--input tests/fixtures/raw_sample.jsonl.")
        return 2
    if bad_lines:
        warn(f"WARNING: {len(bad_lines)} unparseable line(s) in {input_path} "
             f"at {bad_lines[:5]} — skipped.")

    if args.export_for_handcoding is not None:
        out_csv = Path(args.out or (c.DATA_DIR / "handcode.csv"))
        wanted = args.export_for_handcoding
        pool = cot_records(records)
        if wanted <= 0:
            warn(f"--export-for-handcoding {wanted}: nothing to export.")
            return 2
        if not pool:
            warn(f"{input_path} holds no cot records to hand-code "
                 f"(modes present: {dict(Counter(r.get('mode') for r in records))}).")
            return 2

        annotated_ids = read_annotated_ids(annotated_path)
        restrict = annotated_ids or None
        if restrict is None:
            print("NOTE: no annotations found yet, so the sample is drawn from "
                  "all cot records. --kappa needs machine codes to join "
                  "against; annotate first, or re-export afterwards.")
        elif not any(rec.get("run_id") in restrict for rec in pool):
            warn(f"WARNING: none of the cot records in {input_path} appear in "
                 f"{annotated_path}, so the sample is drawn from all of them. "
                 "--kappa will have nothing to join against until they are "
                 "annotated.")
            restrict = None

        chosen = export_for_handcoding(
            records, wanted, out_csv,
            seed=int(config.get("seed", 0)), restrict_ids=restrict)
        strata = Counter((rec.get("model"), rec.get("scenario")) for rec in chosen)
        print(f"wrote {len(chosen)} rows -> {out_csv}")
        if len(chosen) < wanted:
            print(f"(asked for {wanted}; the pool holds {len(pool)} cot records)")
        print(f"stratified over {len(strata)} (model × scenario) cells, "
              f"{min(strata.values())}-{max(strata.values())} per cell")
        print("Fill in answer, depth, flag_no_basis, flag_rigged, flag_proposer, then run:")
        print(f"  python -m src.annotate --kappa {out_csv}")
        return 0

    annotator = args.annotator or config["annotator"]
    concurrency = args.concurrency or int(config.get("max_concurrency",
                                                     DEFAULT_CONCURRENCY))
    print(f"input     : {input_path}")
    print(f"output    : {annotated_path}")
    print(f"annotator : {annotator}  (temperature {ANNOTATION_TEMPERATURE}"
          f"{', json mode' if args.json_mode else ''})")

    if input_path != c.RAW_PATH and annotated_path == c.ANNOTATED_PATH:
        warn(f"WARNING: reading {input_path} but writing the real "
             f"{c.ANNOTATED_PATH}. Those run_ids are not in {c.RAW_PATH}, so "
             "they would join to nothing downstream. Pass --annotated "
             "data/annotated_fixture.jsonl to keep the fixture separate.")

    # contracts.load_api_key, not os.environ: the message below promises a .env
    # fallback and this is the loader that actually provides one (shared with
    # runner.py, so both stages authenticate identically).
    api_key = c.load_api_key()
    if not api_key:
        warn(f"OPENROUTER_API_KEY is not set. Export it, or put it in "
             f"{c.ENV_PATH}, then re-run.")
        return 2

    import httpx        # imported here so --export/--kappa need no network stack

    async def go() -> Stats:
        async with httpx.AsyncClient() as client:
            caller = make_openrouter_caller(client, annotator, api_key,
                                            json_mode=args.json_mode)
            return await annotate_all(
                records, caller=caller, out_path=annotated_path,
                annotator=annotator,
                allowed_answers=config.get("allowed_answers"),
                concurrency=concurrency, limit=args.limit,
                json_mode=args.json_mode, log=print)

    try:
        stats = asyncio.run(go())
    except (AnnotatorMismatch, FatalAnnotatorError) as exc:
        warn(f"\n{exc}")
        return 1

    print()
    print(format_summary(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
