"""Async OpenRouter executor: manifest in, `data/raw.jsonl` out.

    python -m src.runner --config config/budget.yaml [--limit N] \
        [--mode direct|cot|contamination] [--models a,b] [--yes]

The runner is a **dumb executor** (CONTRACTS.md §3): it turns the fields of a
job dict into a request body and adds nothing of its own. Everything about
*what* to send — messages, temperature, max_tokens, logprobs, provider routing
— was decided by `src/manifest.py` and is already in the job.

Four things here are load-bearing, and three of them cost money or data if
skipped:

1. **`contracts.is_usable_response` before every append.** OpenRouter returns
   HTTP 200 with an `{"error": ...}` body and no `choices` when the upstream
   provider fails. `raise_for_status()` does not catch it. Write it as a record
   and that `run_id` looks done forever, so resume never retries it — a
   permanent silent hole in the grid, concentrated exactly where a provider was
   rate-limiting. A False is treated as a retryable failure and **no line is
   written** (CONTRACTS.md §4).
2. **Append-only, one line per call, flushed and fsynced under a lock.**
   Concurrent appends from 12 tasks interleave and corrupt lines otherwise, and
   a 5,000-call job will be interrupted at least once.
3. **Resume is by `run_id` only** — the single join key, and the only thing
   resume looks at. Note CONTRACTS.md §2's warning: the hash does not cover the
   prompt text, so move `data/raw.jsonl` aside if you change a scenario file or
   `allow_randomise_in_prompt`.
4. **`model_returned` and `provider` come from the RESPONSE, never the
   request.** OpenRouter routes to whichever provider is free and they differ in
   quantisation and in logprob values; recording only what you asked for
   silently mixes two models under one label.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

from src import contracts as c
from src import manifest as manifest_mod
from src import prices as prices_mod

MAX_ATTEMPTS = 5
BACKOFF_BASE = 2.0          # PLAN.md §8.2's 2**attempt: 1s, 2s, 4s, 8s
BACKOFF_JITTER = 0.25       # ±25%, so 12 concurrent 429s do not retry in lockstep
MAX_RETRY_AFTER = 60.0
REQUEST_TIMEOUT = 120.0
PROGRESS_EVERY = 50

#: Transient. Back off and try again.
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
#: The key or the credit is wrong. Every remaining job fails identically, so the
#: run aborts rather than burning 5,685 × 5 attempts to discover that.
#: **403 is deliberately not here.** OpenRouter's 403 is per-request — a
#: provider's moderation layer rejecting one prompt — and these scenarios are
#: about poisoning someone. One flagged prompt must cost one job, not the grid,
#: and since a failed job writes no line, a fatal 403 would abort every rerun at
#: the same place forever.
FATAL_STATUS = frozenset({401, 402})

FAILURES_PATH = c.DATA_DIR / "failures.jsonl"


# ------------------------------------------------------------------ utils ----

def utc_stamp() -> str:
    """CONTRACTS.md §4's `ts`: ISO-8601 UTC with a `Z` suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# `load_api_key` moved to `contracts.py` so that `annotate.py` gets the same
# behaviour (it used to read only os.environ while its error message promised
# a .env fallback). Re-exported under the original name: this module is what
# tests monkeypatch and what the CLI calls.
load_api_key = c.load_api_key


def request_body(job: dict) -> dict:
    """The HTTP body for one job. Every field comes straight off the job — the
    runner computes none of them (CONTRACTS.md §3)."""
    body = {
        "model": job["model"],
        "messages": job["messages"],
        "temperature": job["temperature"],
        "max_tokens": job["max_tokens"],
    }
    if job.get("logprobs"):
        body["logprobs"] = True
        if job.get("top_logprobs") is not None:
            body["top_logprobs"] = job["top_logprobs"]
    if job.get("provider_routing"):
        # The job field is `provider_routing`; the wire field is `provider`.
        # CONTRACTS.md §3 keeps them differently named so that the response's
        # `provider` cannot overwrite the request's routing in a raw record.
        body["provider"] = job["provider_routing"]
    return body


def make_record(job: dict, data: dict) -> dict:
    """The raw record: the job dict plus exactly the five fields in
    CONTRACTS.md §4. `provider` is the provider that actually served the call,
    never the routing that was requested."""
    record = dict(job)
    record["response"] = data
    record["model_returned"] = data.get("model")
    record["provider"] = data.get("provider")
    record["usage"] = data.get("usage")
    record["ts"] = utc_stamp()
    return record


def done_ids(raw_path: Path = c.RAW_PATH) -> set[str]:
    """`run_id`s already present in `raw.jsonl`. Resume skips exactly these."""
    ids, _ = scan_raw(raw_path)
    return ids


def scan_raw(raw_path: Path = c.RAW_PATH) -> tuple[set[str], dict]:
    """`(run_ids, diagnostics)`. Unparseable lines are reported, not repaired:
    a half-written final line is a symptom worth seeing, and silently rewriting
    someone's data file is worse than telling them about it."""
    ids: set[str] = set()
    stats = {"lines": 0, "unparseable": 0, "duplicate": 0, "no_run_id": 0}
    path = Path(raw_path)
    if not path.exists():
        return ids, stats
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            stats["lines"] += 1
            try:
                record = json.loads(line)
            except ValueError:
                stats["unparseable"] += 1
                continue
            run_id = record.get("run_id") if isinstance(record, dict) else None
            if not isinstance(run_id, str):
                stats["no_run_id"] += 1
                continue
            if run_id in ids:
                stats["duplicate"] += 1
            ids.add(run_id)
    return ids, stats


def backoff_delay(attempt: int, retry_after: str | None = None) -> float:
    """`BACKOFF_BASE ** attempt` with jitter, floored by any `Retry-After`.
    Set `BACKOFF_BASE = 0` to disable sleeping (tests do)."""
    if BACKOFF_BASE <= 0:
        return 0.0
    delay = float(BACKOFF_BASE) ** attempt
    delay *= 1.0 + random.random() * BACKOFF_JITTER
    if retry_after:
        try:
            delay = max(delay, min(float(retry_after), MAX_RETRY_AFTER))
        except ValueError:
            pass
    return delay


# ------------------------------------------------------------------- sink ----

class JsonlSink:
    """Append-only writer. One `write` per record, serialised by an
    `asyncio.Lock` and flushed to disk immediately, so a kill at any moment
    leaves a file of whole, parseable lines."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self._lock = asyncio.Lock()

    def __enter__(self) -> JsonlSink:
        self._handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    async def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        async with self._lock:
            # No await between here and the flush: a cancellation cannot land
            # mid-line, so the file never grows a torn record.
            self._handle.write(line + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())


# ------------------------------------------------------------------ state ----

def _as_int(value: object) -> int:
    """Best-effort int. A provider that returns `"201"` or `null` for a token
    count must cost this record's accounting, nothing more."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class RunState:
    def __init__(self, total: int, prices: dict, progress_every: int = PROGRESS_EVERY):
        self.total = total
        self.prices = prices or {}
        self.progress_every = progress_every
        self.started = time.monotonic()
        self.done = 0
        self.ok = 0
        self.failed = 0
        self.aborted = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.spend_reported = 0.0     # summed usage.cost, OpenRouter's own number
        self.spend_derived = 0.0      # tokens x live prices, as a cross-check
        self.n_costed = 0
        self.reasons: Counter[str] = Counter()
        self.quality: Counter[str] = Counter()
        self.stop = asyncio.Event()   # graceful: finish in flight, start nothing new
        self.fatal: str | None = None

    def account(self, job: dict, data: dict) -> None:
        """Accumulate tokens and spend. Every field is defended: a provider that
        stringifies `usage`, or omits it, must cost this record's accounting and
        nothing more."""
        usage = data.get("usage")
        if not isinstance(usage, dict):
            self.quality["usage_missing"] += 1
            return
        prompt = _as_int(usage.get("prompt_tokens"))
        completion = _as_int(usage.get("completion_tokens"))
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        cost = usage.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            self.spend_reported += float(cost)
            self.n_costed += 1
        pair = prices_mod.lookup(self.prices, job["model"]) if self.prices else None
        if pair:
            self.spend_derived += prompt * pair["prompt"] + completion * pair["completion"]

    def inspect(self, job: dict, data: dict) -> None:
        """Non-fatal quality checks on a record that is being written anyway.

        `is_usable_response` only asks whether there is content. It cannot see
        the two failures CONTRACTS.md says destroy the data: a CoT reply
        truncated at `max_tokens` (which eats the final answer line and reads as
        disobedience downstream, §11) and a direct-mode reply whose logprobs
        cannot yield a `{K, S}` pair (§8: "**Check it.**"). A paid response is
        never discarded over these — but the operator should see them while the
        run is still cheap to stop, not after annotation.
        """
        if not data.get("provider"):
            self.quality["no_provider_field"] += 1
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError):
            return
        if not isinstance(choice, dict):
            return
        if choice.get("finish_reason") == "length":
            self.quality["truncated_at_max_tokens"] += 1
        if job.get("logprobs"):
            _, status = c.answer_logprobs(choice)
            if status != "ok":
                self.quality[f"direct_unusable_{status}"] += 1

    def progress_line(self) -> str:
        elapsed = max(time.monotonic() - self.started, 1e-9)
        return (
            f"[{self.done:>6,}/{self.total:,}] ok {self.ok:,}  fail {self.failed:,}  "
            f"| spend ${self.spend_reported:.4f} reported "
            f"/ ${self.spend_derived:.4f} from prices  "
            f"| {elapsed:,.0f}s  {self.done / elapsed:.1f} calls/s"
        )

    def tick(self) -> None:
        self.done += 1
        if self.progress_every and self.done % self.progress_every == 0:
            print(self.progress_line(), flush=True)


# -------------------------------------------------------------- execution ----

async def _attempt_job(client: httpx.AsyncClient, job: dict, headers: dict,
                       state: RunState, timeout: float) -> tuple[dict | None, str]:
    """Up to `MAX_ATTEMPTS` tries. `(response_json, "ok")` or `(None, reason)`."""
    body = request_body(job)
    reason = "not_attempted"

    for attempt in range(MAX_ATTEMPTS):
        if state.fatal or state.stop.is_set():
            # Keep the reason from the attempts already made: a job that was
            # issued and 503'd before the stop must not be filed under "never
            # attempted", which reads as the opposite of what happened.
            return None, "aborted" if reason == "not_attempted" else f"aborted_after_{reason}"

        retry_after = None
        try:
            response = await client.post(c.OPENROUTER_URL, headers=headers,
                                         json=body, timeout=timeout)
        except Exception as exc:                      # timeouts, resets, DNS
            reason = f"transport_{type(exc).__name__}"
        else:
            try:
                data = response.json()
            except ValueError:
                data = None

            usable, why = c.is_usable_response(response.status_code, data)
            if usable:
                return data, "ok"

            reason = why
            if response.status_code in FATAL_STATUS:
                state.fatal = f"HTTP {response.status_code} — {_error_text(data)}"
                return None, reason
            if response.status_code >= 400 and response.status_code not in RETRY_STATUS:
                return None, reason                   # 400/404: retrying cannot help
            retry_after = response.headers.get("retry-after")
            # A 200 with an `error` body lands here and IS retried. That is the
            # whole point of CONTRACTS.md §4's warning.

        if attempt == MAX_ATTEMPTS - 1:
            break
        await asyncio.sleep(backoff_delay(attempt, retry_after))

    return None, reason


def _error_text(data: object) -> str:
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if error:
            return str(error)
    return "no error message in body"


async def _run_one(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, job: dict,
                   headers: dict, sink: JsonlSink, failures: JsonlSink | None,
                   state: RunState, timeout: float) -> None:
    """Run one job to completion. **Never raises.**

    An unexpected response shape — a provider that stringifies `usage`, a
    `choices` that is not a list — must cost one job, not the run. Letting it
    escape into `asyncio.gather` tears down the client and closes the sinks
    while up to `max_concurrency` other calls are still in flight: those
    responses are paid for and then dropped, no failure is recorded for
    anything, and the spend summary never prints.
    """
    try:
        await _run_one_inner(client, semaphore, job, headers, sink, failures, state, timeout)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        state.failed += 1
        state.reasons[f"internal_{type(exc).__name__}"] += 1
        if failures is not None:
            try:
                await failures.append(_failure_record(job, f"internal_{type(exc).__name__}: {exc}"))
            except Exception:
                pass
        state.tick()


def _failure_record(job: dict, reason: str) -> dict:
    return {"run_id": job["run_id"], "model": job["model"], "mode": job["mode"],
            "scenario": job["scenario"], "persona": job["persona"],
            "temperature": job["temperature"], "replicate": job["replicate"],
            "reason": reason, "ts": utc_stamp()}


async def _run_one_inner(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, job: dict,
                         headers: dict, sink: JsonlSink, failures: JsonlSink | None,
                         state: RunState, timeout: float) -> None:
    if state.fatal or state.stop.is_set():
        state.aborted += 1
        return
    async with semaphore:
        if state.fatal or state.stop.is_set():
            state.aborted += 1
            return
        data, reason = await _attempt_job(client, job, headers, state, timeout)

    if data is None:
        if reason == "aborted":
            state.aborted += 1        # never issued; nothing to diagnose
            return
        # No line in raw.jsonl: a failed call must stay retryable on resume
        # (CONTRACTS.md §4). The failure is logged elsewhere.
        state.failed += 1
        state.reasons[reason] += 1
        if failures is not None:
            await failures.append(_failure_record(job, reason))
        state.tick()
        return

    # Write first, account second: the record is the thing that must survive.
    await sink.append(make_record(job, data))
    state.ok += 1
    state.account(job, data)
    state.inspect(job, data)
    state.tick()


def _install_sigint(state: RunState) -> None:
    """First Ctrl-C stops scheduling and lets in-flight calls land; a second
    one gives up on them. Either way every line already written is intact."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:                                  # pragma: no cover
        return

    def handler() -> None:
        if state.stop.is_set():
            print("\nsecond interrupt — abandoning in-flight calls", flush=True)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        state.stop.set()
        print("\ninterrupted — no new calls; finishing what is in flight. "
              "Ctrl-C again to abandon them. Rerun to resume.", flush=True)

    try:
        loop.add_signal_handler(signal.SIGINT, handler)
    except (NotImplementedError, RuntimeError, ValueError):   # pragma: no cover
        pass


async def run_jobs(jobs: list[dict], *, api_key: str, max_concurrency: int,
                   raw_path: Path = c.RAW_PATH, failures_path: Path | None = FAILURES_PATH,
                   prices: dict | None = None, transport: httpx.BaseTransport | None = None,
                   timeout: float = REQUEST_TIMEOUT, progress_every: int = PROGRESS_EVERY,
                   handle_sigint: bool = True) -> RunState:
    """Execute `jobs`, appending one record per successful call to `raw_path`.

    Jobs whose `run_id` is already in `raw_path` are skipped, so this is safe to
    call on a partially complete file. `transport` is for tests: pass an
    `httpx.MockTransport` and no socket is opened.
    """
    already = done_ids(raw_path)
    pending = [job for job in jobs if job["run_id"] not in already]

    state = RunState(len(pending), prices or {}, progress_every)
    if not pending:
        return state
    if handle_sigint:
        _install_sigint(state)

    headers = {"Authorization": f"Bearer {api_key}"}
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))
    client_args = {"transport": transport} if transport is not None else {}

    with JsonlSink(raw_path) as sink:
        failures = None
        try:
            if failures_path is not None:
                failures = JsonlSink(failures_path).__enter__()
            async with httpx.AsyncClient(timeout=timeout, **client_args) as client:
                # return_exceptions: belt and braces. `_run_one` already swallows
                # everything, but a leak must never close the sinks under tasks
                # whose responses have been paid for.
                results = await asyncio.gather(*(
                    _run_one(client, semaphore, job, headers, sink, failures, state, timeout)
                    for job in pending
                ), return_exceptions=True)
            for outcome in results:
                if isinstance(outcome, BaseException):
                    state.reasons[f"leaked_{type(outcome).__name__}"] += 1
        finally:
            if failures is not None:
                failures.__exit__(None, None, None)
    return state


# -------------------------------------------------------------- filtering ----

def filter_jobs(jobs: list[dict], modes: list[str] | None = None,
                models: list[str] | None = None) -> list[dict]:
    """`--mode` and `--models`, order preserved.

    `None` means the flag was absent. An *empty* list means the flag was given
    with nothing usable in it, which is an error — `--models ""` from an unset
    shell variable must not quietly widen a one-model pilot into the whole
    5,685-job grid.

    A selector that matches nothing is an error too: a typo in a model id would
    otherwise look like "everything is already done". So is an ambiguous
    substring, because `--models instruct` matches two of the five configured
    ids and would silently double the spend.
    """
    selected = jobs
    if modes is not None:
        if not modes:
            raise ValueError("--mode was given but empty")
        unknown = set(modes) - set(c.ALL_MODES)
        if unknown:
            raise ValueError(f"unknown mode(s) {sorted(unknown)}; "
                             f"expected from {list(c.ALL_MODES)}")
        selected = [j for j in selected if j["mode"] in set(modes)]
        if not selected:
            raise ValueError(f"no jobs with mode in {modes}")
    if models is not None:
        if not models:
            raise ValueError("--models was given but empty")
        available = {j["model"] for j in selected}
        chosen: set[str] = set()
        for token in models:
            if token in available:
                chosen.add(token)
                continue
            hits = {m for m in available if token in m}
            if not hits:
                raise ValueError(f"--models {token!r} matched none of {sorted(available)}")
            if len(hits) > 1:
                raise ValueError(f"--models {token!r} is ambiguous — it matches "
                                 f"{sorted(hits)}. Give the full id.")
            chosen |= hits
        selected = [j for j in selected if j["model"] in chosen]
    return selected


#: Job fields that change what a call MEASURES while leaving its `run_id`
#: untouched. `run_id` covers six fields and none of these (CONTRACTS.md §2),
#: so a change to any of them makes an existing record and a pending job
#: incomparable while resume still treats them as the same cell.
DRIFT_FIELDS = ("messages", "provider_routing", "max_tokens")


def prompt_drift(jobs: list[dict], raw_path: Path, sample: int = 40) -> list[dict]:
    """Jobs whose `run_id` is already recorded but which would now be sent
    differently — see `DRIFT_FIELDS`.

    CONTRACTS.md §2: the hash covers six fields and *not* the prompt, so
    flipping `allow_randomise_in_prompt` or editing a scenario file produces
    identical `run_id`s over different text. Resume then skips all 5,685 jobs
    and reports the arm complete, while every record in the file was generated
    under the opposite wording — the study is silently invalid.

    **`provider_routing` is checked for the same reason and it is not
    hypothetical.** Turning on `pin_providers` left 34 of 70 existing cot
    records served by a provider the config now excludes, and this function —
    then comparing `messages` only — reported no drift. Provider is not
    cosmetic: on `llama-3.1-8b` the same prompt at the same temperature gave a
    median of 3 completion tokens on Novita against 158 on Groq, and
    CONTRACTS.md §8 measured `gemma-4-26b`'s P(KEEP) moving from 0.854 to 0.986
    across providers. Mixing two provider regimes under one model label is
    exactly what the pins exist to prevent.

    `max_tokens` is checked too: raising it (900 -> 2000, §13) changes whether
    verbose models reach their final line at all.

    The contract forbids fixing this by adding fields to the hash (three
    sessions depend on the formula). Detecting it costs one dict comparison per
    sampled line of a file the runner already reads in full.
    """
    path = Path(raw_path)
    if not path.exists():
        return []
    wanted = {job["run_id"]: job for job in jobs}
    drifted: list[dict] = []
    seen = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or seen >= sample:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            job = wanted.get(record.get("run_id"))
            if job is None or "messages" not in record:
                continue
            seen += 1
            if any(record.get(f) != job.get(f) for f in DRIFT_FIELDS):
                drifted.append(job)
    return drifted


# -------------------------------------------------------------------- CLI ----

def _summarise(state: RunState, skipped: int) -> str:
    lines = [
        "",
        f"done      {state.ok:,} written   {state.failed:,} failed   "
        f"{state.aborted:,} not attempted   {skipped:,} already present",
        f"tokens    {state.prompt_tokens:,} prompt   {state.completion_tokens:,} completion",
        f"spend     ${state.spend_reported:.4f} from usage.cost "
        f"({state.n_costed:,}/{state.ok:,} responses carried one)",
        f"          ${state.spend_derived:.4f} re-derived from live prices, as a cross-check",
    ]
    if state.reasons:
        top = ", ".join(f"{reason}×{n}" for reason, n in state.reasons.most_common(8))
        lines.append(f"failures  {top}")
        lines.append("          nothing was written for these; rerun to retry them")
    if state.quality:
        top = ", ".join(f"{name}×{n}" for name, n in state.quality.most_common(8))
        lines.append(f"quality   {top}")
        lines.append("          these records WERE written and paid for, but may not be "
                     "usable — see CONTRACTS.md §8 and §11")
    if state.aborted:
        lines.append(f"INCOMPLETE {state.aborted:,} job(s) never attempted. Rerun to resume.")
    if state.fatal:
        lines.append(f"ABORTED   {state.fatal}")
    return "\n".join(lines)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute the manifest against OpenRouter, appending to data/raw.jsonl. "
                    "Resumable: rerun after any interruption."
    )
    parser.add_argument("--config", type=Path, default=c.CONFIG_PATH)
    parser.add_argument("--limit", type=int, default=None,
                        help="work on the first N jobs of the selection (a stable window: "
                             "rerunning converges on exactly N records, then does nothing)")
    parser.add_argument("--mode", default=None,
                        help="comma-separated: direct, cot, contamination")
    parser.add_argument("--models", default=None,
                        help="comma-separated model ids or unique substrings")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="skip the confirmation prompt")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the estimate and exit without calling anything")
    parser.add_argument("--refresh-prices", action="store_true",
                        help="ignore the 24h price cache and refetch before estimating")
    parser.add_argument("--allow-prompt-drift", action="store_true",
                        help="proceed even though recorded prompts differ from the ones this "
                             "config builds (CONTRACTS.md §2). Almost always the wrong answer: "
                             "move data/raw.jsonl aside instead.")
    parser.add_argument("--raw", type=Path, default=c.RAW_PATH)
    parser.add_argument("--progress-every", type=int, default=PROGRESS_EVERY)
    args = parser.parse_args(argv)

    config = manifest_mod.load_config(args.config)
    jobs = manifest_mod.build_jobs(config)
    try:
        # `None` when the flag is absent; an empty list when it was given empty,
        # which filter_jobs rejects rather than treating as "no filter".
        jobs = filter_jobs(
            jobs,
            None if args.mode is None else [m.strip() for m in args.mode.split(",") if m.strip()],
            None if args.models is None else [m.strip() for m in args.models.split(",") if m.strip()],
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    already, scan = scan_raw(args.raw)
    if scan["unparseable"]:
        print(f"warning: {scan['unparseable']} unparseable line(s) in {args.raw}. They are "
              f"ignored for resume, but downstream stages will trip over them.", file=sys.stderr)
    if scan["duplicate"]:
        print(f"warning: {scan['duplicate']} duplicate run_id(s) already in {args.raw}.",
              file=sys.stderr)

    # `--limit N` selects the first N jobs of the *selection*, before resume,
    # not the first N still outstanding. That makes it a stable window: run
    # `--limit 30`, kill it, rerun, and you converge on exactly those 30 records
    # and then stop. Limiting after the resume filter would instead ask for 30
    # *more* calls on every rerun, so an interrupted pilot of 30 would grow to
    # 53 and then 83 — the subsample CONTRACTS.md §1 wants reproducible would be
    # different every invocation.
    if args.limit is not None:
        jobs = jobs[:max(0, args.limit)]
    pending = [job for job in jobs if job["run_id"] not in already]
    skipped = len(jobs) - len(pending)

    prices, available = {}, True
    try:
        prices = prices_mod.fetch_prices(force=args.refresh_prices)
    except Exception as exc:
        print(f"warning: no prices available ({type(exc).__name__}: {exc}); "
              f"the estimate below has no dollar figures", file=sys.stderr)
        available = False

    # The whole pre-flight goes to stderr. The gate below asks the operator to
    # authorise real spend, and `python -m src.runner | tee run.log` would
    # otherwise put the estimate in the file and leave the terminal blank with a
    # prompt they cannot see.
    report = manifest_mod.format_dry_run(
        pending, manifest_mod.estimate_cost(pending, prices), config, available)
    print(report, file=sys.stderr)
    fetched_at, age_hours = prices_mod.cache_status()
    if available and fetched_at is not None:
        print(f"  prices fetched {fetched_at:%Y-%m-%dT%H:%MZ} ({age_hours:.1f}h old)"
              + ("  !! STALE — a refetch failed" if age_hours > prices_mod.MAX_AGE_HOURS else ""),
              file=sys.stderr)
    print(f"\n{skipped:,} job(s) already in {args.raw} and will be skipped.", file=sys.stderr)
    print(f"{len(pending):,} call(s) to make.", file=sys.stderr)

    if args.dry_run:
        print("\nDry run only. No model was called.", file=sys.stderr)
        return 0
    if not pending:
        print("\nNothing to do.", file=sys.stderr)
        return 0

    drifted = prompt_drift(jobs, args.raw)
    if drifted:
        print(f"\n!! PROMPT DRIFT: {len(drifted)} sampled record(s) in {args.raw} have a "
              f"run_id this config also produces, but were generated from DIFFERENT prompt "
              f"text.\n   run_id covers six fields and not the prompt (CONTRACTS.md §2), so "
              f"resume cannot tell them apart and you would be mixing two wordings in one "
              f"file.\n   Move {args.raw} aside and start a fresh one, recording which config "
              f"produced which.\n   e.g. {drifted[0]['model']} / {drifted[0]['scenario']} / "
              f"{drifted[0]['persona']}", file=sys.stderr)
        if not args.allow_prompt_drift:
            print("   Refusing to continue. Pass --allow-prompt-drift if you really mean it.",
                  file=sys.stderr)
            return 2

    api_key = load_api_key()
    if not api_key:
        print("\nerror: OPENROUTER_API_KEY is not set and no .env holds one. "
              "Export it, or `cp .env.example .env` and fill it in.", file=sys.stderr)
        return 2

    if not args.yes:
        # Both streams: stdin so an answer can arrive, stderr so the estimate
        # the answer is about was actually on screen.
        if not (sys.stdin.isatty() and sys.stderr.isatty()):
            print("\nerror: refusing to spend money without a confirmation, and this is not "
                  "an interactive terminal. Rerun with --yes if that is what you want.",
                  file=sys.stderr)
            return 2
        try:
            sys.stderr.write("\nProceed? [y/N] ")
            sys.stderr.flush()
            answer = (sys.stdin.readline() or "").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer not in ("y", "yes"):
            print("aborted.", file=sys.stderr)
            return 1

    state = asyncio.run(run_jobs(
        pending,
        api_key=api_key,
        max_concurrency=int(config["max_concurrency"]),
        raw_path=args.raw,
        prices=prices,
        progress_every=args.progress_every,
    ))
    print(_summarise(state, skipped))
    if state.fatal:
        return 2
    # An interrupted run is not a successful one: a wrapper like
    # `python -m src.runner --yes && python -m src.annotate` must not read a
    # half-filled grid as complete.
    return 0 if (state.failed == 0 and state.aborted == 0) else 1


if __name__ == "__main__":
    raise SystemExit(_main())
