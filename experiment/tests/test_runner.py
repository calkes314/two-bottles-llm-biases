"""Tests for src/runner.py. No network: every call goes through an
`httpx.MockTransport`.

The three the brief asks for — resume skips completed `run_id`s, a 429 is
retried, concurrent writes produce parseable JSONL — plus the one that costs
real data if it regresses: an HTTP 200 carrying an `error` body must never be
written, because writing it marks that `run_id` done and resume never retries
it (CONTRACTS.md §4).

    python -m pytest tests/ -q
"""

import asyncio
import json

import httpx
import pytest

from src import contracts as c
from src import manifest as m
from src import runner as r


@pytest.fixture(autouse=True)
def no_sleeping(monkeypatch):
    """Retries are instant in tests."""
    monkeypatch.setattr(r, "BACKOFF_BASE", 0.0)


@pytest.fixture
def config():
    return m.load_config(c.CONFIG_PATH)


@pytest.fixture
def cot_jobs(config):
    return [job for job in m.build_jobs(config) if job["mode"] == "cot"][:6]


@pytest.fixture
def raw(tmp_path):
    return tmp_path / "raw.jsonl"


@pytest.fixture
def failures(tmp_path):
    return tmp_path / "failures.jsonl"


def ok_body(model="served/model", provider="Novita", content="KEEP", cost=2.5e-06):
    return {
        "id": "gen-test", "object": "chat.completion", "created": 1,
        "model": model, "provider": provider,
        "choices": [{"index": 0, "finish_reason": "stop", "logprobs": None,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 201, "completion_tokens": 87,
                  "total_tokens": 288, "cost": cost},
    }


def run(jobs, handler, raw, failures, concurrency=4, prices=None):
    return asyncio.run(r.run_jobs(
        jobs, api_key="test-key", max_concurrency=concurrency,
        raw_path=raw, failures_path=failures, prices=prices or {},
        transport=httpx.MockTransport(handler), progress_every=0,
        handle_sigint=False,
    ))


def lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ----------------------------------------------------------------- resume ----

def test_resume_skips_completed_run_ids(cot_jobs, raw, failures):
    """Pre-seed two records; only the remaining four may be called."""
    with raw.open("w", encoding="utf-8") as handle:
        for job in cot_jobs[:2]:
            handle.write(json.dumps(r.make_record(job, ok_body())) + "\n")

    calls = []

    def handler(request):
        calls.append(json.loads(request.content)["messages"])
        return httpx.Response(200, json=ok_body())

    state = run(cot_jobs, handler, raw, failures)

    assert len(calls) == 4
    assert state.ok == 4
    written = lines(raw)
    assert len(written) == 6
    ids = [record["run_id"] for record in written]
    assert len(set(ids)) == 6
    assert set(ids) == {job["run_id"] for job in cot_jobs}


def test_rerunning_a_finished_set_calls_nothing(cot_jobs, raw, failures):
    def handler(request):
        return httpx.Response(200, json=ok_body())

    run(cot_jobs, handler, raw, failures)
    calls = []

    def counting(request):
        calls.append(1)
        return httpx.Response(200, json=ok_body())

    state = run(cot_jobs, counting, raw, failures)
    assert calls == []
    assert state.total == 0
    assert len(lines(raw)) == len(cot_jobs)


def test_done_ids_tolerates_a_torn_line(cot_jobs, raw):
    with raw.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(r.make_record(cot_jobs[0], ok_body())) + "\n")
        handle.write('{"run_id": "half-written-')
    ids, stats = r.scan_raw(raw)
    assert ids == {cot_jobs[0]["run_id"]}
    assert stats["unparseable"] == 1


def test_scan_raw_reports_duplicates(cot_jobs, raw):
    record = r.make_record(cot_jobs[0], ok_body())
    raw.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
    _, stats = r.scan_raw(raw)
    assert stats["duplicate"] == 1


# ---------------------------------------------------------------- retries ----

def test_a_429_triggers_a_retry(cot_jobs, raw, failures):
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "slow down", "code": 429}})
        return httpx.Response(200, json=ok_body())

    state = run(cot_jobs[:1], handler, raw, failures, concurrency=1)

    assert len(calls) == 2, "the 429 must be retried, not written or dropped"
    assert state.ok == 1
    assert len(lines(raw)) == 1


def test_retries_are_capped_at_five_attempts(cot_jobs, raw, failures):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, json={"error": {"message": "upstream down"}})

    state = run(cot_jobs[:1], handler, raw, failures, concurrency=1)

    assert len(calls) == r.MAX_ATTEMPTS == 5
    assert state.ok == 0 and state.failed == 1
    assert lines(raw) == [], "an exhausted call must write nothing"
    assert lines(failures)[0]["run_id"] == cot_jobs[0]["run_id"]


def test_a_transport_error_is_retried_then_gives_up(cot_jobs, raw, failures):
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectTimeout("no route")

    state = run(cot_jobs[:1], handler, raw, failures, concurrency=1)
    assert len(calls) == r.MAX_ATTEMPTS
    assert state.failed == 1 and lines(raw) == []
    assert "transport_" in next(iter(state.reasons))


def test_a_400_is_not_retried(cot_jobs, raw, failures):
    """Retrying a malformed request five times is five times the latency for
    the same answer."""
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "bad model id"}})

    state = run(cot_jobs[:1], handler, raw, failures, concurrency=1)
    assert len(calls) == 1
    assert state.failed == 1 and lines(raw) == []


def test_a_401_aborts_the_whole_run(cot_jobs, raw, failures):
    """A bad key fails every job identically; burning 5,625 × 5 attempts to
    discover that is worse than stopping."""
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "no credentials"}})

    state = run(cot_jobs, handler, raw, failures, concurrency=1)
    assert len(calls) == 1
    assert state.fatal is not None and "401" in state.fatal
    assert state.aborted == len(cot_jobs) - 1
    assert lines(raw) == []


# ------------------------------------------- the HTTP 200 error-body trap ----

def test_http_200_with_an_error_body_is_retried_and_never_written(cot_jobs, raw, failures):
    """CONTRACTS.md §4: OpenRouter returns 200 with an `error` object and no
    `choices` when the upstream provider fails. `raise_for_status()` misses it.
    Writing it marks the run_id done forever."""
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json={"id": "gen-1", "error": {
            "message": "Provider returned error", "code": 429}})

    state = run(cot_jobs[:1], handler, raw, failures, concurrency=1)

    assert len(calls) == r.MAX_ATTEMPTS, "a 200-with-error must be treated as retryable"
    assert lines(raw) == []
    assert state.reasons["error_body_429"] == 1


def test_a_poisoned_run_id_stays_retryable_on_the_next_run(cot_jobs, raw, failures):
    def broken(request):
        return httpx.Response(200, json={"error": {"message": "provider down", "code": 502}})

    run(cot_jobs[:1], broken, raw, failures, concurrency=1)
    assert lines(raw) == []

    def fixed(request):
        return httpx.Response(200, json=ok_body())

    state = run(cot_jobs[:1], fixed, raw, failures, concurrency=1)
    assert state.ok == 1
    assert [record["run_id"] for record in lines(raw)] == [cot_jobs[0]["run_id"]]


def test_a_null_message_content_is_not_written(cot_jobs, raw, failures):
    def handler(request):
        body = ok_body()
        body["choices"][0]["message"]["content"] = None
        return httpx.Response(200, json=body)

    state = run(cot_jobs[:1], handler, raw, failures, concurrency=1)
    assert lines(raw) == []
    assert state.reasons["no_message_content"] == 1


# ------------------------------------------------------- concurrent writes ----

def test_concurrent_writes_produce_valid_jsonl(config, raw, failures):
    """Twelve tasks appending at once must not interleave into torn lines."""
    jobs = [job for job in m.build_jobs(config) if job["mode"] == "cot"][:120]

    async def handler(request):
        await asyncio.sleep(0)          # force a real interleave point
        payload = json.loads(request.content)
        return httpx.Response(200, json=ok_body(content=payload["messages"][-1]["content"][:40]))

    state = run(jobs, handler, raw, failures, concurrency=12)

    text = raw.read_text(encoding="utf-8")
    assert text.endswith("\n")
    raw_lines = [line for line in text.splitlines() if line.strip()]
    assert len(raw_lines) == len(jobs) == state.ok

    records = [json.loads(line) for line in raw_lines]      # every line must parse
    ids = [record["run_id"] for record in records]
    assert len(set(ids)) == len(jobs)
    assert set(ids) == {job["run_id"] for job in jobs}
    for record in records:
        assert record["response"]["choices"][0]["message"]["content"]


# ------------------------------------------- hostile and malformed replies ----

@pytest.mark.parametrize("mangle", [
    lambda body: body.update(usage="n/a"),                      # stringified usage
    lambda body: body.update(choices=3),                        # choices not a list
    lambda body: body.update(usage={"prompt_tokens": "201", "cost": "free"}),
    lambda body: body.update(usage=None),
])
def test_one_malformed_reply_costs_one_job_not_the_run(config, raw, failures, mangle):
    """Before the guard, a single stringified `usage` raised out of
    `asyncio.gather`, which closed the sinks under every in-flight task: 12
    paid-for responses lost, no failures logged, and no spend summary."""
    jobs = [job for job in m.build_jobs(config) if job["mode"] == "cot"][:20]
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] > 4:
            await asyncio.sleep(0.01)        # keep later calls in flight
        body = ok_body()
        if calls["n"] == 3:
            mangle(body)
        return httpx.Response(200, json=body)

    state = run(jobs, handler, raw, failures, concurrency=8)

    assert calls["n"] == len(jobs), "every job must still be attempted"
    assert state.ok + state.failed == len(jobs), "every job must be accounted for"
    assert len(lines(raw)) == state.ok
    assert state.ok >= len(jobs) - 1, "at most the one bad reply may be lost"


def test_a_403_is_not_fatal(cot_jobs, raw, failures):
    """OpenRouter's 403 is per-request moderation, and these scenarios are
    about poisoning someone. One flagged prompt must cost one job — a fatal 403
    would abort every rerun at the same place forever, since the failing job
    writes no line."""
    flagged = cot_jobs[0]["messages"][-1]["content"] + "​"   # one distinguishable prompt
    victim = dict(cot_jobs[0])
    victim["messages"] = [{"role": "user", "content": flagged}]
    jobs = [victim] + cot_jobs[1:]

    def handler(request):
        if json.loads(request.content)["messages"][-1]["content"] == flagged:
            return httpx.Response(403, json={"error": {"message": "flagged by moderation"}})
        return httpx.Response(200, json=ok_body())

    state = run(jobs, handler, raw, failures, concurrency=1)
    assert state.fatal is None, "one moderation rejection must not abort the grid"
    assert state.ok == len(jobs) - 1, "every other job must still run"
    assert state.failed == 1 and state.aborted == 0
    assert lines(failures)[0]["reason"] == "http_403"


def test_quality_counters_flag_records_that_are_written_but_unusable(config, raw, failures):
    """`is_usable_response` only asks whether there is content. Truncation and
    a direct-mode reply with no {K,S} logprobs both pass it, and both destroy
    the data (CONTRACTS.md §8, §11). The record is still written — it was paid
    for — but the operator must see it."""
    direct = next(job for job in m.build_jobs(config) if job["mode"] == "direct")

    def handler(request):
        body = ok_body(content="K")
        body["choices"][0]["finish_reason"] = "length"
        body["choices"][0]["logprobs"] = {"content": [
            {"token": "K", "logprob": -0.1, "top_logprobs": [{"token": "K", "logprob": -0.1}]}]}
        return httpx.Response(200, json=body)

    state = run([direct], handler, raw, failures, concurrency=1)
    assert state.ok == 1 and len(lines(raw)) == 1      # written, never discarded
    assert state.quality["truncated_at_max_tokens"] == 1
    assert state.quality["direct_unusable_no_answer_tokens"] == 1


def test_aborted_jobs_keep_the_reason_they_failed_for(cot_jobs, raw, failures):
    """A job that was issued and 503'd before a stop must not be filed under
    "never attempted" — that reads as the opposite of what happened."""
    state_box = {}

    def handler(request):
        state_box["state"].stop.set()      # stop after the first response
        return httpx.Response(503, json={"error": {"message": "down"}})

    async def go():
        st = r.RunState(len(cot_jobs), {}, 0)
        state_box["state"] = st
        return st

    # Drive it through run_jobs but trip `stop` from inside the handler.
    original = r.RunState

    def spy(*args, **kwargs):
        st = original(*args, **kwargs)
        state_box["state"] = st
        return st

    r.RunState = spy
    try:
        state = run(cot_jobs, handler, raw, failures, concurrency=1)
    finally:
        r.RunState = original

    assert lines(raw) == []
    assert any("aborted_after_" in reason for reason in state.reasons), state.reasons


# ------------------------------------------------------- concurrency bound ----

def test_the_semaphore_actually_bounds_in_flight_requests(config, raw, failures):
    """Nothing else observes concurrency, so the semaphore — the only thing
    keeping 5,685 jobs from hitting OpenRouter at once — would otherwise be
    free to regress silently."""
    jobs = [job for job in m.build_jobs(config) if job["mode"] == "cot"][:50]
    live = {"now": 0, "peak": 0}

    async def handler(request):
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        for _ in range(3):
            await asyncio.sleep(0)
        live["now"] -= 1
        return httpx.Response(200, json=ok_body())

    run(jobs, handler, raw, failures, concurrency=4)
    assert live["peak"] <= 4, f"semaphore breached: {live['peak']} in flight"
    assert live["peak"] > 1, "everything serialised — the semaphore is not the bound"


def test_the_write_lock_survives_an_interleaving_sink(config, raw, failures, monkeypatch):
    """The sink's lock is invisible while `append` has no await inside it. Split
    the write in two so a missing lock actually tears a line."""
    jobs = [job for job in m.build_jobs(config) if job["mode"] == "cot"][:40]

    async def chunked_append(self, record):
        line = json.dumps(record, ensure_ascii=False)
        half = len(line) // 2
        async with self._lock:
            self._handle.write(line[:half])
            await asyncio.sleep(0)          # a torn line lands here without the lock
            self._handle.write(line[half:] + "\n")
            self._handle.flush()

    monkeypatch.setattr(r.JsonlSink, "append", chunked_append)

    async def handler(request):
        await asyncio.sleep(0)
        return httpx.Response(200, json=ok_body())

    run(jobs, handler, raw, failures, concurrency=12)
    records = lines(raw)                      # raises if any line is torn
    assert len(records) == len(jobs)
    assert len({record["run_id"] for record in records}) == len(jobs)


# ------------------------------------------------------------- prompt drift ----

def test_prompt_drift_is_detected(cot_jobs, raw):
    """CONTRACTS.md §2: the hash covers six fields and not the prompt. Editing a
    scenario or flipping `allow_randomise_in_prompt` yields identical run_ids
    over different text, so resume reports the arm complete while every record
    came from the opposite wording."""
    stale = r.make_record(cot_jobs[0], ok_body())
    stale["messages"] = [{"role": "user", "content": "an older wording entirely"}]
    raw.write_text(json.dumps(stale) + "\n", encoding="utf-8")

    drifted = r.prompt_drift(cot_jobs, raw)
    assert [job["run_id"] for job in drifted] == [cot_jobs[0]["run_id"]]


def test_no_drift_reported_for_matching_records(cot_jobs, raw):
    raw.write_text("\n".join(json.dumps(r.make_record(job, ok_body()))
                             for job in cot_jobs) + "\n", encoding="utf-8")
    assert r.prompt_drift(cot_jobs, raw) == []


def test_the_randomise_toggle_is_exactly_the_hazard_the_hash_cannot_see(config):
    """Executable proof of the collision CONTRACTS.md §2 warns about."""
    flipped = json.loads(json.dumps(config))
    flipped["allow_randomise_in_prompt"] = True
    base = {job["run_id"]: job for job in m.build_jobs(config)}
    other = {job["run_id"]: job for job in m.build_jobs(flipped)}
    assert set(base) == set(other), "run_ids are identical across the toggle"
    cot = next(rid for rid, job in base.items() if job["mode"] == "cot")
    assert base[cot]["messages"] != other[cot]["messages"], "...but the prompts differ"


# ------------------------------------------------------------ record shape ----

def test_record_is_the_job_plus_exactly_five_fields(cot_jobs, raw, failures):
    def handler(request):
        return httpx.Response(200, json=ok_body())

    run(cot_jobs[:1], handler, raw, failures, concurrency=1)
    record = lines(raw)[0]
    assert set(record) - set(cot_jobs[0]) == {
        "response", "model_returned", "provider", "usage", "ts"}
    assert all(record[field] == cot_jobs[0][field] for field in m.JOB_FIELDS)
    assert record["ts"].endswith("Z") and "T" in record["ts"]


def test_model_returned_and_provider_come_from_the_response(config, raw, failures):
    """OpenRouter may serve a different quantisation than you asked for; the
    record must say which."""
    direct = next(job for job in m.build_jobs(config) if job["mode"] == "direct")

    def handler(request):
        return httpx.Response(200, json=ok_body(model="somewhere/else", provider="Azure"))

    run([direct], handler, raw, failures, concurrency=1)
    record = lines(raw)[0]
    assert record["model"] == direct["model"]                # what was requested
    assert record["model_returned"] == "somewhere/else"      # what answered
    assert record["provider"] == "Azure"
    assert record["provider_routing"] == direct["provider_routing"]   # not overwritten
    assert record["usage"]["prompt_tokens"] == 201


def test_spend_is_accumulated_from_usage_and_from_prices(cot_jobs, raw, failures):
    def handler(request):
        return httpx.Response(200, json=ok_body(cost=1e-05))

    prices = {job["model"]: {"prompt": 1e-7, "completion": 2e-7} for job in cot_jobs}
    state = run(cot_jobs[:3], handler, raw, failures, concurrency=1, prices=prices)

    assert state.spend_reported == pytest.approx(3e-05)
    assert state.spend_derived == pytest.approx(3 * (201 * 1e-7 + 87 * 2e-7))
    assert state.prompt_tokens == 603 and state.completion_tokens == 261


# ----------------------------------------------------------- request body ----

def test_direct_request_body_carries_the_pin_and_the_logprobs(config):
    direct = next(job for job in m.build_jobs(config) if job["mode"] == "direct")
    body = r.request_body(direct)
    assert body["temperature"] == 1.0
    assert body["max_tokens"] == c.DIRECT_MAX_TOKENS
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 10
    # The wire field is `provider`; the job field is `provider_routing`.
    assert body["provider"] == c.DIRECT_PROVIDER_PIN[direct["model"]]
    assert body["provider"]["allow_fallbacks"] is False
    assert "provider_routing" not in body


def test_cot_request_body_omits_logprobs_but_keeps_routing(cot_jobs):
    """cot needs no logprobs, but it IS provider-pinned now: unpinned cot
    scattered one model's replicates across four providers."""
    body = r.request_body(cot_jobs[0])
    assert set(body) == {"model", "messages", "temperature", "max_tokens", "provider"}
    assert body["max_tokens"] == c.COT_MAX_TOKENS
    assert "logprobs" not in body and "top_logprobs" not in body
    assert body["provider"] == cot_jobs[0]["provider_routing"]


def test_the_body_actually_sent_matches_request_body(config, raw, failures):
    direct = next(job for job in m.build_jobs(config) if job["mode"] == "direct")
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    run([direct], handler, raw, failures, concurrency=1)
    assert sent == [r.request_body(direct)]


# --------------------------------------------------------------- selectors ----

def test_filter_jobs_by_mode_and_model(config):
    jobs = m.build_jobs(config)
    cot = r.filter_jobs(jobs, modes=["cot"])
    assert {job["mode"] for job in cot} == {"cot"}

    target = config["models"][0]["id"]
    picked = r.filter_jobs(jobs, models=[target])
    assert {job["model"] for job in picked} == {target}

    by_substring = r.filter_jobs(jobs, models=[target.split("/")[1]])
    assert {job["model"] for job in by_substring} == {target}


def test_filter_jobs_rejects_a_selector_that_matches_nothing(config):
    jobs = m.build_jobs(config)
    with pytest.raises(ValueError, match="matched none"):
        r.filter_jobs(jobs, models=["typo/not-a-model"])
    with pytest.raises(ValueError, match="unknown mode"):
        r.filter_jobs(jobs, modes=["chain_of_thought"])


def test_filter_jobs_rejects_an_ambiguous_substring(config):
    """`--models instruct` matches two of the five configured ids. Silently
    unioning them doubles the spend with no warning."""
    jobs = m.build_jobs(config)
    with pytest.raises(ValueError, match="ambiguous"):
        r.filter_jobs(jobs, models=["instruct"])


def test_an_empty_selector_is_an_error_not_a_wildcard(config):
    """`--models ""` from an unset shell variable must not widen a one-model
    pilot into the whole 5,685-job grid."""
    jobs = m.build_jobs(config)
    with pytest.raises(ValueError, match="empty"):
        r.filter_jobs(jobs, models=[])
    with pytest.raises(ValueError, match="empty"):
        r.filter_jobs(jobs, modes=[])
    assert r.filter_jobs(jobs, models=None, modes=None) == jobs      # absent = no filter


# --------------------------------------------------------- the money gate ----

@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Run `_main` with `run_jobs` stubbed, recording what it was handed."""
    seen = {}

    def fake_run_jobs(jobs, **kwargs):
        seen["jobs"] = jobs
        seen["kwargs"] = kwargs
        state = r.RunState(len(jobs), {}, 0)
        state.ok = len(jobs)
        return state

    monkeypatch.setattr(r.asyncio, "run", lambda coro: coro)
    monkeypatch.setattr(r, "run_jobs", fake_run_jobs)
    monkeypatch.setattr(r, "load_api_key", lambda *a, **k: "test-key")
    monkeypatch.setattr(r.prices_mod, "fetch_prices", lambda **k: {})
    monkeypatch.setattr(r.prices_mod, "cache_status", lambda **k: (None, None))
    seen["raw"] = tmp_path / "raw.jsonl"
    return seen


def _argv(cli, *extra):
    return ["--raw", str(cli["raw"]), "--progress-every", "0", *extra]


def test_a_non_tty_never_spends_without_yes(cli, monkeypatch):
    monkeypatch.setattr(r.sys.stdin, "isatty", lambda: False)
    assert r._main(_argv(cli, "--limit", "2")) == 2
    assert "jobs" not in cli, "no confirmation, no calls"


def test_yes_bypasses_the_prompt(cli, monkeypatch):
    monkeypatch.setattr(r.sys.stdin, "isatty", lambda: False)
    assert r._main(_argv(cli, "--limit", "2", "--yes")) == 0
    assert len(cli["jobs"]) == 2


def test_answering_no_spends_nothing(cli, monkeypatch):
    monkeypatch.setattr(r.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(r.sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(r.sys.stdin, "readline", lambda: "n\n")
    assert r._main(_argv(cli, "--limit", "2")) == 1
    assert "jobs" not in cli


def test_a_bare_enter_spends_nothing(cli, monkeypatch):
    """[y/N] means N. An empty line must not be read as consent."""
    monkeypatch.setattr(r.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(r.sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(r.sys.stdin, "readline", lambda: "\n")
    assert r._main(_argv(cli, "--limit", "2")) == 1
    assert "jobs" not in cli


def test_dry_run_never_calls_anything(cli):
    assert r._main(_argv(cli, "--dry-run")) == 0
    assert "jobs" not in cli


def test_limit_hands_over_exactly_the_first_n_jobs(cli, config):
    assert r._main(_argv(cli, "--limit", "7", "--mode", "cot", "--yes")) == 0
    expected = [job for job in m.build_jobs(config) if job["mode"] == "cot"][:7]
    assert [job["run_id"] for job in cli["jobs"]] == [job["run_id"] for job in expected]


def test_limit_is_a_stable_window_that_converges(cli, config):
    """`--limit 30`, killed and rerun, must converge on exactly 30 records.
    Applying the limit after the resume filter would ask for 30 *more* calls
    each time: 30, then 53, then 83."""
    window = [job for job in m.build_jobs(config) if job["mode"] == "cot"][:10]
    # Four of the ten already recorded.
    cli["raw"].write_text("\n".join(json.dumps(r.make_record(job, ok_body()))
                                    for job in window[:4]) + "\n", encoding="utf-8")

    assert r._main(_argv(cli, "--limit", "10", "--mode", "cot", "--yes")) == 0
    handed = [job["run_id"] for job in cli["jobs"]]
    assert handed == [job["run_id"] for job in window[4:]], "only the outstanding 6"
    assert len(handed) == 6

    # With all ten present, a further rerun must do nothing at all.
    cli["raw"].write_text("\n".join(json.dumps(r.make_record(job, ok_body()))
                                    for job in window) + "\n", encoding="utf-8")
    del cli["jobs"]
    assert r._main(_argv(cli, "--limit", "10", "--mode", "cot", "--yes")) == 0
    assert "jobs" not in cli


def test_limit_zero_calls_nothing(cli):
    assert r._main(_argv(cli, "--limit", "0", "--yes")) == 0
    assert "jobs" not in cli


def test_the_estimate_goes_to_stderr_so_a_redirect_cannot_hide_it(cli, capsys):
    """`python -m src.runner | tee run.log` must not put the cost estimate in
    the file and leave a blank terminal with an invisible prompt."""
    r._main(_argv(cli, "--dry-run"))
    captured = capsys.readouterr()
    assert "ESTIMATE" in captured.err
    assert "ESTIMATE" not in captured.out


def test_an_empty_models_flag_is_refused_by_the_cli(cli):
    assert r._main(_argv(cli, "--models", "", "--yes")) == 2
    assert "jobs" not in cli


def test_prompt_drift_blocks_the_run(cli, config, monkeypatch):
    """Resume cannot tell two wordings apart, so it must refuse rather than mix
    them in one file."""
    monkeypatch.setattr(r.sys.stdin, "isatty", lambda: False)
    job = next(job for job in m.build_jobs(config) if job["mode"] == "cot")
    stale = r.make_record(job, ok_body())
    stale["messages"] = [{"role": "user", "content": "an older wording"}]
    cli["raw"].write_text(json.dumps(stale) + "\n", encoding="utf-8")

    argv = ["--mode", "cot", "--limit", "3", "--yes"]
    assert r._main(_argv(cli, *argv)) == 2
    assert "jobs" not in cli
    assert r._main(_argv(cli, *argv, "--allow-prompt-drift")) == 0
    assert len(cli["jobs"]) == 2      # the drifted run_id still counts as done


def test_an_interrupted_run_does_not_exit_zero(cli, monkeypatch):
    """`runner --yes && annotate` must not read a half-filled grid as complete."""
    def stopped(jobs, **kwargs):
        state = r.RunState(len(jobs), {}, 0)
        state.ok, state.aborted = 1, len(jobs) - 1
        return state

    monkeypatch.setattr(r, "run_jobs", stopped)
    monkeypatch.setattr(r.sys.stdin, "isatty", lambda: False)
    assert r._main(_argv(cli, "--limit", "5", "--yes")) == 1


def test_filter_preserves_manifest_order(config):
    jobs = m.build_jobs(config)
    picked = r.filter_jobs(jobs, modes=["cot"])
    assert picked == [job for job in jobs if job["mode"] == "cot"]


# ------------------------------------------------------------------- misc ----

def test_load_api_key_prefers_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    assert r.load_api_key(tmp_path / "nope.env") == "from-env"


def test_load_api_key_falls_back_to_dotenv(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('# comment\nexport OPENROUTER_API_KEY="sk-or-v1-abc"\nOTHER=1\n',
                   encoding="utf-8")
    assert r.load_api_key(env) == "sk-or-v1-abc"


def test_load_api_key_returns_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert r.load_api_key(tmp_path / "nope.env") is None


def test_backoff_grows_and_respects_retry_after(monkeypatch):
    monkeypatch.setattr(r, "BACKOFF_BASE", 2.0)
    monkeypatch.setattr(r, "BACKOFF_JITTER", 0.0)
    assert [r.backoff_delay(i) for i in range(4)] == [1.0, 2.0, 4.0, 8.0]
    assert r.backoff_delay(0, "30") == 30.0
    assert r.backoff_delay(0, "9999") == r.MAX_RETRY_AFTER
    assert r.backoff_delay(0, "not-a-number") == 1.0


# ------------------------------------------------------- drift beyond prompt ----

def _drift_setup(tmp_path, mutate):
    """One recorded job, with `mutate` applied to the recorded copy."""
    from src import manifest as manifest_mod
    jobs = manifest_mod.build_jobs(manifest_mod.load_config())
    job = next(j for j in jobs if j["mode"] == "cot")
    record = dict(job)
    mutate(record)
    raw = tmp_path / "raw.jsonl"
    raw.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return jobs, raw


def test_drift_catches_an_edited_prompt(tmp_path):
    def mutate(rec):
        rec["messages"] = [{"role": "user", "content": "a different scenario"}]
    jobs, raw = _drift_setup(tmp_path, mutate)
    assert len(r.prompt_drift(jobs, raw)) == 1


def test_drift_catches_a_changed_provider_pin(tmp_path):
    """Not hypothetical: turning on `pin_providers` left 34 of 70 existing cot
    records served by a provider the config now excludes, and a messages-only
    check reported no drift. Provider changes the measurement — median 3
    completion tokens on Novita against 158 on Groq for the same llama prompt."""
    def mutate(rec):
        rec["provider_routing"] = None          # what an unpinned run recorded
    jobs, raw = _drift_setup(tmp_path, mutate)
    assert "provider_routing" in r.DRIFT_FIELDS
    assert len(r.prompt_drift(jobs, raw)) == 1


def test_drift_catches_a_changed_max_tokens(tmp_path):
    def mutate(rec):
        rec["max_tokens"] = 900                 # the superseded CoT cap
    jobs, raw = _drift_setup(tmp_path, mutate)
    assert len(r.prompt_drift(jobs, raw)) == 1


def test_no_drift_when_the_record_matches_the_job(tmp_path):
    jobs, raw = _drift_setup(tmp_path, lambda rec: None)
    assert r.prompt_drift(jobs, raw) == []
