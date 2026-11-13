"""Tests for the annotator, the rubric prompt and the hand-coding helpers.

No network. Every annotation call in here goes through an injected `caller`
coroutine, so the only thing these tests cannot cover is OpenRouter's own
behaviour — which `contracts.is_usable_response` is the guard for, and which is
exercised against its documented failure shapes below.

    python -m pytest tests/test_annotate.py -q
"""

import asyncio
import csv
import json
import math
from collections import Counter
from inspect import signature
from pathlib import Path

import pytest

from src import annotate as a
from src import contracts as c
from src import rubric as r

FIXTURE = c.ROOT / "tests" / "fixtures" / "raw_sample.jsonl"


# ------------------------------------------------------------------ helpers ----

def load_fixture() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def text_of(rec: dict) -> str:
    return rec["response"]["choices"][0]["message"]["content"]


def ok(payload: str, model: str = "mistralai/mistral-nemo", provider: str = "Parasail",
       cost: float = 1.2e-06) -> tuple[int, dict]:
    """An OpenRouter 200 whose assistant message is `payload`."""
    return 200, {
        "id": "gen-annot-1",
        "model": model,
        "provider": provider,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": payload}}],
        "usage": {"prompt_tokens": 800, "completion_tokens": 40,
                  "total_tokens": 840, "cost": cost},
    }


def caller_returning(*payloads):
    """A caller that returns each payload in turn, recording the prompts it saw.

    A payload may be a string (wrapped in a 200) or an explicit (status, data)
    tuple. The last payload repeats once exhausted.
    """
    prompts: list[str] = []

    async def caller(prompt: str):
        prompts.append(prompt)
        item = payloads[min(len(prompts) - 1, len(payloads) - 1)]
        return item if isinstance(item, tuple) else ok(item)

    caller.prompts = prompts  # type: ignore[attr-defined]
    return caller


def constant_caller(payload):
    async def caller(prompt: str):
        caller.prompts.append(prompt)
        return payload if isinstance(payload, tuple) else ok(payload)

    caller.prompts = []  # type: ignore[attr-defined]
    return caller


VALID_JSON = '{"answer": "SWITCH", "depth": "L1", "flag_no_basis": false, "flag_rigged": true}'


def run(records, out_path, caller, **kw):
    """Synchronous wrapper: annotate `records`, writing to `out_path`."""
    return asyncio.run(a.annotate_all(
        records, caller=caller, out_path=out_path,
        annotator="mistralai/mistral-nemo", sleep=noop_sleep, **kw))


async def noop_sleep(_seconds):  # never actually wait in tests
    return None


def lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ------------------------------------------------------------------ fixture ----

# What each fixture response is meant to be. Written down here rather than in
# the JSONL because CONTRACTS.md §4 fixes the record's fields and a fixture
# carrying extra ones would not be the contract any more. `depth_alt` lists a
# second code a reasonable annotator might defend; the assertion is only that
# the case EXISTS in the fixture, never that a model agrees with me.
EXPECTED = {
    ("meta-llama/llama-3.1-8b-instruct", "default_switch", "none", 0.4, 0):
        {"answer": "KEEP", "depth": "L1", "flag_no_basis": False, "flag_rigged": False},
    ("google/gemma-4-26b-a4b-it", "default_switch", "expert", 0.8, 3):
        {"answer": "SWITCH", "depth": "L2", "flag_no_basis": False, "flag_rigged": False},
    ("google/gemma-4-26b-a4b-it", "default_keep", "expert", 1.2, 11):
        {"answer": "KEEP", "depth": "L3", "flag_no_basis": False, "flag_rigged": False},
    ("ibm-granite/granite-4.1-8b", "isomorph", "expert", 0.8, 5):
        {"answer": "SWITCH", "depth": "MIX", "flag_no_basis": False, "flag_rigged": False},
    ("ibm-granite/granite-4.1-8b", "default_keep", "average_person", 1.6, 19):
        {"answer": "RANDOMISE", "depth": "MIX", "flag_no_basis": False, "flag_rigged": False},
    ("openai/gpt-4o-mini", "default_keep", "none", 0.0, 0):
        {"answer": "THIRD_OPTION", "depth": "NON", "flag_no_basis": False, "flag_rigged": False},
    ("openai/gpt-4o-mini", "isomorph", "average_person", 0.4, 2):
        {"answer": "REFUSE", "depth": "NON", "flag_no_basis": False, "flag_rigged": False},
    ("google/gemma-4-26b-a4b-it", "default_keep", "average_person", 1.2, 8):
        {"answer": "KEEP", "depth": "NON", "flag_no_basis": False, "flag_rigged": True},
    ("meta-llama/llama-3.1-8b-instruct", "isomorph", "none", 1.6, 14):
        {"answer": "KEEP", "depth": "L0", "flag_no_basis": True, "flag_rigged": False},
    ("meta-llama/llama-3.1-8b-instruct", "default_switch", "average_person", 1.6, 21):
        {"answer": "UNPARSEABLE", "depth": "NON", "flag_no_basis": False, "flag_rigged": False},
}

CONDITION_STRINGS = (
    "qwen/qwen3-30b-a3b-instruct-2507", "meta-llama/llama-3.1-8b-instruct",
    "ibm-granite/granite-4.1-8b", "google/gemma-4-26b-a4b-it", "openai/gpt-4o-mini",
    "default_keep", "default_switch", "isomorph",
    "average_person", "budget_v1",
    "0.0", "0.4", "0.8", "1.2", "1.6", "temperature", "persona", "replicate",
)


def test_fixture_has_ten_cot_and_two_direct_records():
    recs = load_fixture()
    assert len(recs) == 12
    assert Counter(rec["mode"] for rec in recs) == {"cot": 10, "direct": 2}


def test_fixture_run_ids_are_the_real_hash_of_their_condition():
    for rec in load_fixture():
        assert rec["run_id"] == c.run_id(
            rec["model"], rec["scenario"], rec["persona"],
            rec["mode"], rec["temperature"], rec["replicate"])


def test_fixture_records_carry_every_contract_field():
    job = {"run_id", "experiment", "model", "tier", "scenario", "persona", "mode",
           "temperature", "replicate", "messages", "max_tokens", "logprobs",
           "top_logprobs", "provider_routing"}
    extra = {"response", "model_returned", "provider", "usage", "ts"}
    for rec in load_fixture():
        assert set(rec) == job | extra, rec["run_id"]
        assert rec["ts"].endswith("Z")
        assert set(rec["usage"]) >= {"prompt_tokens", "completion_tokens", "total_tokens", "cost"}


def test_fixture_messages_are_the_ones_the_contract_builder_produces():
    for rec in load_fixture():
        assert rec["messages"] == c.build_messages(
            rec["scenario"], rec["persona"], rec["mode"]), rec["run_id"]


def test_fixture_cot_records_use_the_contract_request_parameters():
    for rec in load_fixture():
        if rec["mode"] != "cot":
            continue
        assert rec["max_tokens"] == c.COT_MAX_TOKENS
        assert rec["logprobs"] is False
        assert rec["top_logprobs"] is None
        assert rec["provider_routing"] is None


def test_fixture_direct_records_are_pinned_and_yield_usable_logprobs():
    """The two direct records exist to be skipped, but they should still be the
    real thing: pinned per CONTRACTS.md §8 and readable by answer_logprobs."""
    for rec in load_fixture():
        if rec["mode"] != "direct":
            continue
        assert rec["temperature"] == c.DIRECT_TEMPERATURE
        assert rec["replicate"] == 0
        assert rec["max_tokens"] == c.DIRECT_MAX_TOKENS
        assert rec["logprobs"] is True
        assert rec["top_logprobs"] == c.DIRECT_TOP_LOGPROBS
        assert rec["provider_routing"] == c.DIRECT_PROVIDER_PIN[rec["model"]]
        logprobs, status = c.answer_logprobs(rec["response"]["choices"][0])
        assert status == "ok" and set(logprobs) == {"K", "S"}


def test_fixture_covers_every_intended_annotation_case():
    """One record per answer label and per depth code, so the live run is an
    actual test of the annotator rather than a sample of whatever it saw."""
    by_id = {rec["run_id"]: rec for rec in load_fixture()}
    for key, want in EXPECTED.items():
        rid = c.run_id(key[0], key[1], key[2], "cot", key[3], key[4])
        assert rid in by_id, key
    assert {e["answer"] for e in EXPECTED.values()} == set(c.ANSWERS)
    assert {e["depth"] for e in EXPECTED.values()} == set(c.DEPTH_CODES)
    assert any(e["flag_no_basis"] for e in EXPECTED.values())
    assert any(e["flag_rigged"] for e in EXPECTED.values())


def test_fixture_cot_responses_are_long_enough_to_depth_code():
    for rec in load_fixture():
        if rec["mode"] == "cot":
            assert len(text_of(rec)) > 400, rec["run_id"]


def test_fixture_response_texts_leak_no_condition_strings():
    """The annotator is blind to condition, which is only true if the response
    text itself does not name the condition."""
    for rec in load_fixture():
        low = text_of(rec)
        for token in CONDITION_STRINGS:
            assert token not in low, (rec["run_id"], token)


# ------------------------------------------------------------------- rubric ----

def test_prompt_is_the_template_with_only_the_response_substituted():
    text = "I would keep it.\n\nKEEP"
    assert r.build_annotation_prompt(text) == \
        r.ANNOTATION_PROMPT_TEMPLATE.replace("{response_text}", text)


def test_prompt_carries_the_depth_rubric_whole():
    """PLAN.md §8.4 pastes §5's rubric in; CONTRACTS.md §5 requires the flag
    definitions to come with it, or the JSON schema asks for two booleans the
    annotator has never had defined."""
    prompt = r.build_annotation_prompt("x")
    assert c.DEPTH_RUBRIC in prompt
    assert "no rational basis for choosing" in prompt
    assert "suspects a trick" in prompt


def test_prompt_names_every_answer_label_and_depth_code():
    prompt = r.build_annotation_prompt("x")
    for label in c.ANSWERS:
        assert label in prompt
    for code in c.DEPTH_CODES:
        assert code in prompt
    for key in ("answer", "depth", "flag_no_basis", "flag_rigged"):
        assert f'"{key}"' in prompt or f"`{key}`" in prompt


def test_prompt_demands_bare_json():
    prompt = r.build_annotation_prompt("x")
    assert "Return ONLY a JSON object" in prompt
    assert "RESPONSE TO CODE:" in prompt


def test_build_annotation_prompt_takes_only_the_response_text():
    """It must be impossible to hand the rubric a whole raw record: the
    annotator is blind to condition (CONTRACTS.md §5)."""
    params = signature(r.build_annotation_prompt).parameters
    assert list(params) == ["response_text"]
    with pytest.raises(TypeError):
        r.build_annotation_prompt("x", model="openai/gpt-4o-mini")  # type: ignore[call-arg]


def test_rubric_prompt_leaks_no_condition_strings():
    """Everything the rubric itself contributes must be condition-free, for
    every record in the fixture."""
    for rec in load_fixture():
        text = text_of(rec)
        scaffolding = r.build_annotation_prompt(text).replace(text, "")
        for token in CONDITION_STRINGS + (rec["model"], rec["scenario"], rec["persona"]):
            if token == "none":       # the persona baseline's name is a common word
                continue
            assert token not in scaffolding, (rec["run_id"], token)


# ------------------------------------------------------- selecting records ----

def test_only_cot_records_are_selected_from_the_fixture():
    assert [rec["mode"] for rec in a.cot_records(load_fixture())] == ["cot"] * 10


def test_every_mode_but_cot_is_excluded_by_the_whitelist():
    """Whitelisting cot, not blacklisting direct: contamination and
    direct_validation are also in raw.jsonl and must not leak in."""
    recs = [{"run_id": f"id-{mode}", "mode": mode} for mode in c.ALL_MODES]
    assert [rec["mode"] for rec in a.cot_records(recs)] == ["cot"]
    assert len(c.ALL_MODES) == 4


def test_a_record_with_no_mode_is_not_annotated():
    assert a.cot_records([{"run_id": "x"}]) == []


def test_direct_records_are_never_sent_to_the_annotator(tmp_path):
    out = tmp_path / "annotated.jsonl"
    caller = constant_caller(VALID_JSON)
    stats = run(load_fixture(), out, caller)
    assert len(caller.prompts) == 10
    assert stats.annotated == 10
    direct_ids = {rec["run_id"] for rec in load_fixture() if rec["mode"] == "direct"}
    assert direct_ids.isdisjoint({rec["run_id"] for rec in lines(out)})


def test_the_prompt_sent_for_each_record_contains_that_record_s_text(tmp_path):
    out = tmp_path / "annotated.jsonl"
    caller = constant_caller(VALID_JSON)
    run(load_fixture(), out, caller)
    for prompt, rec in zip(caller.prompts, a.cot_records(load_fixture())):
        assert text_of(rec) in prompt


# ------------------------------------------------------------------ parsing ----

def test_valid_json_produces_a_contract_valid_record():
    rec = a.build_annotated_record(
        run_id="r1", raw_output=VALID_JSON, annotator_model="mistralai/mistral-nemo")
    assert rec["run_id"] == "r1"
    assert rec["answer"] == "SWITCH"
    assert rec["depth"] == "L1"
    assert rec["flag_no_basis"] is False
    assert rec["flag_rigged"] is True
    assert rec["annotator_model"] == "mistralai/mistral-nemo"
    assert rec["raw_annotator_output"] == VALID_JSON
    assert rec["parse_ok"] is True
    assert rec["parse_error"] is None


def test_garbage_produces_unparseable_without_raising():
    rec = a.build_annotated_record(
        run_id="r1", raw_output="I think they would probably switch, honestly.",
        annotator_model="m")
    assert rec["answer"] == "UNPARSEABLE"
    assert rec["depth"] == "NON"
    assert rec["flag_no_basis"] is False
    assert rec["flag_rigged"] is False
    assert rec["parse_ok"] is False
    assert rec["parse_error"]


def test_an_unparseable_record_keeps_the_raw_output_for_review():
    junk = "```\nnot json at all\n```"
    rec = a.build_annotated_record(run_id="r1", raw_output=junk, annotator_model="m")
    assert rec["raw_annotator_output"] == junk


@pytest.mark.parametrize("fenced", [
    '```json\n{"answer": "KEEP", "depth": "L2", "flag_no_basis": false, "flag_rigged": false}\n```',
    '```\n{"answer": "KEEP", "depth": "L2", "flag_no_basis": false, "flag_rigged": false}\n```',
    '```JSON\r\n{"answer": "KEEP", "depth": "L2", "flag_no_basis": false, "flag_rigged": false}\r\n```',
    '   ```json {"answer": "KEEP", "depth": "L2", "flag_no_basis": false, "flag_rigged": false} ```  ',
])
def test_markdown_fenced_json_is_parsed(fenced):
    rec = a.build_annotated_record(run_id="r1", raw_output=fenced, annotator_model="m")
    assert (rec["answer"], rec["depth"], rec["parse_ok"]) == ("KEEP", "L2", True)


@pytest.mark.parametrize("fenced,bare", [
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('```\n{"a": 1}\n```', '{"a": 1}'),
    ('```JSON\r\n{"a": 1}\r\n```', '{"a": 1}'),
    ('  ```json {"a": 1} ```  ', '{"a": 1}'),
    ('{"a": 1}', '{"a": 1}'),                      # nothing to strip
    ('the answer is ```json``` shaped', 'the answer is ```json``` shaped'),
])
def test_strip_fences_removes_one_layer_of_fencing(fenced, bare):
    """Pinned on its own, not just through the parser: the brace scanner in
    extract_json_object tolerates fences by itself, so an integration test
    passes even with fence-stripping removed."""
    assert a.strip_fences(fenced) == bare


def test_json_wrapped_in_prose_is_parsed():
    raw = ('Sure! Here is the coding:\n'
           '{"answer": "REFUSE", "depth": "NON", "flag_no_basis": true, "flag_rigged": false}\n'
           'Let me know if you need anything else.')
    rec = a.build_annotated_record(run_id="r1", raw_output=raw, annotator_model="m")
    assert (rec["answer"], rec["depth"], rec["flag_no_basis"]) == ("REFUSE", "NON", True)
    assert rec["parse_ok"] is True


def test_a_nested_json_object_is_still_parsed():
    raw = '{"answer": "KEEP", "depth": "L1", "flag_no_basis": false, "flag_rigged": false, "notes": {"why": "he kept the safe one"}}'
    rec = a.build_annotated_record(run_id="r1", raw_output=raw, annotator_model="m")
    assert (rec["answer"], rec["parse_ok"]) == ("KEEP", True)


@pytest.mark.parametrize("bad", [
    '{"answer": "KEEP", "depth": "L4", "flag_no_basis": false, "flag_rigged": false}',
    '{"answer": "KEEP", "depth": "", "flag_no_basis": false, "flag_rigged": false}',
    '{"answer": "KEEP", "flag_no_basis": false, "flag_rigged": false}',
])
def test_an_invalid_depth_code_is_unparseable(bad):
    rec = a.build_annotated_record(run_id="r1", raw_output=bad, annotator_model="m")
    assert (rec["answer"], rec["depth"], rec["parse_ok"]) == ("UNPARSEABLE", "NON", False)


def test_an_answer_outside_allowed_answers_is_unparseable():
    """config `allowed_answers` is how PLAN.md §4's RANDOMISE toggle is
    expressed (CONTRACTS.md §1): a label it excludes is not a result."""
    raw = '{"answer": "RANDOMISE", "depth": "MIX", "flag_no_basis": false, "flag_rigged": false}'
    allowed = ["KEEP", "SWITCH", "THIRD_OPTION", "REFUSE"]
    rec = a.build_annotated_record(run_id="r1", raw_output=raw, annotator_model="m",
                                   allowed_answers=allowed)
    assert rec["answer"] == "UNPARSEABLE"
    assert "RANDOMISE" in rec["raw_annotator_output"]
    permitted = a.build_annotated_record(run_id="r1", raw_output=raw, annotator_model="m",
                                         allowed_answers=list(c.ANSWERS))
    assert permitted["answer"] == "RANDOMISE"


def test_an_annotator_answer_of_unparseable_is_not_a_parse_failure():
    """CONTRACTS.md §5: 'the annotator returned invalid JSON' and 'the
    annotator said UNPARSEABLE' are different failures and both get reviewed."""
    raw = '{"answer": "UNPARSEABLE", "depth": "NON", "flag_no_basis": false, "flag_rigged": false}'
    rec = a.build_annotated_record(run_id="r1", raw_output=raw, annotator_model="m")
    assert rec["answer"] == "UNPARSEABLE"
    assert rec["parse_ok"] is True
    assert rec["parse_error"] is None


@pytest.mark.parametrize("value,expected", [
    ("true", True), ("False", False), ("TRUE", True), (1, True), (0, False),
    ("yes", True), ("no", False),
])
def test_string_and_integer_booleans_are_coerced(value, expected):
    raw = json.dumps({"answer": "KEEP", "depth": "L1",
                      "flag_no_basis": value, "flag_rigged": False})
    rec = a.build_annotated_record(run_id="r1", raw_output=raw, annotator_model="m")
    assert rec["parse_ok"] is True
    assert rec["flag_no_basis"] is expected


def test_a_flag_that_is_not_a_boolean_is_unparseable():
    raw = '{"answer": "KEEP", "depth": "L1", "flag_no_basis": "maybe", "flag_rigged": false}'
    rec = a.build_annotated_record(run_id="r1", raw_output=raw, annotator_model="m")
    assert (rec["answer"], rec["parse_ok"]) == ("UNPARSEABLE", False)


def test_answers_and_codes_are_case_and_space_normalised():
    raw = '{"answer": " keep ", "depth": "l1", "flag_no_basis": false, "flag_rigged": false}'
    rec = a.build_annotated_record(run_id="r1", raw_output=raw, annotator_model="m")
    assert (rec["answer"], rec["depth"], rec["parse_ok"]) == ("KEEP", "L1", True)


# ----------------------------------------------------------------- the run ----

def test_a_written_record_has_exactly_the_contract_fields(tmp_path):
    out = tmp_path / "annotated.jsonl"
    run(load_fixture()[:1], out, constant_caller(VALID_JSON))
    rec, = lines(out)
    required = {"run_id", "answer", "depth", "flag_no_basis", "flag_rigged",
                "annotator_model", "raw_annotator_output"}
    recommended = {"ts", "annotator_provider", "parse_ok"}
    assert required <= set(rec)
    assert recommended <= set(rec)
    assert rec["annotator_model"] == "mistralai/mistral-nemo"   # from the response
    assert rec["annotator_provider"] == "Parasail"
    assert rec["answer"] in c.ANSWERS and rec["depth"] in c.DEPTH_CODES
    assert rec["ts"].endswith("Z")


def test_annotator_model_comes_from_the_response_not_the_config(tmp_path):
    """CONTRACTS.md §5: PLAN.md §8.4 anticipates swapping annotators, so the
    line must record which model actually answered."""
    out = tmp_path / "annotated.jsonl"
    caller = constant_caller(ok(VALID_JSON, model="mistralai/mistral-small-3.2-24b"))
    run(load_fixture()[:1], out, caller)
    rec, = lines(out)
    assert rec["annotator_model"] == "mistralai/mistral-small-3.2-24b"


def test_rerunning_on_the_same_input_adds_no_records(tmp_path):
    out = tmp_path / "annotated.jsonl"
    recs = load_fixture()
    first = run(recs, out, constant_caller(VALID_JSON))
    assert first.annotated == 10 and len(lines(out)) == 10

    caller = constant_caller(VALID_JSON)
    second = run(recs, out, caller)
    assert second.annotated == 0
    assert second.already_annotated == 10
    assert caller.prompts == []
    assert len(lines(out)) == 10


def test_one_line_per_run_id(tmp_path):
    out = tmp_path / "annotated.jsonl"
    run(load_fixture() + load_fixture(), out, constant_caller(VALID_JSON))
    ids = [rec["run_id"] for rec in lines(out)]
    assert len(ids) == len(set(ids)) == 10


def test_a_transport_failure_writes_no_line_so_a_rerun_retries_it(tmp_path):
    out = tmp_path / "annotated.jsonl"
    stats = run(load_fixture()[:1], out, constant_caller((500, {})))
    assert lines(out) == []
    assert stats.failed == 1 and stats.annotated == 0


def test_an_http_200_error_body_is_a_failure_not_a_record(tmp_path):
    """CONTRACTS.md §4: OpenRouter returns 200 with an error object and no
    choices. Writing that would mark the run_id done and resume would never
    retry it."""
    out = tmp_path / "annotated.jsonl"
    body = (200, {"id": "gen-1", "error": {"message": "Provider returned error", "code": 429}})
    stats = run(load_fixture()[:1], out, constant_caller(body))
    assert lines(out) == []
    assert stats.failed == 1
    assert stats.failure_reasons["error_body_429"] == 1


def test_a_rate_limit_is_retried_and_then_succeeds(tmp_path):
    out = tmp_path / "annotated.jsonl"
    caller = caller_returning((429, {"error": {"code": 429}}), VALID_JSON)
    stats = run(load_fixture()[:1], out, caller)
    assert len(caller.prompts) == 2
    assert stats.annotated == 1 and stats.failed == 0
    assert len(lines(out)) == 1


def test_retries_are_bounded(tmp_path):
    out = tmp_path / "annotated.jsonl"
    caller = constant_caller((503, {}))
    run(load_fixture()[:1], out, caller)
    assert len(caller.prompts) == a.MAX_ATTEMPTS


def test_one_bad_record_does_not_stop_the_others(tmp_path):
    out = tmp_path / "annotated.jsonl"
    payloads = [VALID_JSON] * 10
    payloads[4] = "total nonsense"
    caller = caller_returning(*payloads)
    stats = run(load_fixture(), out, caller)
    assert stats.annotated == 10
    # 1 parse failure + 2 answer-floor corrections: the mock claims SWITCH
    # for every record, including the fixture's THIRD_OPTION and REFUSE
    # responses, which contain no answer token at all.
    assert stats.unparseable == 3
    assert len(lines(out)) == 10


def test_limit_caps_the_number_of_calls(tmp_path):
    out = tmp_path / "annotated.jsonl"
    caller = constant_caller(VALID_JSON)
    stats = run(load_fixture(), out, caller, limit=3)
    assert len(caller.prompts) == 3
    assert stats.annotated == 3 and len(lines(out)) == 3


def test_every_written_line_is_independently_parseable(tmp_path):
    """Concurrent appends must not interleave."""
    out = tmp_path / "annotated.jsonl"
    run(load_fixture(), out, constant_caller(VALID_JSON), concurrency=10)
    for line in out.read_text(encoding="utf-8").splitlines():
        assert isinstance(json.loads(line), dict)


def test_unparseable_rate_and_the_two_percent_warning(tmp_path):
    out = tmp_path / "annotated.jsonl"
    payloads = ["nonsense"] + [VALID_JSON] * 9
    stats = run(load_fixture(), out, caller_returning(*payloads))
    # 1 parse failure + 2 answer-floor corrections: the mock claims SWITCH
    # for every record, including the fixture's THIRD_OPTION and REFUSE
    # responses, which contain no answer token at all.
    assert stats.unparseable == 3
    assert math.isclose(stats.unparseable_rate(), 0.3)
    report = a.format_summary(stats)
    assert "UNPARSEABLE" in report and "30.0%" in report
    assert "WARNING" in report

    # A constant SWITCH payload parses every time but is WRONG for the
    # fixture's two answer-word-free responses (a THIRD_OPTION and a REFUSE),
    # and the answer floor corrects both. So this run has zero parse failures
    # and a non-zero UNPARSEABLE rate — which is the floor doing its job, and
    # the warning correctly fires.
    clean = run(load_fixture(), tmp_path / "b.jsonl", constant_caller(VALID_JSON))
    assert clean.parse_failures == 0
    assert clean.unparseable_rate() == 0.2
    assert "WARNING" in a.format_summary(clean)


def test_spend_is_summed_from_usage(tmp_path):
    out = tmp_path / "annotated.jsonl"
    stats = run(load_fixture(), out, constant_caller(ok(VALID_JSON, cost=2e-06)))
    assert math.isclose(stats.cost, 10 * 2e-06)


def test_mixing_annotators_in_one_file_is_refused(tmp_path):
    """CONTRACTS.md §5: a re-annotation with a different model goes to a NEW
    file, because nothing downstream can tell it from a double-write."""
    out = tmp_path / "annotated.jsonl"
    run(load_fixture()[:1], out, constant_caller(VALID_JSON))
    with pytest.raises(a.AnnotatorMismatch):
        asyncio.run(a.annotate_all(
            load_fixture()[1:2], caller=constant_caller(VALID_JSON), out_path=out,
            annotator="google/gemma-4-26b-a4b-it", sleep=noop_sleep))


def test_the_annotation_call_is_made_at_temperature_zero():
    body = a.annotation_request_body("mistralai/mistral-nemo", "PROMPT")
    assert body["temperature"] == 0
    assert body["model"] == "mistralai/mistral-nemo"
    assert body["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert "response_format" not in body
    assert a.annotation_request_body("m", "P", json_mode=True)["response_format"] == \
        {"type": "json_object"}


# -------------------------------------------------- hand-coding export ----

def test_export_writes_the_hand_coding_columns_with_empty_label_cells(tmp_path):
    out = tmp_path / "handcode.csv"
    a.export_for_handcoding(load_fixture(), 6, out, seed=20260903)
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(a.HANDCODE_COLUMNS)
    assert len(rows) == 6
    for row in rows:
        assert row["run_id"] and row["response_text"]
        assert row["answer"] == row["depth"] == "" == row["flag_no_basis"] == row["flag_rigged"]


def test_export_only_samples_cot_records(tmp_path):
    out = tmp_path / "handcode.csv"
    a.export_for_handcoding(load_fixture(), 12, out, seed=1)
    with out.open(newline="", encoding="utf-8") as f:
        ids = {row["run_id"] for row in csv.DictReader(f)}
    cot = {rec["run_id"] for rec in load_fixture() if rec["mode"] == "cot"}
    assert ids <= cot and len(ids) == 10


def test_export_is_stratified_by_model_and_scenario(tmp_path):
    """PLAN.md §8.4 wants 60 responses spread over the grid, not 60 from
    whichever model happens to sit at the top of the file."""
    recs = []
    for model in ("m1", "m2", "m3"):
        for scenario in ("default_keep", "default_switch", "isomorph"):
            n = 40 if model == "m1" else 4          # m1 dominates the file
            for i in range(n):
                recs.append({"run_id": f"{model}-{scenario}-{i}", "mode": "cot",
                             "model": model, "scenario": scenario,
                             "response": {"choices": [{"message": {"content": f"text {i}"}}]}})
    out = tmp_path / "handcode.csv"
    a.export_for_handcoding(recs, 18, out, seed=20260903)
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_stratum = Counter(row["run_id"].rsplit("-", 1)[0] for row in rows)
    assert len(rows) == 18
    assert set(by_stratum.values()) == {2}, by_stratum


def test_export_fills_from_other_strata_when_one_is_too_small(tmp_path):
    recs = [{"run_id": "big-1", "mode": "cot", "model": "m1", "scenario": "s1",
             "response": {"choices": [{"message": {"content": "a"}}]}}]
    recs += [{"run_id": f"m2-{i}", "mode": "cot", "model": "m2", "scenario": "s1",
              "response": {"choices": [{"message": {"content": "b"}}]}} for i in range(20)]
    out = tmp_path / "handcode.csv"
    a.export_for_handcoding(recs, 10, out, seed=7)
    with out.open(newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 10


def test_export_is_deterministic_for_a_seed(tmp_path):
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    a.export_for_handcoding(load_fixture(), 5, first, seed=20260903)
    a.export_for_handcoding(load_fixture(), 5, second, seed=20260903)
    assert first.read_bytes() == second.read_bytes()


def test_export_round_trips_a_response_with_newlines_and_quotes(tmp_path):
    rec = {"run_id": "r1", "mode": "cot", "model": "m", "scenario": "s",
           "response": {"choices": [{"message": {"content": 'line one\n\nHe said "no".\n\nKEEP'}}]}}
    out = tmp_path / "handcode.csv"
    a.export_for_handcoding([rec], 1, out, seed=1)
    with out.open(newline="", encoding="utf-8") as f:
        row, = list(csv.DictReader(f))
    assert row["response_text"] == 'line one\n\nHe said "no".\n\nKEEP'


def test_export_prefers_records_that_are_already_annotated(tmp_path):
    """The point of the sample is a kappa against the machine codes, so a row
    with no machine code to compare against is a wasted twenty minutes."""
    out = tmp_path / "handcode.csv"
    cot = [rec["run_id"] for rec in a.cot_records(load_fixture())]
    a.export_for_handcoding(load_fixture(), 4, out, seed=1, restrict_ids=set(cot[:4]))
    with out.open(newline="", encoding="utf-8") as f:
        assert {row["run_id"] for row in csv.DictReader(f)} == set(cot[:4])


# ------------------------------------------------------------------- kappa ----

def test_cohens_kappa_matches_a_hand_computed_table():
    # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> po=0.70, pe=0.50, k=0.40
    human = ["yes"] * 25 + ["no"] * 25
    model = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    assert math.isclose(a.cohens_kappa(human, model), 0.4, abs_tol=1e-12)


def test_cohens_kappa_is_one_for_perfect_agreement():
    labels = ["L0", "L1", "L2", "L3", "MIX", "NON", "L1", "L1"]
    assert a.cohens_kappa(labels, list(labels)) == 1.0


def test_cohens_kappa_is_zero_for_chance_agreement():
    human = ["A", "A", "B", "B"]
    model = ["A", "B", "A", "B"]
    assert math.isclose(a.cohens_kappa(human, model), 0.0, abs_tol=1e-12)


def test_cohens_kappa_is_undefined_when_both_raters_are_constant():
    assert math.isnan(a.cohens_kappa(["KEEP"] * 5, ["KEEP"] * 5))


def test_cohens_kappa_is_negative_for_systematic_disagreement():
    assert a.cohens_kappa(["A", "B"] * 4, ["B", "A"] * 4) < 0


def test_confusion_matrix_counts_pairs_in_canonical_order():
    human = ["L1", "L1", "L2", "NON"]
    model = ["L1", "L2", "L2", "L1"]
    matrix = a.confusion_matrix(human, model, list(c.DEPTH_CODES))
    assert matrix["L1"]["L1"] == 1
    assert matrix["L1"]["L2"] == 1
    assert matrix["L2"]["L2"] == 1
    assert matrix["NON"]["L1"] == 1
    assert list(matrix) == list(c.DEPTH_CODES)
    assert sum(sum(row.values()) for row in matrix.values()) == 4


def write_handcode_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(a.HANDCODE_COLUMNS)
        writer.writerows(rows)


def test_kappa_report_joins_on_run_id_and_scores_all_four_fields(tmp_path):
    annotated = tmp_path / "annotated.jsonl"
    with annotated.open("w", encoding="utf-8") as f:
        for i in range(6):
            f.write(json.dumps({
                "run_id": f"r{i}",
                "answer": "KEEP" if i % 2 == 0 else "SWITCH",
                "depth": "L1" if i < 4 else "L2",
                "flag_no_basis": i == 0,
                "flag_rigged": False,
                "annotator_model": "mistralai/mistral-nemo",
                "raw_annotator_output": "{}",
            }) + "\n")

    csv_path = tmp_path / "handcode.csv"
    write_handcode_csv(csv_path, [
        [f"r{i}", "text", "KEEP" if i % 2 == 0 else "SWITCH",
         "L1" if i < 4 else "L2", "TRUE" if i == 0 else "FALSE", "FALSE"]
        for i in range(6)
    ])

    result = a.kappa_report(csv_path, annotated)
    assert result.n == 6
    for field in ("answer", "depth", "flag_no_basis", "flag_rigged"):
        assert field in result.kappa
    assert result.kappa["answer"] == 1.0
    assert result.kappa["depth"] == 1.0
    assert math.isnan(result.kappa["flag_rigged"])       # both raters constant
    text = a.format_kappa_report(result)
    assert "depth" in text and "confusion" in text.lower()


def test_kappa_report_ignores_rows_i_have_not_filled_in(tmp_path):
    annotated = tmp_path / "annotated.jsonl"
    with annotated.open("w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({"run_id": f"r{i}", "answer": "KEEP", "depth": "L1",
                                "flag_no_basis": False, "flag_rigged": False}) + "\n")
    csv_path = tmp_path / "handcode.csv"
    write_handcode_csv(csv_path, [
        ["r0", "t", "KEEP", "L1", "FALSE", "FALSE"],
        ["r1", "t", "", "", "", ""],
        ["r2", "t", "SWITCH", "L2", "FALSE", "FALSE"],
    ])
    result = a.kappa_report(csv_path, annotated)
    assert result.n == 2
    assert result.unfilled == 1


def test_kappa_report_reports_rows_that_do_not_join(tmp_path):
    annotated = tmp_path / "annotated.jsonl"
    annotated.write_text(json.dumps({"run_id": "r0", "answer": "KEEP", "depth": "L1",
                                     "flag_no_basis": False, "flag_rigged": False}) + "\n",
                         encoding="utf-8")
    csv_path = tmp_path / "handcode.csv"
    write_handcode_csv(csv_path, [
        ["r0", "t", "KEEP", "L1", "FALSE", "FALSE"],
        ["ghost", "t", "KEEP", "L1", "FALSE", "FALSE"],
    ])
    result = a.kappa_report(csv_path, annotated)
    assert result.n == 1
    assert result.unjoined == ["ghost"]


def test_kappa_report_rejects_a_hand_label_outside_the_enum(tmp_path):
    annotated = tmp_path / "annotated.jsonl"
    annotated.write_text(json.dumps({"run_id": "r0", "answer": "KEEP", "depth": "L1",
                                     "flag_no_basis": False, "flag_rigged": False}) + "\n",
                         encoding="utf-8")
    csv_path = tmp_path / "handcode.csv"
    write_handcode_csv(csv_path, [["r0", "t", "KEEPP", "L1", "FALSE", "FALSE"]])
    with pytest.raises(a.HandCodingError) as excinfo:
        a.kappa_report(csv_path, annotated)
    assert "KEEPP" in str(excinfo.value)


def test_kappa_report_warns_below_the_depth_threshold(tmp_path):
    annotated = tmp_path / "annotated.jsonl"
    with annotated.open("w", encoding="utf-8") as f:
        for i in range(8):
            f.write(json.dumps({"run_id": f"r{i}", "answer": "KEEP",
                                "depth": "L1" if i % 2 else "L2",
                                "flag_no_basis": False, "flag_rigged": False}) + "\n")
    csv_path = tmp_path / "handcode.csv"
    write_handcode_csv(csv_path, [
        [f"r{i}", "t", "KEEP", "L2" if i % 2 else "L1", "FALSE", "FALSE"] for i in range(8)
    ])
    result = a.kappa_report(csv_path, annotated)
    assert result.kappa["depth"] < 0.7
    text = a.format_kappa_report(result)
    assert "0.7" in text and ("WARNING" in text or "mid-tier" in text)


# --------------------------------------------------------------------- cli ----

def test_cli_export_and_kappa_round_trip(tmp_path, capsys):
    csv_path = tmp_path / "handcode.csv"
    rc = a.main(["--input", str(FIXTURE), "--export-for-handcoding", "4",
                 "--out", str(csv_path), "--config", str(c.CONFIG_PATH)])
    assert rc == 0 and csv_path.exists()
    out = capsys.readouterr().out
    assert str(csv_path) in out


def test_cli_refuses_to_annotate_without_an_api_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    rc = a.main(["--input", str(FIXTURE), "--annotated", str(tmp_path / "a.jsonl"),
                 "--config", str(c.CONFIG_PATH)])
    assert rc == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_cli_warns_when_a_fixture_would_be_written_into_the_real_dataset(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    a.main(["--input", str(FIXTURE), "--config", str(c.CONFIG_PATH)])
    combined = capsys.readouterr()
    assert "raw.jsonl" in (combined.out + combined.err)


# ------------------------------------------- post-review regression tests ----
# Findings from an adversarial review pass. Each test below was written to fail
# against the code as it stood, and the production change that fixes it is
# named in the docstring.

def test_resume_survives_an_annotator_id_that_differs_from_the_request(tmp_path):
    """CONTRACTS.md §4 says the returned model may differ from the requested
    one, and §5 says annotated.jsonl records the RETURNED id. Guarding resume
    on the returned id therefore locks the file the moment OpenRouter echoes a
    routed variant: every later run aborts instead of resuming."""
    out = tmp_path / "annotated.jsonl"
    routed = ok(VALID_JSON, model="mistralai/mistral-nemo:free")
    first = run(load_fixture()[:2], out, constant_caller(routed))
    assert first.annotated == 2
    assert {rec["annotator_model"] for rec in lines(out)} == {"mistralai/mistral-nemo:free"}

    second = run(load_fixture(), out, constant_caller(routed))
    assert second.already_annotated == 2
    assert second.annotated == 8


def test_unparseable_count_includes_annotations_the_annotator_coded_unparseable(tmp_path):
    """PLAN.md §4 sets the 2% target on the UNPARSEABLE *category*, not on
    invalid JSON. Counting only parse failures under-reports it — a run where
    the annotator codes 100% UNPARSEABLE would report 0%."""
    out = tmp_path / "annotated.jsonl"
    coded = '{"answer": "UNPARSEABLE", "depth": "NON", "flag_no_basis": false, "flag_rigged": false}'
    stats = run(load_fixture(), out, constant_caller(coded))
    assert all(rec["answer"] == "UNPARSEABLE" for rec in lines(out))
    assert stats.unparseable == 10
    assert stats.unparseable_rate() == 1.0
    assert "WARNING" in a.format_summary(stats)


def test_parse_failures_are_counted_separately_from_unparseable_answers(tmp_path):
    """CONTRACTS.md §5: 'invalid JSON' and 'the annotator said UNPARSEABLE' are
    different failures and both get hand-reviewed, so both get a number."""
    out = tmp_path / "annotated.jsonl"
    coded = '{"answer": "UNPARSEABLE", "depth": "NON", "flag_no_basis": false, "flag_rigged": false}'
    payloads = ["not json"] * 2 + [coded] * 3 + [VALID_JSON] * 5
    stats = run(load_fixture(), out, caller_returning(*payloads))
    # 2 unparsed + 3 coded UNPARSEABLE + 2 the answer floor corrects
    assert stats.unparseable == 7
    assert stats.parse_failures == 2
    report = a.format_summary(stats)
    assert "invalid JSON" in report


def test_limit_resumes_where_the_previous_run_stopped(tmp_path):
    """--limit must truncate AFTER the resume filter. Truncating before it
    re-selects the same already-done ids and the run can never progress."""
    out = tmp_path / "annotated.jsonl"
    first = run(load_fixture(), out, constant_caller(VALID_JSON), limit=3)
    second = run(load_fixture(), out, constant_caller(VALID_JSON), limit=3)
    assert (first.annotated, second.annotated) == (3, 3)
    assert second.already_annotated == 3
    ids = [rec["run_id"] for rec in lines(out)]
    assert len(ids) == len(set(ids)) == 6


def test_an_unauthorised_key_stops_the_run_instead_of_retrying(tmp_path):
    """A dead key fails all 5,625 calls. Retrying each five times with backoff
    is two hours of 401s, so 401/403 aborts."""
    out = tmp_path / "annotated.jsonl"
    caller = constant_caller((401, {"error": {"message": "No auth credentials found"}}))
    with pytest.raises(a.FatalAnnotatorError):
        run(load_fixture(), out, caller)
    # one call per record at most: no record is retried MAX_ATTEMPTS times
    assert len(caller.prompts) <= len(a.cot_records(load_fixture()))
    assert lines(out) == []


def test_export_falls_back_when_no_sampled_record_is_annotated_yet(tmp_path, capsys):
    """restrict_ids narrows the pool to records with machine codes. If that
    empties it, the export must still produce rows rather than crash on an
    empty Counter."""
    annotated = tmp_path / "annotated.jsonl"
    annotated.write_text(json.dumps({"run_id": "not-in-the-fixture", "answer": "KEEP",
                                     "depth": "L1", "flag_no_basis": False,
                                     "flag_rigged": False}) + "\n", encoding="utf-8")
    out_csv = tmp_path / "handcode.csv"
    rc = a.main(["--input", str(FIXTURE), "--annotated", str(annotated),
                 "--export-for-handcoding", "4", "--out", str(out_csv),
                 "--config", str(c.CONFIG_PATH)])
    assert rc == 0
    with out_csv.open(newline="", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 4


def test_export_of_an_empty_pool_fails_cleanly(tmp_path, capsys):
    empty = tmp_path / "only_direct.jsonl"
    direct = [rec for rec in load_fixture() if rec["mode"] == "direct"]
    empty.write_text("".join(json.dumps(rec) + "\n" for rec in direct), encoding="utf-8")
    rc = a.main(["--input", str(empty), "--annotated", str(tmp_path / "a.jsonl"),
                 "--export-for-handcoding", "4", "--out", str(tmp_path / "h.csv"),
                 "--config", str(c.CONFIG_PATH)])
    assert rc == 2
    assert "cot" in capsys.readouterr().err


def test_export_warns_about_cells_a_spreadsheet_would_read_as_a_formula(tmp_path, capsys):
    """The CSV exists to be opened in Excel or Sheets. A response starting with
    = or + renders as a formula there, so the hand-coder would code a response
    whose text they never saw. The text is left byte-exact — mangling it would
    be worse — but the export says which rows to check."""
    rec = {"run_id": "r1", "mode": "cot", "model": "m", "scenario": "s",
           "response": {"choices": [{"message": {"content": "=SUM(1,2) then KEEP"}}]}}
    out = tmp_path / "handcode.csv"
    a.export_for_handcoding([rec], 1, out, seed=1)
    captured = capsys.readouterr()
    warning = captured.out + captured.err
    assert "r1" in warning or "formula" in warning.lower()
    with out.open(newline="", encoding="utf-8") as f:
        row, = list(csv.DictReader(f))
    assert row["response_text"] == "=SUM(1,2) then KEEP"      # unmangled


def test_kappa_reads_a_csv_saved_by_excel_with_a_byte_order_mark(tmp_path):
    """Excel's 'CSV UTF-8' and Google Sheets both prepend a BOM, which turns
    the first header into '﻿run_id' and made the column check fail."""
    annotated = tmp_path / "annotated.jsonl"
    annotated.write_text(json.dumps({"run_id": "r0", "answer": "KEEP", "depth": "L1",
                                     "flag_no_basis": False, "flag_rigged": False}) + "\n",
                         encoding="utf-8")
    csv_path = tmp_path / "handcode.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(a.HANDCODE_COLUMNS)
        writer.writerow(["r0", "text", "KEEP", "L1", "FALSE", "FALSE"])
    result = a.kappa_report(csv_path, annotated)
    assert result.n == 1


def test_cli_exports_then_scores_a_filled_csv(tmp_path, capsys):
    """The whole hand-coding loop through the CLI, not just the export half."""
    annotated = tmp_path / "annotated.jsonl"
    with annotated.open("w", encoding="utf-8") as f:
        for rec in a.cot_records(load_fixture()):
            f.write(json.dumps({"run_id": rec["run_id"], "answer": "KEEP", "depth": "L1",
                                "flag_no_basis": False, "flag_rigged": False,
                                "annotator_model": "mistralai/mistral-nemo",
                                "raw_annotator_output": "{}"}) + "\n")
    csv_path = tmp_path / "handcode.csv"
    rc = a.main(["--input", str(FIXTURE), "--annotated", str(annotated),
                 "--export-for-handcoding", "4", "--out", str(csv_path),
                 "--config", str(c.CONFIG_PATH)])
    assert rc == 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row.update(answer="KEEP", depth="L1", flag_no_basis="FALSE", flag_rigged="FALSE")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(a.HANDCODE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    capsys.readouterr()
    rc = a.main(["--kappa", str(csv_path), "--annotated", str(annotated),
                 "--config", str(c.CONFIG_PATH)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "joined on run_id: 4 rows" in out
    assert "confusion" in out.lower()


# ----------------------------------------------- integration-session tests ----
# The annotate CLI used to read `os.environ` directly while its own error
# message told the operator to "put it in .env", which silently did nothing.
# Both stages now share `contracts.load_api_key`.

def test_cli_finds_the_key_in_dotenv_not_only_the_environment(tmp_path, monkeypatch):
    """The message promises a .env fallback; this asserts there really is one."""
    env = tmp_path / ".env"
    env.write_text('OPENROUTER_API_KEY="sk-or-v1-from-dotenv"\n', encoding="utf-8")
    monkeypatch.setattr(c, "ENV_PATH", env)
    assert c.load_api_key() == "sk-or-v1-from-dotenv"


def test_runner_and_annotate_authenticate_through_the_same_loader():
    """Two stages, one answer to 'where is the key'. CONTRACTS.md §0."""
    from src import runner as r
    assert r.load_api_key is c.load_api_key
    # And no module keeps its own copy of the path: an import-time copy of a
    # value the loader resolves at call time can only ever go stale.
    assert not hasattr(r, "ENV_PATH")


def test_cli_refuses_when_neither_environment_nor_dotenv_has_a_key(
        tmp_path, monkeypatch, capsys):
    """The regression that matters: with a real .env on disk this test used to
    fall through to the operator's key and annotate 10 records for real."""
    monkeypatch.setattr(c, "ENV_PATH", tmp_path / "definitely-absent.env")
    rc = a.main(["--input", str(FIXTURE), "--annotated", str(tmp_path / "a.jsonl"),
                 "--config", str(c.CONFIG_PATH)])
    assert rc == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


# ------------------------------------------- the fatal-status seam finding ----
# annotate.py treated 403 as fatal and let FatalAnnotatorError escape
# asyncio.gather, which closed the output handle under up to `concurrency - 1`
# siblings whose responses were already paid for. runner.py had already solved
# both halves and documented why; the two modules disagreed.

_GOOD_ANNOTATION = {
    "choices": [{"message": {"content":
        '{"answer":"KEEP","depth":"L1","flag_no_basis":false,"flag_rigged":false}'}}],
    "model": "mistralai/mistral-nemo",
    "usage": {"cost": 3e-05},
}


def _ten_cot_records():
    return [{"run_id": f"{i:040x}", "mode": "cot",
             "response": {"choices": [{"message": {
                 "content": f"reasoning {i}\nKEEP"}}]}}
            for i in range(10)]


def _caller_where_the_third_call_returns(status, body):
    """The third of ten calls fails fast; the other nine return slowly. That
    ordering is what puts nine paid responses in flight when the failure lands."""
    state = {"n": 0}

    async def caller(prompt):
        state["n"] += 1
        if state["n"] == 3:
            await asyncio.sleep(0.01)
            return status, body
        await asyncio.sleep(0.08)
        return 200, _GOOD_ANNOTATION
    return caller, state


def _run_annotate_all(records, caller, out_path):
    return asyncio.run(a.annotate_all(
        records, caller=caller, out_path=out_path,
        annotator="mistralai/mistral-nemo", allowed_answers=None,
        concurrency=10, limit=None, json_mode=False, log=None))


def test_a_moderation_403_costs_one_record_not_the_whole_pass(tmp_path):
    """403 is per-request — a provider's moderation layer rejecting one prompt.
    The annotator is fed model reasoning about poisoning someone, so a flagged
    prompt is expected. runner.py:FATAL_STATUS excludes 403 for exactly this
    reason and says so; annotate.py included it."""
    out = tmp_path / "annotated.jsonl"
    caller, state = _caller_where_the_third_call_returns(
        403, {"error": {"message": "moderation", "code": 403}})
    stats = _run_annotate_all(_ten_cot_records(), caller, out)

    assert state["n"] == 10
    assert stats.annotated == 9          # nine codings kept
    assert stats.failed == 1             # the flagged one, retryable on rerun
    assert stats.fatal is None           # and the pass did NOT abort
    assert len(out.read_text().splitlines()) == 9


def test_403_is_not_fatal_and_402_is(tmp_path):
    """Matching runner.py, whose constant is the one with the reasoning."""
    assert 403 not in a.FATAL_STATUS
    assert 401 in a.FATAL_STATUS
    assert 402 in a.FATAL_STATUS
    from src import runner as r
    assert a.FATAL_STATUS == r.FATAL_STATUS


def test_a_fatal_401_still_writes_the_annotations_already_paid_for(tmp_path):
    """A bad key IS fatal — every remaining call fails identically. But it must
    not take the in-flight siblings with it: their responses are billed, and
    since a failed record writes no line, discarding them means every rerun
    makes the same partial progress and aborts in the same place."""
    out = tmp_path / "annotated.jsonl"
    caller, state = _caller_where_the_third_call_returns(
        401, {"error": {"message": "bad key", "code": 401}})

    with pytest.raises(a.FatalAnnotatorError) as excinfo:
        _run_annotate_all(_ten_cot_records(), caller, out)
    assert "401" in str(excinfo.value)

    # The operator still stops. Nothing paid for is lost.
    assert state["n"] == 10
    assert len(out.read_text().splitlines()) == 9
    for line in out.read_text().splitlines():
        assert json.loads(line)["answer"] == "KEEP"


def test_a_latched_fatal_stops_further_calls_being_paid_for(tmp_path):
    """Once a fatal status is seen, records not yet started must not be called:
    they would fail identically. Counted as `aborted`, not `failed`."""
    out = tmp_path / "annotated.jsonl"
    state = {"n": 0}

    async def caller(prompt):
        state["n"] += 1
        if state["n"] == 1:
            return 401, {"error": {"message": "bad key", "code": 401}}
        return 200, _GOOD_ANNOTATION

    with pytest.raises(a.FatalAnnotatorError):
        asyncio.run(a.annotate_all(
            _ten_cot_records(), caller=caller, out_path=out,
            annotator="mistralai/mistral-nemo", allowed_answers=None,
            concurrency=1, limit=None, json_mode=False, log=None))
    # Serialised, so the latch is seen before any sibling starts.
    assert state["n"] == 1
    assert not out.exists() or out.read_text() == ""


# ------------------------------------------------------------- depth floor ----
# mistral-nemo gave a depth code of L1 or L2 to all 44 zero-reasoning responses
# in the pilot and never once L0, including for a response that was the single
# token "KEEP". It infers the reasoning it expects to accompany an answer.

@pytest.mark.parametrize("text, words", [
    ("KEEP", 0),
    ("SWITCH", 0),
    ("SWITCH.", 0),
    ("**KEEP**", 0),
    ("K", 0),
    ('You must say "KEEP".', 3),
])
def test_a_response_that_is_only_the_answer_has_no_reasoning(text, words):
    assert a.codeable_reasoning_words(text) == words
    assert not a.has_codeable_reasoning(text)


def test_real_reasoning_is_codeable():
    text = ("Since the man knows which bottle is which, he kept the safe one "
            "for himself, so the one in my hand is the poison.\n\nSWITCH")
    assert a.has_codeable_reasoning(text)


def test_the_threshold_sits_in_the_measured_gap():
    """44 pilot responses had 0 non-answer words, one had 3, and the next had
    24. The threshold must fall strictly between 3 and 24 or it is not the
    rule the measurement supports."""
    assert 3 < a.DEPTH_FLOOR_MAX_WORDS <= 24


def _annotator_said(depth, answer="KEEP"):
    return json.dumps({"answer": answer, "depth": depth,
                       "flag_no_basis": False, "flag_rigged": False})


def test_depth_is_forced_to_L0_when_there_is_no_reasoning_to_code():
    rec = a.build_annotated_record(
        run_id="x" * 40, raw_output=_annotator_said("L2"),
        annotator_model="mistralai/mistral-nemo", response_text="KEEP")
    assert rec["depth"] == "L0"
    assert rec["depth_source"] == "no_reasoning_floor"
    # The answer still comes from the annotator, whose extraction is good.
    assert rec["answer"] == "KEEP"
    # And the override is auditable: what it actually claimed is still on record.
    assert '"depth": "L2"' in rec["raw_annotator_output"]


def test_the_annotators_depth_stands_when_there_is_reasoning():
    text = ("He kept the safe bottle for himself because he prepared them, so "
            "the one he handed me must be the poisoned one.\n\nSWITCH")
    rec = a.build_annotated_record(
        run_id="y" * 40, raw_output=_annotator_said("L1", "SWITCH"),
        annotator_model="mistralai/mistral-nemo", response_text=text)
    assert rec["depth"] == "L1"
    assert rec["depth_source"] == "annotator"


def test_no_override_is_recorded_when_the_annotator_already_said_L0():
    rec = a.build_annotated_record(
        run_id="z" * 40, raw_output=_annotator_said("L0"),
        annotator_model="mistralai/mistral-nemo", response_text="KEEP")
    assert rec["depth"] == "L0"
    assert rec["depth_source"] == "annotator"


def test_the_floor_is_off_when_no_response_text_is_supplied():
    """Callers that do not pass the response text keep the old behaviour, so
    the existing unit tests and any re-annotation path are unaffected."""
    rec = a.build_annotated_record(
        run_id="w" * 40, raw_output=_annotator_said("L2"),
        annotator_model="mistralai/mistral-nemo")
    assert rec["depth"] == "L2"
    assert rec["depth_source"] == "annotator"
