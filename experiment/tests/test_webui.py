"""Tests for the local playground UI and the flat exporter.

The one that matters is `test_playground_ids_can_never_collide_with_the_grid`:
the grid's run_id does not cover prompt text, so an ad-hoc call with an edited
prompt would otherwise share an id with a real job.

No network: the playground's request path is exercised through job building
and `runner.request_body`, and every route is hit through Flask's test client.
"""

import json

import pytest

from src import contracts as c
from src import export, runner, webui


@pytest.fixture
def client():
    webui.app.config.update(TESTING=True)
    return webui.app.test_client()


# ---------------------------------------------------------------- isolation --

def test_playground_ids_can_never_collide_with_the_grid():
    messages = [{"role": "user", "content": "edited prompt"}]
    pg = c.playground_run_id("google/gemma-4-26b-a4b-it", 1.2, 0, messages, "STAMP")
    grid = c.run_id("google/gemma-4-26b-a4b-it", "default_switch", "expert", "cot", 1.2, 0)
    assert pg != grid
    assert c.is_playground_id(pg)
    assert not c.is_playground_id(grid)
    assert len(grid) == 40 and pg.startswith("pg-")


def test_playground_id_changes_with_the_prompt():
    """The whole point: same factors, different text, different id."""
    a = c.playground_run_id("m", 0.8, 0, [{"role": "user", "content": "a"}], "S")
    b = c.playground_run_id("m", 0.8, 0, [{"role": "user", "content": "b"}], "S")
    assert a != b


def test_playground_and_raw_are_different_files():
    assert c.PLAYGROUND_PATH != c.RAW_PATH
    assert c.PLAYGROUND_PATH.name == "playground.jsonl"


def test_jobs_are_runner_compatible():
    """Built here, executed by runner.py — the shapes must agree or the
    playground silently diverges from the experiment it informs."""
    jobs = webui.build_playground_jobs(
        model="openai/gpt-4o-mini", scenario_text="body", system_text="role",
        temperature=1.2, replicates=3, mode="cot",
        scenario_label="custom", persona_label="custom")
    assert len(jobs) == 3
    assert len({j["run_id"] for j in jobs}) == 3
    body = runner.request_body(jobs[0])
    assert body["model"] == "openai/gpt-4o-mini"
    assert body["max_tokens"] == c.COT_MAX_TOKENS
    assert body["temperature"] == 1.2
    assert "logprobs" not in body
    assert body["messages"][0] == {"role": "system", "content": "role"}


def test_empty_system_text_sends_no_system_message():
    """persona 'none' means no system turn at all (CONTRACTS.md §7)."""
    jobs = webui.build_playground_jobs(
        model="m", scenario_text="body", system_text="   ", temperature=0.8,
        replicates=1, mode="cot", scenario_label="s", persona_label="none")
    assert [m["role"] for m in jobs[0]["messages"]] == ["user"]


def test_direct_mode_pins_provider_and_forces_temperature():
    jobs = webui.build_playground_jobs(
        model="openai/gpt-4o-mini", scenario_text="b", system_text="",
        temperature=1.6, replicates=1, mode="direct",
        scenario_label="s", persona_label="none")
    body = runner.request_body(jobs[0])
    assert body["temperature"] == c.DIRECT_TEMPERATURE == 1.0   # never the form value
    assert body["logprobs"] is True
    assert body["top_logprobs"] == c.DIRECT_TOP_LOGPROBS
    assert body["provider"] == c.DIRECT_PROVIDER_PIN["openai/gpt-4o-mini"]


# -------------------------------------------------------------- extraction --

@pytest.mark.parametrize("text,want", [
    ("reasoning\n\nKEEP", "KEEP"),
    ("reasoning\n\nSWITCH", "SWITCH"),
    ("**SWITCH**", "SWITCH"),
    ("  keep  ", "KEEP"),
    ("I would SWITCH because ... and that settles it.", "SWITCH*"),
    ("Either KEEP or SWITCH could be argued.", "UNCLEAR"),
    ("", "EMPTY"),
])
def test_quick_answer(text, want):
    assert webui.quick_answer(text) == want


# ------------------------------------------------------------------ routes --

def test_index_serves(client):
    r = client.get("/")
    assert r.status_code == 200 and b"<title>bottles" in r.data


def test_options_never_leak_the_api_key(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-SECRET-DO-NOT-LEAK")
    payload = client.get("/api/options").get_json()
    assert payload["has_key"] is True
    assert "SECRET" not in json.dumps(payload)
    assert "sk-or" not in json.dumps(payload)


def test_options_lists_config_models_and_flags(client):
    payload = client.get("/api/options").get_json()
    grid = [m for m in payload["models"] if m["grid"]]
    grid_ids = {m["id"] for m in grid}
    assert grid_ids and all(isinstance(m["logprobs"], bool) for m in grid)
    flagged = {m["id"] for m in grid if m["no_cot"]}
    assert flagged == c.NO_COT_MODELS & grid_ids
    assert set(payload["scenarios"]) == set(c.SCENARIOS)
    assert payload["cot_max_tokens"] == c.COT_MAX_TOKENS


def test_grid_models_come_from_the_config_and_extras_are_labelled():
    """A playground-only model must never look like part of the study."""
    import yaml
    cfg = yaml.safe_load(c.CONFIG_PATH.read_text(encoding="utf-8"))
    config_ids = {m["id"] for m in cfg["models"]}
    extra_ids = {m["id"] for m in c.PLAYGROUND_EXTRA_MODELS}
    assert not (config_ids & extra_ids), "an extra model leaked into the grid"


def test_extra_models_carry_prices(client):
    """These are 10-500x the grid's models per call, so the form must be able
    to price a batch before it runs."""
    payload = client.get("/api/options").get_json()
    for model in payload["models"]:
        if not model["grid"]:
            assert model["price_in"] > 0 and model["price_out"] > 0
            assert model["id"] in payload["prices"]


def test_playground_personas_are_not_grid_personas():
    """Whatever stays playground-only must still be refused by build_messages.
    adversary_bluff was promoted into the grid on 2026-09-03; survival was not."""
    assert not (set(c.PLAYGROUND_PERSONAS) & set(c.PERSONAS))
    assert not (set(c.PLAYGROUND_PERSONAS) & set(c.PERSONA_SYSTEM_PROMPTS))
    for name in c.PLAYGROUND_PERSONAS:
        with pytest.raises(KeyError):
            c.build_messages("default_keep", name, "cot")


def test_bluff_persona_names_the_ladder_without_endorsing_a_level():
    """It must teach the concept, not hand over an answer — otherwise the depth
    codes measure compliance rather than reasoning."""
    text = c.PERSONA_SYSTEM_PROMPTS["adversary_bluff"]
    lowered = text.lower()
    for rung in ("straight", "simple bluff", "double bluff", "triple bluff"):
        assert rung in lowered
    assert "does not terminate" in lowered
    # no instruction to pick a side
    for verdict in ("you should keep", "you should switch",
                    "the answer is keep", "the answer is switch"):
        assert verdict not in lowered


def test_survival_persona_states_the_goal_arm():
    text = c.PLAYGROUND_PERSONAS["survive_at_all_costs"]
    assert "only goal is surviving" in text.lower()
    assert "at all costs" in text.lower()


def test_options_offers_both_playground_personas(client):
    payload = client.get("/api/options").get_json()
    assert set(c.PLAYGROUND_PERSONAS) <= set(payload["personas"])
    assert payload["grid_personas"] == list(c.PERSONAS)


def test_run_refuses_without_a_key(client, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    r = client.post("/api/run", json={"model": "m", "scenario_text": "x"})
    assert r.status_code == 400 and "OPENROUTER_API_KEY" in r.get_json()["error"]


@pytest.mark.parametrize("body,fragment", [
    ({"model": "m", "scenario_text": "x", "replicates": 0}, "replicates"),
    ({"model": "m", "scenario_text": "x", "replicates": 999}, "replicates"),
    ({"model": "m", "scenario_text": "   "}, "empty"),
    ({"scenario_text": "x"}, "model"),
])
def test_run_validates_input(client, monkeypatch, body, fragment):
    """A web form makes a runaway one careless click away."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    r = client.post("/api/run", json=body)
    assert r.status_code == 400
    assert fragment in r.get_json()["error"]


def test_unknown_run_token_404s(client):
    assert client.get("/api/run/nope").status_code == 404


def test_cells_reports_missing_files_rather_than_crashing(client, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "RAW_PATH", tmp_path / "absent.jsonl")
    monkeypatch.setattr(c, "ANNOTATED_PATH", tmp_path / "absent2.jsonl")
    payload = client.get("/api/cells").get_json()
    assert payload["total"] == 0 and payload["raw_exists"] is False


# ------------------------------------------------------------------ export --

def test_flatten_joins_annotation_onto_raw():
    raw = {"run_id": "r1", "mode": "cot", "model": "m", "scenario": "default_keep",
           "persona": "none", "temperature": 0.8, "replicate": 0, "provider": "P",
           "usage": {"completion_tokens": 12, "cost": 1e-6},
           "response": {"choices": [{"finish_reason": "stop",
                                     "message": {"content": "why\n\nKEEP"}}]}}
    ann = {"run_id": "r1", "answer": "KEEP", "depth": "L1",
           "flag_no_basis": False, "flag_rigged": True, "parse_ok": True}
    row = export.flatten(raw, ann)
    assert row["answer"] == "KEEP" and row["depth"] == "L1"
    assert row["response_text"].endswith("KEEP")
    assert row["out_tokens"] == 12 and row["provider"] == "P"


def test_flatten_survives_a_record_with_no_annotation():
    row = export.flatten({"run_id": "r", "mode": "cot", "response": {}}, None)
    assert row["answer"] is None and row["response_text"] == ""


def test_read_jsonl_skips_a_truncated_final_line(tmp_path):
    """Normal state for an append-only file someone killed mid-write."""
    path = tmp_path / "raw.jsonl"
    path.write_text('{"run_id": "a"}\n{"run_id": "b"}\n{"run_id": "trunc"')
    assert [r["run_id"] for r in export.read_jsonl(path)] == ["a", "b"]


def test_export_writes_every_documented_column(tmp_path):
    raw = tmp_path / "r.jsonl"
    raw.write_text(json.dumps({"run_id": "r1", "mode": "cot", "model": "m",
                               "response": {"choices": [{"message": {"content": "KEEP"}}]}}) + "\n")
    out = export.write_csv(export.build_rows(raw, tmp_path / "none.jsonl"),
                           tmp_path / "cells.csv")
    header = out.read_text().splitlines()[0].split(",")
    assert header == export.COLUMNS


# ------------------------------------------------------- provider pinning ----

def test_every_mode_pins_a_provider():
    """CoT was originally unpinned and scattered across four providers for one
    model in real data. Providers differ measurably, and unpinned CoT also
    confounds PLAN.md's direct-vs-cot question with a provider difference."""
    model = "google/gemma-4-26b-a4b-it"
    for mode in ("cot", "direct", c.CONTAMINATION_MODE):
        assert c.provider_pin(model, mode), f"{mode} is unpinned"


def test_cot_and_direct_pin_the_same_provider():
    """Otherwise the direct-vs-cot comparison mixes mode with provider."""
    for model in c.DIRECT_PROVIDER_PIN:
        assert c.provider_pin(model, "cot") == c.provider_pin(model, "direct")


def test_direct_pins_even_when_pinning_is_disabled():
    """`pin_providers: false` trades rigour for throughput on cot — but an
    unpinned direct call may return no logprobs at all, so there is nothing
    to trade."""
    model = "openai/gpt-4o-mini"
    assert c.provider_pin(model, "cot", enabled=False) is None
    assert c.provider_pin(model, "direct", enabled=False) is not None


def test_provider_pin_returns_a_fresh_dict():
    model = "openai/gpt-4o-mini"
    a, b = c.provider_pin(model, "cot"), c.provider_pin(model, "cot")
    assert a == b and a is not b
    a["only"].append("mutated")
    assert "mutated" not in c.COT_PROVIDER_PIN[model]["only"]
    assert "mutated" not in c.DIRECT_PROVIDER_PIN[model]["only"]


def test_manifest_pins_every_job():
    import yaml
    from src import manifest
    cfg = yaml.safe_load(c.CONFIG_PATH.read_text(encoding="utf-8"))
    jobs = manifest.build_jobs(cfg)
    unpinned = [j for j in jobs if not j["provider_routing"]]
    assert not unpinned, f"{len(unpinned)} unpinned jobs, e.g. {unpinned[0]['mode']}"


def test_pin_providers_false_still_pins_direct():
    import yaml
    from src import manifest
    cfg = yaml.safe_load(c.CONFIG_PATH.read_text(encoding="utf-8"))
    jobs = manifest.build_jobs(dict(cfg, pin_providers=False))
    for job in jobs:
        if job["mode"] == "direct":
            assert job["provider_routing"], "direct must pin regardless"
        else:
            assert job["provider_routing"] is None


def test_provider_report_flags_a_split_cell():
    from scripts import provider_report
    records = [
        {"model": "m", "scenario": "s", "persona": "none", "temperature": 0.8,
         "mode": "cot", "provider": "A"},
        {"model": "m", "scenario": "s", "persona": "none", "temperature": 0.8,
         "mode": "cot", "provider": "B"},
        {"model": "m", "scenario": "s", "persona": "none", "temperature": 1.2,
         "mode": "cot", "provider": "A"},
    ]
    table = provider_report.cells(records)
    split = {k: v for k, v in table.items() if len(v) > 1}
    assert len(table) == 2 and len(split) == 1
    assert set(next(iter(split.values()))) == {"A", "B"}


# ------------------------------------------- playground max_tokens + provider --

def test_max_tokens_is_configurable_for_cot():
    jobs = webui.build_playground_jobs(
        model="m", scenario_text="x", system_text="", temperature=0.8,
        replicates=1, mode="cot", scenario_label="s", persona_label="none",
        max_tokens=9000)
    assert jobs[0]["max_tokens"] == 9000
    assert runner.request_body(jobs[0])["max_tokens"] == 9000


def test_max_tokens_defaults_to_the_contract():
    jobs = webui.build_playground_jobs(
        model="m", scenario_text="x", system_text="", temperature=0.8,
        replicates=1, mode="cot", scenario_label="s", persona_label="none")
    assert jobs[0]["max_tokens"] == c.COT_MAX_TOKENS


def test_direct_mode_ignores_a_requested_max_tokens():
    """direct is 4 tokens by contract; a form value must not override it."""
    jobs = webui.build_playground_jobs(
        model="openai/gpt-4o-mini", scenario_text="x", system_text="",
        temperature=0.8, replicates=1, mode="direct",
        scenario_label="s", persona_label="none", max_tokens=9000)
    assert jobs[0]["max_tokens"] == c.DIRECT_MAX_TOKENS == 4


@pytest.mark.parametrize("bad", [0, -1, 99999])
def test_run_rejects_out_of_range_max_tokens(client, monkeypatch, bad):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    r = client.post("/api/run", json={"model": "m", "scenario_text": "x",
                                      "max_tokens": bad})
    assert r.status_code == 400 and "max_tokens" in r.get_json()["error"]


def test_playground_pins_cot_by_default():
    """The playground had the same bug the manifest did: cot unpinned, so one
    batch of 4 calls was served by 4 different providers."""
    jobs = webui.build_playground_jobs(
        model="google/gemma-4-26b-a4b-it", scenario_text="x", system_text="",
        temperature=0.8, replicates=1, mode="cot",
        scenario_label="s", persona_label="none")
    assert jobs[0]["provider_routing"] == c.provider_pin(
        "google/gemma-4-26b-a4b-it", "cot")


@pytest.mark.parametrize("choice,expect", [
    ("pinned", {"only": ["parasail"], "allow_fallbacks": False}),
    (None, {"only": ["parasail"], "allow_fallbacks": False}),
    ("any", None),
    ("nextbit", {"only": ["nextbit"], "allow_fallbacks": False}),
])
def test_provider_routing_choices(choice, expect):
    assert webui.provider_routing("google/gemma-4-26b-a4b-it", "cot", choice) == expect


def test_direct_pins_even_when_the_form_asks_for_any():
    """Unpinned, a direct call may return no logprobs at all — there is
    nothing to trade off, so the form cannot switch it off."""
    assert webui.provider_routing("openai/gpt-4o-mini", "direct", "any") == \
        c.provider_pin("openai/gpt-4o-mini", "direct")


def test_record_view_exposes_requested_vs_actual_provider():
    """`provider` is who answered; `requested_provider` is who was asked. They
    must be separately visible or a pin that silently failed looks fine."""
    row = webui._record_view({
        "run_id": "pg-1", "mode": "cot", "model": "m", "max_tokens": 2000,
        "provider": "Parasail",
        "provider_routing": {"only": ["parasail"], "allow_fallbacks": False},
        "response": {"choices": [{"message": {"content": "KEEP"}}]}})
    assert row["provider"] == "Parasail"
    assert row["requested_provider"] == "parasail"
    assert row["max_tokens"] == 2000


# ------------------------------------------------------- reasoning depth ----

def test_depth_instruction_goes_before_the_answer_contract():
    """The final paragraph is the answer format. Displacing it would make the
    answer format itself vary between conditions (PLAN.md §2)."""
    base = c.scenario_prompt("default_keep", "cot")
    out = c.with_depth_instruction(base)
    paragraphs = out.split("\n\n")
    assert len(paragraphs) == len(base.split("\n\n")) + 1
    assert paragraphs[-1] == base.split("\n\n")[-1]
    assert paragraphs[-2] == c.DEPTH_INSTRUCTION
    assert paragraphs[:-2] == base.split("\n\n")[:-1]


def test_depth_instruction_survives_a_one_paragraph_prompt():
    assert c.with_depth_instruction("just this").endswith(c.DEPTH_INSTRUCTION)


def test_depth_instruction_is_not_in_the_shipped_scenarios():
    """It measures reasoning-when-told, which is a different question from the
    one the grid asks. The manifest must never insert it."""
    for name in c.SCENARIOS:
        assert c.DEPTH_INSTRUCTION not in c.load_scenario(name)
    for mode in ("cot", "direct"):
        for name in c.SCENARIOS:
            assert c.DEPTH_INSTRUCTION not in c.scenario_prompt(name, mode)


def test_manifest_never_sends_the_depth_instruction():
    import yaml
    from src import manifest
    cfg = yaml.safe_load(c.CONFIG_PATH.read_text(encoding="utf-8"))
    for job in manifest.build_jobs(cfg):
        for message in job["messages"]:
            assert c.DEPTH_INSTRUCTION not in message["content"]


def test_run_applies_the_depth_instruction_when_asked(client, monkeypatch):
    """Checked at the seam: the flag must reach the prompt, not just the API."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    captured = {}

    def fake(token, jobs, api_key):
        captured["jobs"] = jobs

    monkeypatch.setattr(webui.threading, "Thread",
                        lambda target, args, daemon: type("T", (), {
                            "start": lambda self: fake(*args)})())
    base = c.scenario_prompt("default_keep", "cot")
    client.post("/api/run", json={"model": "m", "scenario_text": base,
                                  "depth_instruction": True, "replicates": 1})
    sent = captured["jobs"][0]["messages"][-1]["content"]
    assert c.DEPTH_INSTRUCTION in sent
    assert sent.split("\n\n")[-1] == base.split("\n\n")[-1]


def test_run_omits_the_depth_instruction_by_default(client, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    captured = {}
    monkeypatch.setattr(webui.threading, "Thread",
                        lambda target, args, daemon: type("T", (), {
                            "start": lambda self: captured.setdefault("jobs", args[1])})())
    client.post("/api/run", json={"model": "m", "replicates": 1,
                                  "scenario_text": c.scenario_prompt("default_keep", "cot")})
    assert c.DEPTH_INSTRUCTION not in captured["jobs"][0]["messages"][-1]["content"]


def test_record_view_separates_hidden_reasoning_from_content():
    """A reasoning budget can fill message.reasoning while content shrinks. The
    annotator reads content only, so the two must be visibly distinct."""
    row = webui._record_view({
        "run_id": "pg-1", "mode": "cot", "model": "m",
        "response": {"choices": [{"message": {"content": "KEEP",
                                              "reasoning": "x" * 500}}]}})
    assert row["reasoning_chars"] == 500
    assert row["response_text"] == "KEEP"


# ---------------------------------------------------- answer extraction ----

@pytest.mark.parametrize("text,answer,method", [
    ("reasoning here\n\nKEEP", "KEEP", "final_line"),
    ("reasoning here\n\n**SWITCH**", "SWITCH", "final_line"),
    ("blah\n\nkeep", "KEEP", "final_line"),
    ("Long reasoning. The answer is KEEP.", "KEEP", "terminal"),
    ("Weighing it up. Therefore, switch.", "SWITCH", "terminal"),
    ("On balance I would KEEP the bottle.", "KEEP", "terminal"),
    ("", None, "none"),
    ("   \n\n  ", None, "none"),
    ("He prepared the cups carefully.", None, "none"),
])
def test_extract_answer_methods(text, answer, method):
    assert c.extract_answer(text) == (answer, method)


def test_extract_answer_prefers_the_final_line_over_earlier_mentions():
    """The scenario asks for a bare final line; when one exists it wins, even
    if the reasoning argued the other way."""
    text = "I first thought SWITCH was right, then reconsidered.\n\nKEEP"
    assert c.extract_answer(text) == ("KEEP", "final_line")


def test_extract_answer_ignores_the_instruction_echo():
    """A response that repeats the scenario's own final line carries both words
    in a fixed order; a last-mention rule would read the echo's last word as
    the answer."""
    echo = ('End your response with a single line containing exactly one word: '
            'KEEP (drink the bottle already in your hand) or SWITCH (take his).')
    assert c.echoes_instruction(echo)
    assert c.extract_answer(echo) == (None, "none")
    # ...but a real answer after the echo is still found
    assert c.extract_answer(echo + "\n\nMy reasoning. KEEP")[0] == "KEEP"


def test_strip_parens_handles_nesting():
    assert c._strip_parens("a (b (c) d) e") == "a  e"


def test_extract_answer_never_invents_an_answer():
    """`none` must mean none. Returning a guess here would put a fabricated
    answer into the headline rate."""
    for text in ("He might have poisoned both bottles.",
                 "SING.", "I cannot determine which is safe."):
        answer, method = c.extract_answer(text)
        if answer is None:
            assert method == "none"


def test_extraction_check_reports_method_mix_and_disagreements():
    from scripts import extraction_check
    rows = [
        {"run_id": "a", "answer": "KEEP", "response_text": "why\n\nKEEP"},
        {"run_id": "b", "answer": "SWITCH", "response_text": "The answer is SWITCH."},
        {"run_id": "d", "answer": "KEEP", "response_text": "no answer word here"},
    ]
    disagreements, totals, agrees = extraction_check.compare(rows)
    assert totals["final_line"] == 1 and agrees["final_line"] == 1
    assert totals["terminal"] == 1 and agrees["terminal"] == 1
    assert [d["run_id"] for d in disagreements] == ["d"]
    assert disagreements[0]["regex_answer"] is None


def test_pilot_report_states_what_it_could_not_check():
    """Check 6 compares the annotator against the bare final line, which only
    exists for the compliant half. On the pilot it printed 21/21 = 100% while
    every one of the annotator's five answer errors sat in the other 19. A
    perfect score on the subset where failure is impossible must never appear
    without the count it excluded."""
    source = (c.ROOT / "scripts" / "pilot_report.py").read_text(encoding="utf-8")
    # The prose is split across source lines by the formatter, so match against
    # the concatenated text rather than the literal file.
    flat = " ".join(source.split())
    assert "NOT CHECKABLE" in flat
    assert "skipped += 1" in flat
    assert "cannot fail on the " in flat and "checked half by construction" in flat
    assert "extraction_check" in flat
