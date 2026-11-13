"""Regression tests for the scaffold contracts.

These guard the things three other sessions assume without checking: that the
scenario texts are still the ones PLAN.md specifies, that run_id formatting has
not drifted, and that direct mode differs from CoT in exactly one paragraph.

    python -m pytest tests/ -q
"""

import hashlib
import re

import pytest
import yaml

from src import contracts as c


# ------------------------------------------------------------- scenarios ----

@pytest.mark.parametrize("name", c.SCENARIOS)
def test_scenario_has_five_paragraphs(name):
    assert len(c.scenario_paragraphs(name)) == 5


@pytest.mark.parametrize("name", c.SCENARIOS)
def test_final_paragraph_is_the_answer_format(name):
    last = c.scenario_paragraphs(name)[4]
    assert last.startswith("End your response with a single line containing exactly one word:")
    assert "KEEP" in last and "SWITCH" in last


@pytest.mark.parametrize("name", c.SCENARIOS)
def test_randomise_clause_is_paragraph_four(name):
    assert c.scenario_paragraphs(name)[3] == c.RANDOMISE_CLAUSE[name][False]


def test_default_keep_and_switch_differ_only_in_the_offer():
    """PLAN.md §2: the game is identical; only which option requires an act
    changes. Paragraphs 1, 2 and 4 must be byte-identical."""
    k = c.scenario_paragraphs("default_keep")
    s = c.scenario_paragraphs("default_switch")
    assert k[0] == s[0]
    assert k[1] == s[1]
    assert k[3] == s[3]
    assert k[2] != s[2]          # the manipulation
    assert k[4] != s[4]          # its matching answer gloss


def test_scenarios_are_not_hard_wrapped():
    """Paragraphs are single lines. Hard wrapping at different columns would be
    an uncontrolled difference between the three conditions."""
    for name in c.SCENARIOS:
        for para in c.scenario_paragraphs(name):
            assert "\n" not in para


def test_no_scenario_mentions_sherlock():
    for name in c.SCENARIOS:
        text = c.load_scenario(name).lower()
        for tell in ("sherlock", "holmes", "study in pink", "moriarty", "cabbie"):
            assert tell not in text


# ----------------------------------------------------------- direct mode ----

@pytest.mark.parametrize("name", c.SCENARIOS)
def test_direct_instruction_matches_cot_glosses(name):
    """The single-letter instruction must carry the CoT parentheticals word for
    word, so direct and CoT differ only in answer format."""
    cot = c.scenario_paragraphs(name)[4]
    expected = (
        cot.replace(
            "End your response with a single line containing exactly one word:",
            "Answer with a single letter and nothing else:",
        )
        .replace("KEEP (", "K (")
        .replace("SWITCH (", "S (")
    )
    assert c.DIRECT_FINAL_INSTRUCTION[name] == expected


@pytest.mark.parametrize("name", c.SCENARIOS)
def test_direct_prompt_differs_from_cot_in_exactly_one_paragraph(name):
    direct = c.scenario_prompt(name, "direct").split("\n\n")
    cot = c.scenario_prompt(name, "cot").split("\n\n")
    assert len(direct) == len(cot) == 5
    assert [d != k for d, k in zip(direct, cot)] == [False, False, False, False, True]


def test_direct_request_constants():
    assert c.DIRECT_TEMPERATURE == 1.0        # never anything else
    assert c.DIRECT_MAX_TOKENS > 1            # 1 is brittle; see CONTRACTS.md §8
    assert c.DIRECT_TOP_LOGPROBS == 10


def test_answer_logprobs_sums_token_variants():
    import math
    choice = {"logprobs": {"content": [{"token": "K", "logprob": -0.1, "top_logprobs": [
        {"token": "K", "logprob": math.log(0.5)},
        {"token": " K", "logprob": math.log(0.2)},
        {"token": "(S", "logprob": math.log(0.05)},   # must NOT count
        {"token": "S", "logprob": math.log(0.25)},
    ]}]}}
    lps, status = c.answer_logprobs(choice)
    assert status == "ok"
    assert math.isclose(math.exp(lps["K"]), 0.7, abs_tol=1e-9)
    assert math.isclose(math.exp(lps["S"]), 0.25, abs_tol=1e-9)


def test_answer_logprobs_skips_a_misaligned_position():
    """mistral-nemo @ Parasail returns '</s>' at position 0 with no answer
    letters in its top-10. That must be reported, not silently mis-read."""
    import math
    choice = {"logprobs": {"content": [
        {"token": "</s>", "logprob": -0.04, "top_logprobs": [
            {"token": "</s>", "logprob": -0.04}, {"token": "\n\n", "logprob": -3.4}]},
    ]}}
    assert c.answer_logprobs(choice) == (None, "no_answer_tokens")


def test_answer_logprobs_reports_missing_logprobs():
    assert c.answer_logprobs({"logprobs": None}) == (None, "no_logprobs")
    assert c.answer_logprobs({}) == (None, "no_logprobs")


# --------------------------------------------------------------- run_id ----

def test_run_id_key_formatting():
    assert (c.run_id_key("m", "default_keep", "none", "cot", 0, 3)
            == "m|default_keep|none|cot|0.0|3")
    assert (c.run_id_key("m", "default_keep", "none", "cot", 1.20, 0)
            == "m|default_keep|none|cot|1.2|0")


def test_run_id_is_sha1_of_the_key():
    key = c.run_id_key("openai/gpt-4o-mini", "default_keep", "none", "direct", 1.0, 0)
    assert c.run_id("openai/gpt-4o-mini", "default_keep", "none", "direct", 1.0, 0) == \
        hashlib.sha1(key.encode("utf-8")).hexdigest()


def test_run_id_pinned_values():
    """Frozen so a formatting change cannot silently invalidate a half-finished run."""
    assert c.run_id("meta-llama/llama-3.1-8b-instruct", "default_switch", "expert",
                    "cot", 1.2, 7) == "660460d5e8cd0784ab3877242a11b57e43ee3ed9"
    assert c.run_id("openai/gpt-4o-mini", "default_keep", "none",
                    "direct", 1.0, 0) == "2820f375be14c6ebeaf684c1af8fc60accaedf4f"


def test_run_ids_are_unique_across_the_full_grid():
    cfg = yaml.safe_load(c.CONFIG_PATH.read_text(encoding="utf-8"))
    ids = set()
    n = 0
    for m in cfg["models"]:
        for sc in cfg["scenarios"]:
            for p in cfg["personas"]:
                for t in cfg["temperatures"]:
                    for r in range(cfg["replicates_cot"]):
                        ids.add(c.run_id(m["id"], sc, p, "cot", t, r)); n += 1
                ids.add(c.run_id(m["id"], sc, p, "direct", c.DIRECT_TEMPERATURE, 0)); n += 1
            ids.add(c.run_id(m["id"], sc, "none", c.CONTAMINATION_MODE, 1.0, 0)); n += 1
    assert len(ids) == n


# ------------------------------------------------------------- messages ----

def test_persona_none_emits_no_system_message():
    msgs = c.build_messages("default_keep", "none", "cot")
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    assert "none" not in c.PERSONA_SYSTEM_PROMPTS


@pytest.mark.parametrize("persona", ["average_person", "expert"])
def test_persona_prompts_verbatim(persona):
    msgs = c.build_messages("default_keep", persona, "cot")
    assert msgs[0] == {"role": "system", "content": c.PERSONA_SYSTEM_PROMPTS[persona]}
    assert msgs[1]["role"] == "user"


def test_persona_prompt_text():
    assert c.PERSONA_SYSTEM_PROMPTS["average_person"] == (
        "You are an ordinary person with no special training in logic, game "
        "theory, or psychology. Answer as you actually would.")
    assert c.PERSONA_SYSTEM_PROMPTS["expert"] == (
        "You are an expert game theorist. Answer as you actually would.")


def test_randomise_toggle_changes_only_paragraph_four():
    off = c.scenario_prompt("default_keep", "cot", False).split("\n\n")
    on = c.scenario_prompt("default_keep", "cot", True).split("\n\n")
    assert [a != b for a, b in zip(off, on)] == [False, False, False, True, False]
    assert "random" not in on[3].lower()


def test_contamination_prompt_drops_the_answer_format():
    p = c.contamination_prompt("isomorph")
    assert p.startswith("Read the following scenario.")
    assert "End your response with" not in p
    assert len(p.split("\n\n")) == 5     # question + 4 scenario paragraphs


# --------------------------------------------------------------- config ----

def test_config_matches_the_documented_schema():
    cfg = yaml.safe_load(c.CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(cfg["experiment"], str)
    assert isinstance(cfg["seed"], int)
    assert set(cfg["scenarios"]) <= set(c.SCENARIOS)
    assert set(cfg["personas"]) <= set(c.PERSONAS)
    assert set(cfg["answer_modes"]) <= set(c.MODES)
    assert all(isinstance(t, float) for t in cfg["temperatures"])
    assert isinstance(cfg["replicates_cot"], int)
    assert set(cfg["allowed_answers"]) <= set(c.ANSWERS)
    assert isinstance(cfg["allow_randomise_in_prompt"], bool)
    assert isinstance(cfg["annotator"], str)
    assert isinstance(cfg["max_concurrency"], int)
    for m in cfg["models"]:
        assert set(m) == {"id", "tier", "logprobs"}
        assert m["tier"] in {"small", "mid"}
        assert isinstance(m["logprobs"], bool)
        assert re.match(r"^[a-z0-9~._-]+/[a-z0-9.:_-]+$", m["id"]), m["id"]


def test_every_configured_scenario_has_a_file():
    cfg = yaml.safe_load(c.CONFIG_PATH.read_text(encoding="utf-8"))
    for name in cfg["scenarios"]:
        assert (c.SCENARIO_DIR / f"{name}.txt").exists()


def test_curve_models_are_a_subset_of_logprobs_capable_models():
    cfg = yaml.safe_load(c.CONFIG_PATH.read_text(encoding="utf-8"))
    capable = {m["id"] for m in cfg["models"] if m["logprobs"]}
    assert c.DIRECT_CURVE_MODELS <= capable
    assert set(c.DIRECT_PROVIDER_PIN) == set(c.DIRECT_CURVE_MODELS)


def test_enums_are_verbatim():
    assert c.ANSWERS == ("KEEP", "SWITCH", "RANDOMISE", "THIRD_OPTION", "REFUSE", "UNPARSEABLE")
    assert c.DEPTH_CODES == ("L0", "L1", "L2", "L3", "MIX", "NON")


# ------------------------------------------- post-review contract guards ----

def test_depth_rubric_defines_both_flags():
    """PLAN.md §8.4's JSON schema asks for flag_no_basis and flag_rigged; the
    rubric pasted into the prompt must define them or the annotator is guessing."""
    assert "flag_no_basis" in c.DEPTH_RUBRIC
    assert "flag_rigged" in c.DEPTH_RUBRIC
    assert "no rational basis for choosing" in c.DEPTH_RUBRIC
    assert "suspects a trick" in c.DEPTH_RUBRIC
    for code in c.DEPTH_CODES:
        assert f"`{code}`" in c.DEPTH_RUBRIC


def test_direct_validation_is_declared_but_not_in_the_grid():
    """It must be a real mode (so build_messages works) but never a config
    answer_mode, or it would be emitted as part of the factorial grid."""
    assert c.DIRECT_VALIDATION_MODE in c.ALL_MODES
    assert c.DIRECT_VALIDATION_MODE not in c.MODES
    assert c.CONTAMINATION_MODE not in c.MODES


@pytest.mark.parametrize("name", c.SCENARIOS)
def test_direct_validation_prompt_is_the_direct_prompt(name):
    assert c.scenario_prompt(name, c.DIRECT_VALIDATION_MODE) == c.scenario_prompt(name, "direct")


def test_unknown_mode_still_rejected():
    with pytest.raises(KeyError):
        c.scenario_prompt("default_keep", "chain_of_thought")


def test_randomise_clause_entries_are_not_aliased():
    assert c.RANDOMISE_CLAUSE["default_switch"] is not c.RANDOMISE_CLAUSE["default_keep"]
    assert c.RANDOMISE_CLAUSE["default_switch"] == c.RANDOMISE_CLAUSE["default_keep"]


def test_is_usable_response_rejects_http_200_error_bodies():
    """Measured on OpenRouter: a provider failure comes back as HTTP 200 with an
    error object and no choices. Writing it would poison resume permanently."""
    assert c.is_usable_response(
        200, {"id": "gen-1", "error": {"message": "Provider returned error", "code": 429}}
    ) == (False, "error_body_429")
    assert c.is_usable_response(200, {"id": "gen-1"}) == (False, "no_choices")
    assert c.is_usable_response(200, {"choices": []}) == (False, "no_choices")
    assert c.is_usable_response(200, {"choices": [{"message": {"content": None}}]}) \
        == (False, "no_message_content")
    assert c.is_usable_response(429, {"choices": [{"message": {"content": "K"}}]})[0] is False
    assert c.is_usable_response(200, {"choices": [{"message": {"content": "K"}}]}) == (True, "ok")


def test_direct_temperature_carveout_is_documented():
    """The 'temperature is always 1.0' rule and the validation exception must
    both be stated, or the two halves of §8 contradict each other."""
    doc = (c.ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
    assert "with exactly one" in doc and "named exception" in doc
    assert "direct_validation" in doc


# ----------------------------------------- measured model-selection guards ----

def test_every_configured_model_has_a_direct_provider_pin():
    """A direct-mode job without a pin gets routed to whichever provider is free,
    and several advertise logprobs without delivering them (CONTRACTS.md §8)."""
    cfg = yaml.safe_load(c.CONFIG_PATH.read_text(encoding="utf-8"))
    for m in cfg["models"]:
        if m["logprobs"]:
            assert m["id"] in c.DIRECT_PROVIDER_PIN, f"{m['id']} has no pin"
            pin = c.DIRECT_PROVIDER_PIN[m["id"]]
            assert pin["allow_fallbacks"] is False
            assert pin["only"], "empty provider list"


def test_no_pin_uses_require_parameters():
    """Measured NOT to work: it routed mistral-nemo to a provider returning
    no logprobs at all. Only an explicit `only` list is reliable."""
    for pin in c.DIRECT_PROVIDER_PIN.values():
        assert "require_parameters" not in pin


def test_cot_max_tokens_is_large_enough_to_reach_the_final_line():
    """At 900, verbose models were truncated mid-answer, destroying the
    final-line answer the scenario asks for."""
    assert c.COT_MAX_TOKENS >= 2000


def test_no_cot_models_are_flagged():
    """A model emitting no visible reasoning cannot be depth-coded; the
    analysis must be able to find out which ones without re-probing."""
    assert c.NO_COT_MODELS, "expected at least one known no-CoT model"
    assert c.NO_COT_MODELS <= set(c.COT_CONTENT_CHARS)
    for m in c.NO_COT_MODELS:
        assert c.COT_CONTENT_CHARS[m] < 120


def test_cot_content_chars_covers_every_configured_model():
    cfg = yaml.safe_load(c.CONFIG_PATH.read_text(encoding="utf-8"))
    for m in cfg["models"]:
        assert m["id"] in c.COT_CONTENT_CHARS, f"{m['id']} unmeasured"


def test_annotator_is_not_required_to_produce_reasoning():
    """mistral-nemo emits ~6 chars of content, which disqualifies it as a
    SUBJECT but is irrelevant for annotation (strict JSON only)."""
    cfg = yaml.safe_load(c.CONFIG_PATH.read_text(encoding="utf-8"))
    subject_ids = {m["id"] for m in cfg["models"]}
    assert cfg["annotator"] not in subject_ids


# -------------------------------------- guards from the consistency review ----

def test_direct_pins_name_exactly_one_provider():
    """OpenRouter's `only` is a permitted SET, not a preference order: a
    two-name list still lets the router choose either and reintroduces the
    provider mixing the pin exists to prevent."""
    for model, pin in c.DIRECT_PROVIDER_PIN.items():
        assert len(pin["only"]) == 1, f"{model} pins {len(pin['only'])} providers"
        assert pin["allow_fallbacks"] is False


def test_alternates_never_overlap_the_pin():
    for model, alts in c.DIRECT_PROVIDER_ALTERNATES.items():
        assert model in c.DIRECT_PROVIDER_PIN
        assert c.DIRECT_PROVIDER_PIN[model]["only"][0] not in alts


def test_docs_never_hardcode_a_stale_max_tokens():
    """CONTRACTS.md §3 shipped `900` for cot after COT_MAX_TOKENS became 2000.
    Truncation is silent and run_id does not cover max_tokens, so resume would
    not re-run the grid after the correction."""
    doc = (c.ROOT / "CONTRACTS.md").read_text(encoding="utf-8")
    assert '"max_tokens": 900' not in doc
    assert "`900` cot" not in doc
    assert f'"max_tokens": {c.COT_MAX_TOKENS}' in doc


def test_docs_do_not_claim_a_stale_model_count():
    """Counts like "three of five models fail" and "two models, not five" went
    stale the moment the model set changed. Derive from the constants."""
    n = len(c.DIRECT_CURVE_MODELS)
    for name in ("CONTRACTS.md", "claude_code_prompts.md"):
        doc = (c.ROOT / name).read_text(encoding="utf-8")
        for stale in ("Three of\nfive models fail", "three of the\n    five configured models fail",
                      "currently two of five", "exists for two models, not five"):
            assert stale not in doc, f"{name}: stale claim {stale!r}"
    src = (c.ROOT / "src" / "contracts.py").read_text(encoding="utf-8")
    assert "three of the five" not in src
    assert n == 5


def test_prompts_carry_the_constants_the_sessions_must_import():
    """Each downstream prompt must name the constant, or its session will
    re-derive the value and drift."""
    doc = (c.ROOT / "claude_code_prompts.md").read_text(encoding="utf-8")
    for const in ("contracts.DEPTH_RUBRIC", "contracts.answer_logprobs",
                  "contracts.DIRECT_PROVIDER_PIN", "contracts.DIRECT_CURVE_MODELS",
                  "contracts.NO_COT_MODELS", "contracts.is_usable_response",
                  "COT_MAX_TOKENS"):
        assert const in doc, f"no prompt mentions {const}"
