"""Tests for src/manifest.py — the job grid and the pre-flight cost estimate.

No network. `estimate_cost` is exercised against a synthetic price table so the
arithmetic is checked without depending on what OpenRouter charges today.

    python -m pytest tests/ -q
"""

import copy
import hashlib
import importlib
import json

import pytest

from src import contracts as c
from src import manifest as m
from src import prices as prices_mod


@pytest.fixture
def config():
    return m.load_config(c.CONFIG_PATH)


@pytest.fixture
def jobs(config):
    return m.build_jobs(config)


# ------------------------------------------------------------ grid shape ----

def test_job_count_equals_the_expected_cross_product(config, jobs):
    """CONTRACTS.md §3, derived rather than hardcoded: the direct arm follows
    `config.models[].logprobs`, so swapping a model changes the count."""
    n_models = len(config["models"])
    n_logprobs = sum(1 for model in config["models"] if model["logprobs"])
    n_scenarios = len(config["scenarios"])
    n_personas = len(config["personas"])
    n_temps = len(config["temperatures"])
    n_reps = config["replicates_cot"]

    expected_direct = n_logprobs * n_scenarios * n_personas
    expected_cot = n_models * n_temps * n_scenarios * n_personas * n_reps
    expected_contam = n_models * n_scenarios

    counts = m.count_jobs(jobs)["by_mode"]
    assert counts["direct"] == expected_direct
    assert counts["cot"] == expected_cot
    assert counts[c.CONTAMINATION_MODE] == expected_contam
    assert len(jobs) == expected_direct + expected_cot + expected_contam


def test_shipped_config_produces_the_documented_grid(jobs):
    """The numbers CONTRACTS.md §3 states for the config as shipped. This is a
    regression guard on the *config*, not a hardcoded rule in `build_jobs` —
    the test above is the rule."""
    counts = m.count_jobs(jobs)["by_mode"]
    # 5 models x 2 scenarios x 4 personas (direct); x 5 temperatures x 25
    # replicates (cot); 5 x 2 (contamination). Changed 2026-09-03: isomorph
    # dropped from the grid and adversary_bluff promoted into it, which is a
    # net reduction from 45/5625/15.
    # 5 models x 4 scenarios x 4 personas (direct); x 5 temperatures x 25
    # replicates (cot); 5 x 4 (contamination). Changed 2026-09-03 again:
    # neutral_switch and symmetric_blind added to separate the mechanisms the
    # original pair confounds (CONTRACTS.md §7c).
    assert (counts["direct"], counts["cot"], counts[c.CONTAMINATION_MODE]) == (80, 10000, 20)


def test_run_ids_are_unique(jobs):
    ids = [job["run_id"] for job in jobs]
    assert len(set(ids)) == len(ids)


def test_build_jobs_refuses_to_emit_a_run_id_collision(config, monkeypatch):
    """A collision is silent otherwise: resume writes one record and skips the
    rest, and three stages disagree about which cell a row belongs to."""
    monkeypatch.setattr(c, "run_id", lambda *args: "0" * 40)
    with pytest.raises(AssertionError, match="duplicate run_id"):
        m.build_jobs(config)


def test_run_ids_come_from_contracts(jobs):
    for job in jobs:
        assert job["run_id"] == c.run_id(job["model"], job["scenario"], job["persona"],
                                         job["mode"], job["temperature"], job["replicate"])
        assert len(job["run_id"]) == 40


def test_pinned_run_ids_are_present(jobs):
    """CONTRACTS.md §2's worked examples must actually appear in the grid."""
    ids = {job["run_id"] for job in jobs}
    assert "660460d5e8cd0784ab3877242a11b57e43ee3ed9" in ids   # llama|switch|expert|cot|1.2|7
    assert "2820f375be14c6ebeaf684c1af8fc60accaedf4f" in ids   # 4o-mini|keep|none|direct|1.0|0


# ---------------------------------------------------------- determinism ----

def test_the_same_config_produces_byte_identical_job_lists(config):
    first = json.dumps(m.build_jobs(config), sort_keys=False)
    second = json.dumps(m.build_jobs(config), sort_keys=False)
    assert first == second


def test_reloading_the_config_produces_byte_identical_job_lists():
    """Determinism must survive a fresh YAML load, not just a reused dict."""
    first = json.dumps(m.build_jobs(m.load_config(c.CONFIG_PATH)))
    second = json.dumps(m.build_jobs(m.load_config(c.CONFIG_PATH)))
    assert first == second


def test_job_lists_are_identical_across_processes(tmp_path):
    """Two `build_jobs` calls in one interpreter share set and dict iteration
    order, so an in-process comparison cannot see the nondeterminism that
    actually threatens `--limit`: a reordering would make a resumed pilot pick a
    different subsample every invocation, skewed by whichever axis moved."""
    import os
    import subprocess
    import sys

    digests = []
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-m", "src.manifest", "--json"],
            cwd=c.ROOT, capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert result.returncode == 0, result.stderr.decode()[:2000]
        digests.append(hashlib.sha1(result.stdout).hexdigest())
    assert len(set(digests)) == 1, f"emission order varies with PYTHONHASHSEED: {digests}"


def test_emission_key_sequence_is_the_nested_loop_order(config, jobs):
    """Pins the order in-process too, so a reordering fails fast rather than
    only under a different hash seed."""
    expected = []
    for model in config["models"]:
        if model["logprobs"]:
            for scenario in config["scenarios"]:
                for persona in config["personas"]:
                    expected.append(("direct", model["id"], scenario, persona, 1.0, 0))
    for model in config["models"]:
        for scenario in config["scenarios"]:
            for persona in config["personas"]:
                for temperature in config["temperatures"]:
                    for replicate in range(config["replicates_cot"]):
                        expected.append(("cot", model["id"], scenario, persona,
                                         float(temperature), replicate))
    for model in config["models"]:
        for scenario in config["scenarios"]:
            expected.append((c.CONTAMINATION_MODE, model["id"], scenario, "none", 1.0, 0))

    actual = [(job["mode"], job["model"], job["scenario"], job["persona"],
               job["temperature"], job["replicate"]) for job in jobs]
    assert actual == expected


def test_emission_order_is_mode_then_config_order(config, jobs):
    modes_in_order = [job["mode"] for job in jobs]
    # Each mode occupies one contiguous block, in MODE_ORDER.
    seen = [mode for i, mode in enumerate(modes_in_order)
            if i == 0 or mode != modes_in_order[i - 1]]
    assert seen == [mode for mode in m.MODE_ORDER if mode in set(modes_in_order)]

    # Within a mode, models appear in config order.
    cot_models = [job["model"] for job in jobs if job["mode"] == "cot"]
    first_seen = list(dict.fromkeys(cot_models))
    assert first_seen == [model["id"] for model in config["models"]]


# ------------------------------------------------------------ job fields ----

def test_every_job_has_exactly_the_contract_fields(jobs):
    for job in jobs:
        assert tuple(job) == m.JOB_FIELDS


def test_no_job_carries_a_provider_key(jobs):
    """CONTRACTS.md §3's name collision: the request's routing is
    `provider_routing`, so the response's `provider` cannot overwrite it."""
    for job in jobs:
        assert "provider" not in job


def test_direct_jobs_match_the_direct_convention(config, jobs):
    direct = [job for job in jobs if job["mode"] == "direct"]
    assert direct
    for job in direct:
        assert job["temperature"] == c.DIRECT_TEMPERATURE == 1.0
        assert job["replicate"] == 0
        assert job["logprobs"] is True
        assert job["top_logprobs"] == c.DIRECT_TOP_LOGPROBS == 10
        assert job["max_tokens"] == c.DIRECT_MAX_TOKENS
        assert job["provider_routing"] == c.DIRECT_PROVIDER_PIN[job["model"]]
        assert job["provider_routing"]["allow_fallbacks"] is False
        assert job["provider_routing"]["only"]
        assert job["messages"] == c.build_messages(
            job["scenario"], job["persona"], "direct", config["allow_randomise_in_prompt"])

    # Exactly one per (model, scenario, persona).
    cells = [(job["model"], job["scenario"], job["persona"]) for job in direct]
    assert len(set(cells)) == len(cells)


def test_direct_jobs_only_for_logprobs_capable_models(config, jobs):
    capable = {model["id"] for model in config["models"] if model["logprobs"]}
    assert {job["model"] for job in jobs if job["mode"] == "direct"} == capable


def test_provider_routing_is_deep_copied(jobs):
    """Mutating one job must not reach through to contracts.DIRECT_PROVIDER_PIN
    or to every other direct job for the same model."""
    direct = [job for job in jobs if job["mode"] == "direct"]
    model_id = direct[0]["model"]
    original = copy.deepcopy(c.DIRECT_PROVIDER_PIN[model_id])
    direct[0]["provider_routing"]["only"].append("mutated")
    assert c.DIRECT_PROVIDER_PIN[model_id] == original
    siblings = [j for j in direct if j["model"] == model_id and j is not direct[0]]
    assert all(job["provider_routing"] == original for job in siblings)


def test_cot_jobs_cover_the_full_cross_product(config, jobs):
    cot = [job for job in jobs if job["mode"] == "cot"]
    for job in cot:
        assert job["logprobs"] is False
        assert job["top_logprobs"] is None
        # Pinned since the provider-mixing fix: cot used to be routed freely and
        # one model's replicates landed on four providers, which are not the
        # same computational object and confound the direct-vs-cot comparison
        # (CONTRACTS.md §8, contracts.COT_PROVIDER_PIN).
        assert job["provider_routing"] == c.provider_pin(job["model"], "cot")
        assert job["provider_routing"]["allow_fallbacks"] is False
        assert job["max_tokens"] == c.COT_MAX_TOKENS
    cells = {(job["model"], job["scenario"], job["persona"],
              job["temperature"], job["replicate"]) for job in cot}
    assert len(cells) == len(cot)
    assert {job["temperature"] for job in cot} == {float(t) for t in config["temperatures"]}
    assert {job["replicate"] for job in cot} == set(range(config["replicates_cot"]))


def test_contamination_jobs(config, jobs):
    contamination = [job for job in jobs if job["mode"] == c.CONTAMINATION_MODE]
    assert len(contamination) == len(config["models"]) * len(config["scenarios"])
    for job in contamination:
        assert job["persona"] == "none"
        assert job["temperature"] == c.CONTAMINATION_TEMPERATURE == 1.0
        assert job["replicate"] == 0
        assert job["logprobs"] is False
        assert job["max_tokens"] == c.CONTAMINATION_MAX_TOKENS
        assert job["messages"] == c.build_contamination_messages(job["scenario"])
        assert "End your response with" not in job["messages"][0]["content"]


def test_temperatures_are_floats(jobs):
    """`run_id` formats with `:.1f`; an int 0 from YAML would still hash the
    same, but the job dict must not carry one."""
    assert all(isinstance(job["temperature"], float) for job in jobs)


# ------------------------------------------------------- config handling ----

def test_direct_arm_is_skipped_for_a_logprobs_false_model(config):
    tweaked = copy.deepcopy(config)
    dropped = tweaked["models"][0]["id"]
    tweaked["models"][0]["logprobs"] = False
    jobs = m.build_jobs(tweaked)
    assert dropped not in {job["model"] for job in jobs if job["mode"] == "direct"}
    # ...but its CoT and contamination arms survive.
    assert dropped in {job["model"] for job in jobs if job["mode"] == "cot"}
    assert dropped in {job["model"] for job in jobs if job["mode"] == c.CONTAMINATION_MODE}


def test_logprobs_true_without_a_provider_pin_is_a_hard_error(config):
    """CONTRACTS.md §8: an unpinned direct call gets routed to a provider that
    returns no logprobs, wasting the entire direct grid. Refuse to emit it."""
    tweaked = copy.deepcopy(config)
    tweaked["models"][0]["id"] = "someone/unpinned-model"
    with pytest.raises(KeyError, match="DIRECT_PROVIDER_PIN"):
        m.build_jobs(tweaked)


def test_answer_modes_gates_the_grid_but_not_contamination(config):
    tweaked = copy.deepcopy(config)
    tweaked["answer_modes"] = ["cot"]
    modes = {job["mode"] for job in m.build_jobs(tweaked)}
    assert modes == {"cot", c.CONTAMINATION_MODE}

    tweaked["answer_modes"] = ["direct"]
    modes = {job["mode"] for job in m.build_jobs(tweaked)}
    assert modes == {"direct", c.CONTAMINATION_MODE}


def test_side_run_modes_are_rejected_in_answer_modes(config):
    for bad in (c.CONTAMINATION_MODE, c.DIRECT_VALIDATION_MODE):
        tweaked = copy.deepcopy(config)
        tweaked["answer_modes"] = ["cot", bad]
        with pytest.raises(ValueError, match="answer_modes"):
            m.build_jobs(tweaked)


def test_direct_validation_is_never_emitted(jobs):
    assert c.DIRECT_VALIDATION_MODE not in {job["mode"] for job in jobs}


def test_randomise_toggle_reaches_the_prompt(config):
    tweaked = copy.deepcopy(config)
    tweaked["allow_randomise_in_prompt"] = True
    job = next(j for j in m.build_jobs(tweaked) if j["mode"] == "cot")
    assert "random method" not in job["messages"][-1]["content"]


# ------------------------------------------------------------------ cost ----

def _fake_job(model, mode, max_tokens, chars):
    return {"model": model, "tier": "small", "mode": mode,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": "x" * chars}]}


def test_prompt_tokens_are_anchored_on_the_measurement():
    """CONTRACTS.md §10 measured 201 prompt tokens for the reference prompt."""
    scenario, persona, mode = m.MEASURED_PROMPT_REFERENCE
    job = {"messages": c.build_messages(scenario, persona, mode)}
    assert m.estimate_prompt_tokens(job) == m.MEASURED_PROMPT_TOKENS == 201


def test_persona_adds_roughly_twenty_tokens():
    """CONTRACTS.md §10: "Personas add ~20"."""
    base = m.estimate_prompt_tokens({"messages": c.build_messages("default_keep", "none", "cot")})
    for persona in ("average_person", "expert"):
        with_persona = m.estimate_prompt_tokens(
            {"messages": c.build_messages("default_keep", persona, "cot")})
        assert 10 <= with_persona - base <= 30


def test_completion_estimate_is_capped_by_max_tokens():
    """A model whose CoT median is 476 still cannot emit more than 4 in direct
    mode."""
    verbose = "google/gemma-4-26b-a4b-it"
    assert m.estimate_completion_tokens(_fake_job(verbose, "cot", 2000, 10)) == (476, True)
    assert m.estimate_completion_tokens(_fake_job(verbose, "direct", 4, 10)) == (4, True)


def test_unmeasured_model_falls_back_to_the_ceiling():
    tokens, measured = m.estimate_completion_tokens(_fake_job("who/knows", "cot", 2000, 10))
    assert (tokens, measured) == (2000, False)


def test_cost_uses_usd_per_token_with_no_million_divisor():
    """PLAN.md §6's formula divides by 1e6 because it assumes per-million
    prices. `prices.fetch_prices` returns USD per token, so there is no
    divisor. This test is the guard on that unit."""
    prices = {"openai/gpt-4o-mini": {"prompt": 1e-6, "completion": 2e-6}}
    job = _fake_job("openai/gpt-4o-mini", "cot", 2000, 871)   # 871 chars -> 201 tokens
    result = m.estimate_cost([job], prices)

    entry = result["by_model"]["openai/gpt-4o-mini"]
    assert entry["prompt_tokens"] == 201
    assert entry["completion_tokens"] == 156          # the measured median
    assert entry["cost"] == pytest.approx(201 * 1e-6 + 156 * 2e-6)
    assert entry["cost_ceiling"] == pytest.approx(201 * 1e-6 + 2000 * 2e-6)
    # A stray /1e6 would make this 4.5e-10 instead of 4.5e-4.
    assert entry["cost"] > 1e-5


def test_cost_reproduces_a_real_billed_call():
    """CONTRACTS.md §4 records an actual OpenRouter response for
    `openai/gpt-4o-mini`: 201 prompt tokens, 1 completion token, and a billed
    `usage.cost` of exactly 3.075e-05. At the catalogue's $0.15/$0.60 per M
    that is `201 × 1.5e-7 + 1 × 6e-7`. If the arithmetic here cannot reproduce
    a real invoice, the whole pre-flight gate is decoration."""
    prices = {"openai/gpt-4o-mini": {"prompt": 1.5e-07, "completion": 6e-07}}
    assert 201 * 1.5e-07 + 1 * 6e-07 == pytest.approx(3.075e-05)

    # A direct job: max_tokens 4, and gpt-4o-mini's median (156) caps to 4.
    job = _fake_job("openai/gpt-4o-mini", "direct", 4, 871)
    entry = m.estimate_cost([job], prices)["by_model"]["openai/gpt-4o-mini"]
    assert entry["prompt_tokens"] == 201 and entry["completion_tokens"] == 4
    assert entry["cost"] == pytest.approx(201 * 1.5e-07 + 4 * 6e-07)


def test_estimate_cost_rejects_per_million_prices():
    """The $0.54-vs-$0.00000054 trap: PLAN.md §6's formula divides by 1e6
    because it assumes per-million prices, and both figures read as "under
    $5". The unit has to be caught mechanically."""
    per_million = {"openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60}}
    with pytest.raises(ValueError, match="per token"):
        m.estimate_cost([_fake_job("openai/gpt-4o-mini", "cot", 2000, 871)], per_million)


def test_cost_breaks_down_per_model_and_totals_agree():
    prices = {"a/terse": {"prompt": 1e-7, "completion": 2e-7},
              "b/verbose": {"prompt": 1e-7, "completion": 5e-7}}
    jobs = [_fake_job("a/terse", "cot", 2000, 871) for _ in range(3)]
    jobs += [_fake_job("b/verbose", "cot", 2000, 871) for _ in range(2)]
    result = m.estimate_cost(jobs, prices)

    assert set(result["by_model"]) == {"a/terse", "b/verbose"}
    assert result["by_model"]["a/terse"]["n_calls"] == 3
    assert result["by_model"]["b/verbose"]["n_calls"] == 2
    assert result["total"]["n_calls"] == 5
    assert result["total"]["cost"] == pytest.approx(
        sum(e["cost"] for e in result["by_model"].values()))
    assert result["total"]["cost_ceiling"] >= result["total"]["cost"]
    # Neither model is in the measured table, so both are ceiling-estimated.
    assert set(result["models_without_measured_output"]) == {"a/terse", "b/verbose"}


def test_cost_flags_a_model_it_cannot_price():
    result = m.estimate_cost([_fake_job("nobody/prices-me", "cot", 900, 871)], {})
    assert result["models_missing_price"] == ["nobody/prices-me"]
    assert result["total"]["cost"] == 0.0
    assert result["total"]["n_calls"] == 1          # counted, just not priced


def test_cost_by_mode_partitions_the_calls(config, jobs):
    prices = {model["id"]: {"prompt": 1e-7, "completion": 1e-7} for model in config["models"]}
    result = m.estimate_cost(jobs, prices)
    for entry in result["by_model"].values():
        assert sum(cell["n_calls"] for cell in entry["by_mode"].values()) == entry["n_calls"]
    assert result["total"]["n_calls"] == len(jobs)
    assert not result["models_missing_price"]


# ---------------------------------------------------------------- prices ----
# src/prices.py produces the only number the operator approves spend against,
# and manifest.estimate_cost is its sole consumer, so its parsing lives here.

CATALOGUE = {"data": [
    {"id": "openai/gpt-4o-mini", "pricing": {"prompt": "0.00000015",
                                             "completion": "0.0000006"}},
    {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
    {"id": "vendor/no-completion", "pricing": {"prompt": "0.000001"}},
    {"id": "vendor/free", "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "vendor/unparseable", "pricing": {"prompt": "n/a", "completion": "0.1"}},
]}


def test_parse_catalogue_keeps_usd_per_token_and_drops_the_rest():
    parsed = prices_mod.parse_catalogue(CATALOGUE)
    assert parsed["openai/gpt-4o-mini"] == {"prompt": 1.5e-07, "completion": 6e-07}
    assert parsed["vendor/free"] == {"prompt": 0.0, "completion": 0.0}   # free is a price
    # A router meta-model priced "-1" would *reduce* an estimate.
    assert "openrouter/auto" not in parsed
    assert "vendor/no-completion" not in parsed
    assert "vendor/unparseable" not in parsed


def test_parsed_prices_are_per_token_not_per_million():
    """The unit contract, asserted against a catalogue body rather than a
    hand-written dict: $0.15/M must arrive as 1.5e-07."""
    parsed = prices_mod.parse_catalogue(CATALOGUE)
    assert parsed["openai/gpt-4o-mini"]["prompt"] * 1e6 == pytest.approx(0.15)
    m.estimate_cost([_fake_job("openai/gpt-4o-mini", "cot", 2000, 871)], parsed)


def test_parse_catalogue_rejects_a_useless_payload():
    for payload in ({}, {"data": []}, {"data": [{"id": "x"}]}, "not json"):
        with pytest.raises(ValueError):
            prices_mod.parse_catalogue(payload)


def test_cache_round_trip_and_freshness(tmp_path):
    from datetime import datetime, timedelta, timezone
    cache = tmp_path / "prices.json"
    calls = {"n": 0}

    def fake_get(timeout):
        calls["n"] += 1
        return CATALOGUE

    prices_mod._get_catalogue = fake_get
    try:
        first = prices_mod.fetch_prices(cache_path=cache)
        assert calls["n"] == 1 and "openai/gpt-4o-mini" in first

        # Fresh cache: no second fetch.
        assert prices_mod.fetch_prices(cache_path=cache) == first
        assert calls["n"] == 1

        # Older than 24h: refetch.
        prices_mod.fetch_prices(cache_path=cache,
                                now=datetime.now(timezone.utc) + timedelta(hours=25))
        assert calls["n"] == 2

        _, age = prices_mod.cache_status(cache)
        assert age is not None and age < 1.0
    finally:
        importlib.reload(prices_mod)


def test_a_failed_refetch_falls_back_to_a_stale_cache(tmp_path, capsys):
    from datetime import datetime, timedelta, timezone
    cache = tmp_path / "prices.json"
    prices_mod._get_catalogue = lambda timeout: CATALOGUE
    try:
        good = prices_mod.fetch_prices(cache_path=cache)

        def boom(timeout):
            raise RuntimeError("openrouter unreachable")

        prices_mod._get_catalogue = boom
        stale = prices_mod.fetch_prices(cache_path=cache,
                                        now=datetime.now(timezone.utc) + timedelta(hours=48))
        assert stale == good, "yesterday's prices beat refusing to start"
        assert "could not refresh prices" in capsys.readouterr().err

        # ...but with no cache at all, an estimate from no prices is not an estimate.
        with pytest.raises(RuntimeError):
            prices_mod.fetch_prices(cache_path=tmp_path / "absent.json")
    finally:
        importlib.reload(prices_mod)


def test_lookup_falls_back_across_variant_suffixes():
    parsed = prices_mod.parse_catalogue(CATALOGUE)
    assert prices_mod.lookup(parsed, "openai/gpt-4o-mini:free") == \
        prices_mod.lookup(parsed, "openai/gpt-4o-mini")
    assert prices_mod.lookup(parsed, "nobody/here") is None


def test_dry_run_report_names_counts_and_both_numbers(config, jobs):
    prices = {model["id"]: {"prompt": 1e-7, "completion": 1e-7} for model in config["models"]}
    report = m.format_dry_run(jobs, m.estimate_cost(jobs, prices), config)
    assert "JOBS BY MODE" in report and "JOBS BY MODEL AND MODE" in report
    assert "ESTIMATE" in report and "CEILING" in report
    for model in config["models"]:
        assert model["id"] in report
    assert "10,000" in report and "80" in report
