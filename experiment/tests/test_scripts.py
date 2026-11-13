"""Tests for the two side-run scripts the parallel sessions left unbuilt.

`scripts/direct_validation.py` implements CONTRACTS.md §8's `direct_validation`
sub-run, which `manifest.py` deliberately refuses to emit. `scripts/
contamination_report.py` reads the §9 records, which `analyze.py` deliberately
filters out. Both are one-off operator tools, so the tests here cover the parts
that would silently produce wrong data rather than every branch.
"""

from __future__ import annotations

import json

import pytest

from src import contracts as c
from scripts import contamination_report as cr
from scripts import direct_validation as dv


# --------------------------------------------------------- the §8 trap ----

def test_validation_jobs_do_not_collapse_onto_one_run_id():
    """CONTRACTS.md §8's named trap: reuse direct mode's temperature=1.0 and
    replicate=0 and all 200 draws share two run_ids, so resume writes the first
    of each and silently discards the other 198."""
    jobs = dv.build_validation_jobs()
    assert len(jobs) == 200
    assert len({j["run_id"] for j in jobs}) == 200


def test_validation_jobs_carry_real_temperatures_and_replicates():
    jobs = dv.build_validation_jobs()
    assert sorted({j["temperature"] for j in jobs}) == [0.4, 1.2]
    assert sorted({j["replicate"] for j in jobs}) == list(range(100))
    for T in (0.4, 1.2):
        assert sum(1 for j in jobs if j["temperature"] == T) == 100
    # Not the direct-mode constants.
    assert all(j["temperature"] != c.DIRECT_TEMPERATURE for j in jobs)


def test_validation_mode_cannot_leak_into_the_factorial_grid():
    jobs = dv.build_validation_jobs()
    assert {j["mode"] for j in jobs} == {c.DIRECT_VALIDATION_MODE}
    assert c.DIRECT_VALIDATION_MODE not in c.MODES


def test_validation_prompt_is_byte_identical_to_the_direct_prompt():
    """CONTRACTS.md §8. A different prompt would validate a different question."""
    jobs = dv.build_validation_jobs()
    direct = c.build_messages(dv.VALIDATION_SCENARIO, dv.VALIDATION_PERSONA,
                              "direct", False)
    assert all(j["messages"] == direct for j in jobs)


def test_validation_jobs_match_the_contracts_table():
    jobs = dv.build_validation_jobs()
    for job in jobs:
        assert job["model"] == "openai/gpt-4o-mini"
        assert job["scenario"] == "default_keep"
        assert job["persona"] == "none"
        assert job["max_tokens"] == c.DIRECT_MAX_TOKENS
        assert job["logprobs"] is True
        assert job["top_logprobs"] == c.DIRECT_TOP_LOGPROBS
        assert job["provider_routing"] == c.DIRECT_PROVIDER_PIN["openai/gpt-4o-mini"]


def test_validation_jobs_do_not_alias_the_shared_provider_pin():
    """A job mutating its routing must not rewrite every other direct job's."""
    jobs = dv.build_validation_jobs()
    jobs[0]["provider_routing"]["only"] = ["mutated"]
    assert c.DIRECT_PROVIDER_PIN["openai/gpt-4o-mini"]["only"] == ["openai"]
    assert jobs[1]["provider_routing"]["only"] == ["openai"]


# --------------------------------------------- reading the sampled letter ----

@pytest.mark.parametrize("content, expected", [
    ("K", "K"),
    ("S", "S"),
    (" K", "K"),
    ("\n\nS", "S"),
    ("k", "K"),
    ("KEEP", "K"),          # first bare letter wins; max_tokens is 4
    ("(K)", None),          # not a bare answer letter, per contracts §8 rule 3
    ("I cannot", None),
    ("", None),
    (None, None),
])
def test_sampled_letter(content, expected):
    rec = {"response": {"choices": [{"message": {"content": content}}]}}
    assert dv.sampled_letter(rec) == expected


def test_sampled_letter_reads_content_not_logprobs():
    """The run validates the provider's SAMPLING, so the observed proportion has
    to come from what was emitted, not from the distribution it was drawn from."""
    rec = {"response": {"choices": [{
        "message": {"content": "S"},
        "logprobs": {"content": [{"token": "K", "logprob": -0.01, "top_logprobs": [
            {"token": "K", "logprob": -0.01}, {"token": "S", "logprob": -5.0}]}]},
    }]}}
    assert dv.sampled_letter(rec) == "S"


def test_wilson_interval_stays_inside_zero_one():
    for k, n in [(0, 100), (100, 100), (1, 100), (50, 100)]:
        lo, hi = dv.wilson(k, n)
        assert 0.0 <= lo <= hi <= 1.0
    assert dv.wilson(50, 100)[0] < 0.5 < dv.wilson(50, 100)[1]


def test_prediction_comes_from_the_matching_direct_cell(tmp_path):
    """CONTRACTS.md §8 measured P(KEEP) moving 0.854 -> 0.986 across providers
    for one model, so a prediction read off a different cell is not a
    prediction about this one."""
    raw = tmp_path / "raw.jsonl"
    def record(scenario, persona, lp_k):
        return {
            "run_id": c.run_id("openai/gpt-4o-mini", scenario, persona, "direct", 1.0, 0),
            "mode": "direct", "model": "openai/gpt-4o-mini",
            "scenario": scenario, "persona": persona, "provider": "OpenAI",
            "response": {"choices": [{"logprobs": {"content": [{
                "token": "K", "logprob": lp_k, "top_logprobs": [
                    {"token": "K", "logprob": lp_k},
                    {"token": "S", "logprob": -5.0}]}]}}]},
        }
    raw.write_text("".join(json.dumps(r) + "\n" for r in [
        record("isomorph", "none", -3.0),          # wrong scenario
        record("default_keep", "expert", -2.0),    # wrong persona
        record("default_keep", "none", -0.01),     # the one we want
    ]), encoding="utf-8")
    logprobs, status, prov = dv.predicted_from_grid(raw)
    assert status == "ok"
    assert logprobs["K"] == pytest.approx(-0.01)
    assert prov["provider"] == "OpenAI"


def test_prediction_is_absent_rather_than_wrong_when_the_grid_has_not_run(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("", encoding="utf-8")
    logprobs, status, _ = dv.predicted_from_grid(raw)
    assert logprobs is None
    assert status == "missing_direct_record"


# ------------------------------------------------- contamination reporter ----

def test_contamination_reporter_reads_only_contamination_records(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [
        {"mode": "cot", "model": "m", "scenario": "default_keep",
         "response": {"choices": [{"message": {"content": "KEEP"}}]}},
        {"mode": "contamination", "model": "m", "scenario": "default_keep",
         "response": {"choices": [{"message": {"content": "A Study in Pink"}}]}},
        {"mode": "direct", "model": "m", "scenario": "default_keep",
         "response": {"choices": [{"message": {"content": "K"}}]}},
    ]
    raw.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    got = cr.load(raw)
    assert len(got) == 1
    assert got[0]["mode"] == "contamination"


@pytest.mark.parametrize("text, label", [
    ("A Study in Pink", "NAMED"),
    ("This resembles the Sherlock episode.", "NAMED"),
    ("NONE", "NONE"),
    ("None.", "NONE"),
    ("No, it does not resemble anything I know.", "NONE"),
    # A title the model NAMES but does not recognise as a story is not a
    # recognition. Three of the five models confabulate game-theory
    # abstractions and fabricated episodes; flag them for reading.
    ('It reminds me of "The Cask of Amontillado".', "UNCLEAR"),
    ('This resembles the **"Two Generals\' Problem"** in game theory', "UNCLEAR"),
])
def test_contamination_verdict(text, label):
    assert cr.verdict(text)[0] == label


@pytest.mark.parametrize("text", [
    # Measured shapes from the real 15. Both thrash through candidates and then
    # conclude NONE; an earlier rule scored both NAMED off the quoted titles
    # they were rejecting, which is the wrong way to be wrong for a yes/no check.
    'The scenario reminds me of "The Hardest Logic Puzzle Ever" ... but more '
    'specifically ... "The Prisoner\'s Dilemma" no, it is more like ... no, I '
    'mean the most suitable one is...\n\nNONE',
    'closely resembles the **"Two-Envelope Paradox"** ... the closest known '
    'depiction is in **"The Two Bottles of Reliquary"** ... For the purpose of '
    'this task, the most accurate answer given the constraints is:\n\n**NONE**',
])
def test_a_trailing_none_outranks_the_titles_dismissed_on_the_way_there(text):
    assert cr.verdict(text)[0] == "NONE"


def test_a_cut_off_response_is_flagged():
    """6 of the real 15 hit max_tokens=200 mid-sentence. These models put the
    verdict last, so a cut-off response may have lost it."""
    cut = {"response": {"choices": [{"finish_reason": "length",
                                     "message": {"content": "and so the answer is"}}]}}
    whole = {"response": {"choices": [{"finish_reason": "stop",
                                       "message": {"content": "NONE"}}]}}
    assert cr._truncated(cut)
    assert not cr._truncated(whole)


def test_contamination_reporter_tolerates_a_bad_line(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text('{"mode": "contamination", "model": "m", "scenario": "isomorph",'
                   ' "response": {"choices": [{"message": {"content": "NONE"}}]}}\n'
                   'not json at all\n\n', encoding="utf-8")
    assert len(cr.load(raw)) == 1


# ------------------------------------------------------- pilot reporter ----

from scripts import pilot_report as pr   # noqa: E402


@pytest.mark.parametrize("text, expected", [
    ("I would keep it.\n\nKEEP", "KEEP"),
    ("reasoning\nSWITCH", "SWITCH"),
    ("reasoning\n\n**KEEP**", "KEEP"),
    ("reasoning\nKEEP.", "KEEP"),
    ("reasoning\n`SWITCH`", "SWITCH"),
    ("reasoning\nKEEP\n\n", "KEEP"),
    ("reasoning\nswitch", "SWITCH"),
    # Format NOT followed: the word is there but not alone on the last line.
    ("reasoning\nI would KEEP the bottle.", None),
    ("reasoning\nMy answer is: KEEP", None),
    # No answer at all.
    ("I refuse to engage with this.", None),
    ("", None),
    # Truncated mid-sentence: the final line is not a bare word.
    ("Let me think about whether he would expect me to", None),
])
def test_final_line_answer(text, expected):
    assert pr.final_line_answer(text) == expected


def test_final_line_answer_is_the_format_check_not_the_answer_extractor():
    """`annotate.py` recovers an answer from anywhere in the text and is right
    to. This function answers a different question: did the model obey PLAN.md
    §2's final-line instruction? A buried answer must read as NOT complied."""
    buried = "The right move is to KEEP.\nBut it hardly matters."
    assert pr.final_line_answer(buried) is None


def test_reporter_surfaces_reasoning_the_annotator_cannot_see():
    """CONTRACTS.md §8: thinking models leave `content` nearly empty and put the
    trace in `message.reasoning`, which the annotator is never shown."""
    rec = {"response": {"choices": [{"message": {
        "content": "KEEP", "reasoning": "long hidden trace " * 20}}]}}
    assert len(pr.reasoning_text(rec)) > 200
    assert pr.response_text(rec) == "KEEP"


def test_a_named_title_plus_a_hedged_none_is_unclear_not_either_one():
    """The real gpt-4o-mini isomorph response. It names The Princess Bride,
    explains the Vizzini goblet mapping, then hedges with a conditional NONE.

    Calling it NONE is a false negative on the most important cell in the
    table; calling it NAMED is the false positive that the three confabulating
    models produce. No rule gets both right, so ambiguity must read as
    ambiguous — the label is a reading aid and this is the case it cannot
    decide."""
    text = ('The scenario you describe resembles elements from the book "The '
            'Princess Bride," particularly the famous "Battle of Wits" scene '
            'between Vizzini and the Man in Black. If neither of these '
            'resonates strongly, I\'ll conclude with "NONE," but "The Princess '
            'Bride" is a notable reference related to the core themes.')
    label, detail = cr.verdict(text)
    assert label == "UNCLEAR"
    assert "read it" in detail


def test_a_none_verdict_is_found_wherever_it_sits():
    """gemma's real isomorph response: it answers NONE on its own line, then
    adds a parenthetical noting that films like The Princess Bride are NOT the
    match. A tail-window check scored that NAMED off the counter-example."""
    text = ('This scenario is a variation of a classic logic puzzle known as '
            'the **"Two Cups Puzzle."**\n\n'
            'However, if you are looking for a specific literary match:\n\n'
            '**NONE**\n\n'
            '(Note: While many films feature "poisoned drink" scenes—such as '
            '*The Princess Bride* or various Agatha Christie adaptations—the '
            'specific logical setup is a thought experiment rather than a '
            'specific famous scene.)')
    assert cr.verdict(text)[0] != "NAMED"


def test_an_inline_answer_is_none_still_counts():
    text = ('This closely resembles the classic "Two-Glass Problem" used in '
            'logic textbooks. It is not tied to a specific work. Therefore, '
            'the answer is **NONE**.')
    assert cr.verdict(text)[0] == "NONE"


def test_a_confident_detailed_recognition_reads_named():
    """qwen's real isomorph response — wrong on the details (it says Westley
    and Count Rugen in the Fire Swamp; it was Vizzini on a hilltop) but a
    genuine recognition of the structure and its resolution."""
    text = ('Yes, this scenario strongly resembles a famous scene from the '
            'film **"The Princess Bride" (1987)**. Specifically, it mirrors '
            'the **"Battle of Wits"** scene: two competitors, two goblets of '
            'wine, one poisoned, and the opponent knows which.')
    assert cr.verdict(text)[0] == "NAMED"
