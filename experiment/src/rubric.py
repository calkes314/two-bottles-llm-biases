"""The annotation prompt — PLAN.md §8.4, with PLAN.md §5's rubric pasted in.

`build_annotation_prompt` takes **the response text and nothing else**. That is
not a stylistic preference: CONTRACTS.md §5 requires the annotator to be blind
to condition, and the cheapest way to guarantee it is to make the condition
unavailable at the call site. There is deliberately no parameter through which a
model id, a persona, a scenario name or a temperature could reach the prompt.

The rubric itself is `contracts.DEPTH_RUBRIC`, imported rather than re-pasted.
That constant carries PLAN.md §5's depth table *and* its two flag definitions,
which matters because the JSON schema below asks for `flag_no_basis` and
`flag_rigged` without saying what either means — pasting the table alone would
name two booleans the annotator has never had defined.

The two enum lines are built from `contracts.ANSWERS` and
`contracts.DEPTH_CODES` in their canonical order, so the prompt cannot drift
from the labels `annotate.py` is willing to accept.
"""

from __future__ import annotations

from src.contracts import ANSWERS, DEPTH_CODES, DEPTH_RUBRIC

_ANSWER_ENUM = "|".join(ANSWERS)
_DEPTH_ENUM = "|".join(DEPTH_CODES)

# PLAN.md §8.4 verbatim. The line breaks are PLAN.md's own; the `[rubric table
# pasted here]` placeholder is filled with DEPTH_RUBRIC.
ANNOTATION_PROMPT_TEMPLATE = (
    "You are coding responses to a decision scenario. Return ONLY a JSON object,\n"
    "no prose, no markdown fences.\n"
    "\n"
    '{"answer": one of ' + _ANSWER_ENUM + ",\n"
    ' "depth":  one of ' + _DEPTH_ENUM + ",\n"
    ' "flag_no_basis": true|false,\n'
    ' "flag_rigged":   true|false,\n'
    ' "flag_proposer": true|false}\n'
    "\n"
    + DEPTH_RUBRIC + "\n"
    "\n"
    "RESPONSE TO CODE:\n"
    '"""{response_text}"""'
)


def build_annotation_prompt(response_text: str) -> str:
    """The full annotation prompt for one response.

    Takes the response text only — see the module docstring. The text is
    substituted verbatim, never summarised, truncated or reformatted: the
    annotator must code what the subject model actually said.

    Known limitation, kept rather than worked around: a response containing a
    literal triple quote can run into the delimiter. PLAN.md §8.4 fixes the
    delimiter, and escaping it would mean coding text the subject model did not
    emit, so such a case is left to fail loudly instead — it lands in
    `raw_annotator_output` (CONTRACTS.md §5) with `answer: UNPARSEABLE`, which
    is exactly the hand-review queue PLAN.md §4 asks for.
    """
    return ANNOTATION_PROMPT_TEMPLATE.replace("{response_text}", response_text)
