"""Session-wide guards. Nothing in `tests/` may spend money.

Every stage of this pipeline authenticates through `contracts.load_api_key`,
which reads `OPENROUTER_API_KEY` from the environment and falls back to the
repo's `.env`. That fallback is right for an operator and wrong for a test
run: a test that only clears the environment variable would otherwise fall
through to the real key and make real calls. That happened — the annotate CLI
tests annotated 10 fixture records into the real `data/annotated.jsonl` — so
the guard is structural rather than per-test.
"""

from __future__ import annotations

import pytest

from src import contracts as c


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch, tmp_path_factory):
    """Point the `.env` fallback at a path that does not exist, for every test.

    A test that *wants* a key still sets one — `monkeypatch.setenv` and an
    explicit `env_path=` argument both take precedence over this. What it
    cannot do is reach the operator's real credentials by accident.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        c, "ENV_PATH", tmp_path_factory.mktemp("no-env") / "absent.env"
    )


@pytest.fixture(autouse=True)
def _no_writes_to_the_real_dataset(monkeypatch, tmp_path_factory):
    """Redirect the default data paths so a CLI test cannot append to the study.

    `annotate.py` and `runner.py` both default to `data/raw.jsonl` /
    `data/annotated.jsonl`. A test that exercises a CLI without overriding
    those paths would write into live experiment data, where the fixture
    `run_id`s join to nothing and are indistinguishable from real records.
    """
    sandbox = tmp_path_factory.mktemp("no-data")
    monkeypatch.setattr(c, "DATA_DIR", sandbox)
    monkeypatch.setattr(c, "RAW_PATH", sandbox / "raw.jsonl")
    monkeypatch.setattr(c, "ANNOTATED_PATH", sandbox / "annotated.jsonl")
