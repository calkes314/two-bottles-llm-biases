#!/usr/bin/env python3
"""Smoke test: one hardcoded OpenRouter call.

Verifies three things before anyone builds on this scaffold:
  1. OPENROUTER_API_KEY works and the endpoint is reachable.
  2. The cheapest model in config/budget.yaml actually answers the scenario.
  3. The response JSON has the shape CONTRACTS.md §4 claims (choices[0].
     message.content, model, provider, usage).

Hardcoded on purpose — it must not depend on the manifest, which another
session owns. Cost: well under $0.001.

    python scripts/smoke.py
"""

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.contracts import (  # noqa: E402
    COT_MAX_TOKENS,
    ENV_PATH,
    OPENROUTER_URL,
    build_messages,
    load_api_key,
)

MODEL = "mistralai/mistral-nemo"   # cheapest id in config/budget.yaml at
                                   # $0.019/M in. It appears there as the
                                   # annotator, not as a subject model — see
                                   # CONTRACTS.md §11. Fine here: this test
                                   # checks the key, connectivity and the
                                   # response envelope, not model behaviour.
SCENARIO = "default_keep"
PERSONA = "none"
MODE = "cot"
TEMPERATURE = 0.8


def main() -> int:
    # contracts.load_api_key, shared with runner.py and annotate.py: the .env
    # fallback the message promises is the one it actually reads.
    key = load_api_key()
    if not key:
        print(f"OPENROUTER_API_KEY is not set. Export it, or put it in "
              f"{ENV_PATH}, then re-run.", file=sys.stderr)
        return 2

    body = {
        "model": MODEL,
        "messages": build_messages(SCENARIO, PERSONA, MODE),
        "temperature": TEMPERATURE,
        "max_tokens": COT_MAX_TOKENS,
    }

    print(f"POST {OPENROUTER_URL}")
    print(f"  model={MODEL} scenario={SCENARIO} persona={PERSONA} "
          f"mode={MODE} temperature={TEMPERATURE}\n")

    r = httpx.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {key}"},
        json=body,
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()

    choice = data["choices"][0]
    print("=" * 72)
    print("RESPONSE TEXT")
    print("=" * 72)
    print(choice["message"]["content"])
    print("=" * 72)
    print(f"finish_reason  : {choice.get('finish_reason')}")
    print(f"model_returned : {data.get('model')}")
    print(f"provider       : {data.get('provider')}")
    print(f"usage          : {json.dumps(data.get('usage'))}")
    print(f"top-level keys : {sorted(data.keys())}")
    print(f"choice keys    : {sorted(choice.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
