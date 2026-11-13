"""Local web UI: an ad-hoc experiment playground and a viewer for the grid.

    export OPENROUTER_API_KEY=sk-or-...
    python -m src.webui                 # http://127.0.0.1:5000

Two tabs.

  Playground  Pick a model, edit the scenario and role text freely, choose a
              temperature and a replicate count, run it. Results are saved to
              data/playground.jsonl and charted immediately.

  Experiment  Browse, filter and read the real grid from data/raw.jsonl +
              data/annotated.jsonl, alongside the figures analyze.py wrote.

DESIGN NOTES, the load-bearing ones:

* The API key is read from the environment SERVER-SIDE and never leaves this
  process. No route returns it and the page never sees it.
* Playground calls are written to `contracts.PLAYGROUND_PATH`, never to
  `raw.jsonl`, and carry `contracts.playground_run_id` ids which are prefixed
  and hash the prompt text. Both defences matter: the grid's run_id does NOT
  cover prompt text, so an ad-hoc call with an edited prompt would otherwise
  collide with a real job and either block its collection or contaminate it.
* Request building, retry, backoff and record shape are reused from
  `runner.py` rather than reimplemented, so the playground cannot drift from
  the experiment it is meant to inform.
* This binds to 127.0.0.1. It is a single-user local tool with no
  authentication; do not expose it.
"""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml
from flask import Flask, jsonify, request, send_from_directory

from src import contracts as c
from src import export, runner

MAX_REPLICATES = 50          # a web form makes a runaway trivially easy
MAX_TOTAL_CALLS = 200        # per playground submission
MAX_OUTPUT_TOKENS = 16000    # a slip of the keyboard should not cost real money
_ENDPOINT_CACHE: dict[str, list[dict]] = {}

app = Flask(__name__, static_folder=None)
_runs: dict[str, dict] = {}
_runs_lock = threading.Lock()


# ------------------------------------------------------------------ helpers --

def load_config() -> dict:
    return yaml.safe_load(c.CONFIG_PATH.read_text(encoding="utf-8"))


def quick_answer(text: str) -> str:
    """Cheap last-line extraction, for the instant chart only.

    This is NOT the annotator. It reproduces the scenario's own contract — the
    final line holds one word — and nothing more. Depth codes and the real
    answer labels come from `annotate.py`, which the UI can invoke separately.
    """
    if not text or not text.strip():
        return "EMPTY"
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    last = lines[-1].strip().strip("*#>_`.!:;,\"' ").upper()
    if last in ("KEEP", "SWITCH"):
        return last
    upper = text.upper()
    has_keep, has_switch = "KEEP" in upper, "SWITCH" in upper
    if has_keep != has_switch:
        return ("KEEP" if has_keep else "SWITCH") + "*"   # * = buried, not final line
    return "UNCLEAR"


def provider_routing(model: str, mode: str, choice: str | None) -> dict | None:
    """Routing for a playground call.

    `choice` is what the form asked for:
      None / "pinned"  the model's verified pin — matches the grid, the default
      "any"            unpinned, so OpenRouter routes freely. Useful ONLY for
                       deliberately observing the provider spread; it is the
                       thing `contracts.provider_pin` exists to prevent.
      <provider slug>  that provider specifically, for comparing two of them.
    """
    if choice in (None, "", "pinned"):
        return c.provider_pin(model, mode)
    if choice == "any":
        # Direct still pins: unpinned, the logprobs may simply not come back.
        return c.provider_pin(model, mode) if mode == "direct" else None
    return {"only": [choice], "allow_fallbacks": False}


def build_playground_jobs(*, model: str, scenario_text: str, system_text: str,
                          temperature: float, replicates: int, mode: str,
                          scenario_label: str, persona_label: str,
                          max_tokens: int | None = None,
                          provider: str | None = None,
                          tier: str = "playground") -> list[dict]:
    """Job dicts in the same shape `runner.request_body` expects, so the
    playground shares one request path with the grid."""
    stamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    messages: list[dict] = []
    if system_text.strip():
        messages.append({"role": "system", "content": system_text.strip()})
    messages.append({"role": "user", "content": scenario_text})

    direct = mode == "direct"
    jobs = []
    for replicate in range(replicates):
        jobs.append({
            "run_id": c.playground_run_id(model, temperature, replicate, messages, stamp),
            "experiment": "playground",
            "model": model,
            "tier": tier,
            "scenario": scenario_label,
            "persona": persona_label,
            "mode": mode,
            "temperature": c.DIRECT_TEMPERATURE if direct else float(temperature),
            "replicate": replicate,
            "messages": messages,
            "max_tokens": c.DIRECT_MAX_TOKENS if direct else int(
                max_tokens or c.COT_MAX_TOKENS),
            "logprobs": direct,
            "top_logprobs": c.DIRECT_TOP_LOGPROBS if direct else None,
            # Pinned for EVERY mode, like the grid. Leaving cot unpinned scattered
            # one model's replicates across four measurably different providers
            # (CONTRACTS.md §8) — and a playground answering from a different
            # provider than the grid cannot inform the grid.
            "provider_routing": provider_routing(model, mode, provider),
            "batch": stamp,
        })
    return jobs


def _execute(token: str, jobs: list[dict], api_key: str) -> None:
    """Run one playground batch on a worker thread."""
    try:
        before = {r["run_id"] for r in export.read_jsonl(c.PLAYGROUND_PATH)}
        asyncio.run(runner.run_jobs(
            jobs, api_key=api_key, max_concurrency=6,
            raw_path=c.PLAYGROUND_PATH, failures_path=None,
            handle_sigint=False,          # not the main thread
        ))
        fresh = [r for r in export.read_jsonl(c.PLAYGROUND_PATH)
                 if r["run_id"] not in before]
        with _runs_lock:
            _runs[token].update(state="done", records=len(fresh),
                                results=[_record_view(r) for r in fresh])
    except Exception as exc:                       # surfaced in the UI, not swallowed
        with _runs_lock:
            _runs[token].update(state="error", error=f"{type(exc).__name__}: {exc}")


def _record_view(record: dict) -> dict:
    row = export.flatten(record, None)
    row["quick_answer"] = quick_answer(row["response_text"])
    row["batch"] = record.get("batch")
    row["max_tokens"] = record.get("max_tokens")
    row["requested_provider"] = ((record.get("provider_routing") or {})
                                 .get("only") or [None])[0]
    message = ((record.get("response") or {}).get("choices") or [{}])[0].get("message") or {}
    # Where the tokens went. A reasoning budget can produce thousands of
    # characters here while `content` shrinks — worth seeing, since the
    # annotator only ever reads `content`.
    row["reasoning_chars"] = len(message.get("reasoning") or "")
    row["reasoning_text"] = message.get("reasoning") or ""
    return row


# ------------------------------------------------------------------- routes --

@app.get("/")
def index():
    return send_from_directory(Path(__file__).parent, "webui_index.html")


@app.get("/api/options")
def options():
    cfg = load_config()
    scenarios = {name: c.load_scenario(name) for name in c.SCENARIOS}
    prices = {m["id"]: {"in": m["in"], "out": m["out"]}
              for m in c.PLAYGROUND_EXTRA_MODELS}
    models = [{"id": m["id"], "tier": m["tier"], "logprobs": m["logprobs"],
               "grid": True,
               "no_cot": m["id"] in c.NO_COT_MODELS,
               "cot_chars": c.COT_CONTENT_CHARS.get(m["id"])}
              for m in cfg["models"]]
    # Playground-only, and labelled so: putting one of these in the grid is a
    # decision about the experiment, not a click in a form.
    models += [{"id": m["id"], "tier": m["tier"], "logprobs": None,
                "grid": False, "no_cot": False, "cot_chars": None,
                "price_in": m["in"], "price_out": m["out"]}
               for m in c.PLAYGROUND_EXTRA_MODELS]
    return jsonify({
        "models": models,
        "prices": prices,
        "scenarios": scenarios,
        "personas": {"none": "", **c.PERSONA_SYSTEM_PROMPTS,
                     **c.PLAYGROUND_PERSONAS},
        "grid_personas": list(c.PERSONAS),
        "temperatures": cfg["temperatures"],
        "answers": list(c.ANSWERS),
        "depths": list(c.DEPTH_CODES),
        "max_replicates": MAX_REPLICATES,
        "has_key": bool(os.environ.get("OPENROUTER_API_KEY")),
        "cot_max_tokens": c.COT_MAX_TOKENS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "depth_instruction": c.DEPTH_INSTRUCTION,
    })


@app.post("/api/run")
def start_run():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENROUTER_API_KEY is not set in the server's "
                                 "environment. Restart with it exported."}), 400
    body = request.get_json(force=True)
    replicates = int(body.get("replicates", 1))
    if not 1 <= replicates <= MAX_REPLICATES:
        return jsonify({"error": f"replicates must be 1-{MAX_REPLICATES}"}), 400
    scenario_text = (body.get("scenario_text") or "").strip()
    if not scenario_text:
        return jsonify({"error": "scenario text is empty"}), 400
    if body.get("depth_instruction"):
        scenario_text = c.with_depth_instruction(scenario_text)
    model = body.get("model")
    if not model:
        return jsonify({"error": "no model selected"}), 400

    # `or` would turn an explicit 0 into the default, silently accepting a
    # value the caller meant literally. Absent means default; present means
    # validated.
    raw_max = body.get("max_tokens")
    max_tokens = c.COT_MAX_TOKENS if raw_max in (None, "") else int(raw_max)
    if not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
        return jsonify({"error": f"max_tokens must be 1-{MAX_OUTPUT_TOKENS}"}), 400

    jobs = build_playground_jobs(
        model=model,
        scenario_text=scenario_text,
        system_text=body.get("system_text") or "",
        temperature=float(body.get("temperature", 0.8)),
        replicates=replicates,
        mode=body.get("mode", "cot"),
        scenario_label=body.get("scenario_label") or "custom",
        persona_label=body.get("persona_label") or "custom",
        max_tokens=max_tokens,
        provider=body.get("provider"),
    )
    if len(jobs) > MAX_TOTAL_CALLS:
        return jsonify({"error": f"refusing {len(jobs)} calls in one batch"}), 400

    token = uuid.uuid4().hex[:12]
    with _runs_lock:
        _runs[token] = {"state": "running", "total": len(jobs), "results": [],
                        "model": model, "records": 0}
    threading.Thread(target=_execute, args=(token, jobs, api_key), daemon=True).start()
    return jsonify({"token": token, "total": len(jobs)})


@app.get("/api/run/<token>")
def run_status(token: str):
    with _runs_lock:
        state = _runs.get(token)
    if not state:
        return jsonify({"error": "unknown run"}), 404
    return jsonify(state)


@app.get("/api/providers/<path:model>")
def providers(model: str):
    """Providers OpenRouter lists for one model, so the form can offer them.

    Cached per process: a catalogue lookup, not experimental data. Failure is
    non-fatal — the form falls back to pinned/any.
    """
    if model not in _ENDPOINT_CACHE:
        try:
            import httpx
            endpoints = httpx.get(
                f"https://openrouter.ai/api/v1/models/{model}/endpoints",
                timeout=15).json()["data"]["endpoints"]
            _ENDPOINT_CACHE[model] = [
                {"name": e["provider_name"],
                 "slug": e["provider_name"].lower().replace(" ", "-"),
                 "quantization": e.get("quantization"),
                 "logprobs": "logprobs" in (e.get("supported_parameters") or [])}
                for e in endpoints]
        except Exception:
            _ENDPOINT_CACHE[model] = []
    pin = c.provider_pin(model, "cot")
    return jsonify({"providers": _ENDPOINT_CACHE[model],
                    "pinned": (pin or {}).get("only", [None])[0]})


@app.get("/api/playground")
def playground_history():
    rows = [_record_view(r) for r in export.read_jsonl(c.PLAYGROUND_PATH)]
    rows.reverse()
    return jsonify({"rows": rows, "path": str(c.PLAYGROUND_PATH)})


@app.get("/api/cells")
def cells():
    """The real grid, joined and filtered. `q` searches response text."""
    rows = export.build_rows(c.RAW_PATH, c.ANNOTATED_PATH)
    for key in ("model", "scenario", "persona", "mode", "answer", "depth", "provider"):
        want = request.args.get(key)
        if want:
            rows = [r for r in rows if str(r.get(key)) == want]
    temperature = request.args.get("temperature")
    if temperature:
        rows = [r for r in rows if str(r.get("temperature")) == temperature]
    query = (request.args.get("q") or "").lower()
    if query:
        rows = [r for r in rows if query in (r.get("response_text") or "").lower()]

    offset = int(request.args.get("offset", 0))
    limit = min(int(request.args.get("limit", 50)), 500)
    facets = {}
    for key in ("model", "scenario", "persona", "mode", "answer", "depth",
                "provider", "temperature"):
        facets[key] = sorted({str(r.get(key)) for r in rows if r.get(key) is not None})
    return jsonify({
        "total": len(rows),
        "rows": rows[offset:offset + limit],
        "facets": facets,
        "raw_exists": c.RAW_PATH.exists(),
        "annotated_exists": c.ANNOTATED_PATH.exists(),
    })


@app.get("/api/summary")
def summary():
    path = c.OUT_DIR / "summary.md"
    figures = sorted(p.name for p in c.OUT_DIR.glob("*.png"))
    return jsonify({
        "markdown": path.read_text(encoding="utf-8") if path.exists() else "",
        "figures": figures,
    })


@app.get("/figures/<path:name>")
def figure(name: str):
    return send_from_directory(c.OUT_DIR, name)


def main() -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("!! OPENROUTER_API_KEY not set — the viewer works, the playground "
              "cannot call anything.")
    print(f"playground writes to {c.PLAYGROUND_PATH} (never {c.RAW_PATH.name})")
    print("http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)


if __name__ == "__main__":
    main()
