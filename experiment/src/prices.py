"""Live OpenRouter prices, cached to `data/prices.json`.

PLAN.md §6: "Have the app fetch live prices from
`https://openrouter.ai/api/v1/models` at startup and print a dry-run estimate
before any call goes out. Prices change; hardcoded numbers go stale."
CONTRACTS.md §10 says the same, and adds: use `usage.cost` from each response
for *actual* spend; these prices are for the *pre-flight estimate*.

**Unit: USD per token, not per million.** That is OpenRouter's own unit —
`pricing.prompt` for `openai/gpt-4o-mini` is the string `"0.00000015"`, i.e.
$0.15/M. PLAN.md §6's formula divides by `1e6` because it assumes per-million
prices; with these numbers the divisor is 1. Getting that wrong is a factor of
a million in either direction, so `manifest.estimate_cost` never divides.

The cache lives under `data/`, which is gitignored. That is intended: prices
are a local, refreshable artefact, not a repo fact. Do not "fix" it by moving
the cache somewhere tracked — a stale committed price table is exactly what
PLAN.md §6 warns against.

    python -m src.prices                # print the five configured models
    python -m src.prices --refresh      # force a refetch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from src.contracts import DATA_DIR, OPENROUTER_MODELS_URL

CACHE_PATH = DATA_DIR / "prices.json"
MAX_AGE_HOURS = 24.0
FETCH_TIMEOUT = 60.0

#: Cache-file schema version. Bumped if the on-disk shape ever changes, so a
#: cache written by an older build is refetched rather than misread.
CACHE_VERSION = 1

Prices = dict[str, dict[str, float]]


def _cache_stamp() -> str:
    """ISO-8601 UTC with a Z suffix. Internal to the cache file — this is *not*
    the raw-record `ts` field, which CONTRACTS.md §4 owns."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_stamp(text: object) -> datetime | None:
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_catalogue(payload: object) -> Prices:
    """Pull `{model_id: {"prompt": usd_per_token, "completion": usd_per_token}}`
    out of an `/api/v1/models` body.

    Entries whose price is missing, unparseable or negative are dropped.
    OpenRouter prices its router meta-models (`openrouter/auto` and friends) at
    `"-1"`, meaning "depends what it routes to"; a negative price would silently
    *reduce* an estimate, so those must not survive into the table.
    """
    if not isinstance(payload, dict):
        raise ValueError("models payload is not a JSON object")
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise ValueError("models payload has no non-empty `data` list")

    prices: Prices = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        model_id = row.get("id")
        pricing = row.get("pricing")
        if not isinstance(model_id, str) or not isinstance(pricing, dict):
            continue
        pair: dict[str, float] = {}
        for field in ("prompt", "completion"):
            try:
                value = float(pricing[field])
            except (KeyError, TypeError, ValueError):
                break
            if value < 0:
                break
            pair[field] = value
        if len(pair) == 2:
            prices[model_id] = pair
    if not prices:
        raise ValueError("models payload contained no usable prices")
    return prices


def _read_cache(cache_path: Path) -> tuple[Prices | None, datetime | None]:
    try:
        blob = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(blob, dict) or blob.get("version") != CACHE_VERSION:
        return None, None
    prices = blob.get("prices")
    if not isinstance(prices, dict) or not prices:
        return None, None
    clean: Prices = {}
    for model_id, pair in prices.items():
        if (isinstance(model_id, str) and isinstance(pair, dict)
                and isinstance(pair.get("prompt"), (int, float))
                and isinstance(pair.get("completion"), (int, float))):
            clean[model_id] = {"prompt": float(pair["prompt"]),
                               "completion": float(pair["completion"])}
    if not clean:
        return None, None
    return clean, _parse_stamp(blob.get("fetched_at"))


def _write_cache(cache_path: Path, prices: Prices) -> None:
    """Write atomically, so a crash mid-write cannot leave a truncated cache
    that then reads as "no prices" on the next run."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "version": CACHE_VERSION,
        "fetched_at": _cache_stamp(),
        "source": OPENROUTER_MODELS_URL,
        "unit": "usd_per_token",
        "prices": prices,
    }
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps(blob, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(cache_path)


def _get_catalogue(timeout: float) -> object:
    headers = {"Accept": "application/json"}
    # The catalogue is public; the key is sent only if it happens to be around,
    # because authenticated callers get a friendlier rate limit.
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    response = httpx.get(OPENROUTER_MODELS_URL, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_prices(
    *,
    cache_path: Path = CACHE_PATH,
    max_age_hours: float = MAX_AGE_HOURS,
    force: bool = False,
    timeout: float = FETCH_TIMEOUT,
    now: datetime | None = None,
) -> Prices:
    """`{model_id: {"prompt": usd_per_token, "completion": usd_per_token}}`.

    Served from `data/prices.json` when that cache is younger than
    `max_age_hours`; otherwise refetched and rewritten.

    If the refetch fails but a *stale* cache exists, the stale cache is
    returned with a warning on stderr. A pre-flight estimate from
    yesterday's prices is far better than a runner that will not start
    because OpenRouter is briefly unreachable. With no cache at all the
    error propagates — an estimate from no prices is not an estimate.
    """
    cache_path = Path(cache_path)
    now = now or datetime.now(timezone.utc)

    cached, fetched_at = _read_cache(cache_path)
    if cached and not force and fetched_at is not None:
        if now - fetched_at < timedelta(hours=max_age_hours):
            return cached

    try:
        prices = parse_catalogue(_get_catalogue(timeout))
    except Exception as exc:
        if cached:
            age = "unknown age" if fetched_at is None else f"fetched {fetched_at:%Y-%m-%d %H:%M}Z"
            print(f"warning: could not refresh prices ({type(exc).__name__}: {exc}); "
                  f"using the cached table ({age}) at {cache_path}", file=sys.stderr)
            return cached
        raise

    _write_cache(cache_path, prices)
    return prices


def cache_status(cache_path: Path = CACHE_PATH,
                 now: datetime | None = None) -> tuple[datetime | None, float | None]:
    """`(fetched_at, age_hours)` for the price cache, or `(None, None)`.

    The runner prints this above the confirmation prompt: dollars from a table
    of unknown vintage are worth less than dollars you know the age of, and a
    failed refetch is otherwise invisible past the one stderr warning.
    """
    _, fetched_at = _read_cache(Path(cache_path))
    if fetched_at is None:
        return None, None
    now = now or datetime.now(timezone.utc)
    return fetched_at, (now - fetched_at).total_seconds() / 3600.0


def lookup(prices: Prices, model_id: str) -> dict[str, float] | None:
    """Price for one model id, tolerating OpenRouter's `:variant` suffixes
    (`:free`, `:nitro`, `:batch`). Exact match wins; the base id is the
    fallback."""
    pair = prices.get(model_id)
    if pair is not None:
        return pair
    base = model_id.split(":", 1)[0]
    return prices.get(base)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true",
                        help="ignore the cache age and refetch now")
    parser.add_argument("--cache", type=Path, default=CACHE_PATH)
    parser.add_argument("models", nargs="*",
                        help="model ids to print; default: everything in config/budget.yaml")
    args = parser.parse_args(argv)

    prices = fetch_prices(cache_path=args.cache, force=args.refresh)

    wanted = args.models
    if not wanted:
        import yaml

        from src.contracts import CONFIG_PATH
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        wanted = [m["id"] for m in cfg["models"]] + [cfg["annotator"]]

    print(f"{len(prices)} models priced, cache {args.cache}")
    print(f"{'model':<40} {'$/M in':>10} {'$/M out':>10}")
    for model_id in wanted:
        pair = lookup(prices, model_id)
        if pair is None:
            print(f"{model_id:<40} {'—':>10} {'—':>10}   NOT IN CATALOGUE")
        else:
            print(f"{model_id:<40} {pair['prompt'] * 1e6:>10.4f} "
                  f"{pair['completion'] * 1e6:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
