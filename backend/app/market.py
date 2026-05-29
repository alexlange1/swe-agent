"""TAO/USD spot price — real, free, no key.

Fetches the TAO price from CoinGecko's public endpoint with a short in-memory cache
so we can render USD values without hammering the API. Returns None on failure (the
UI then shows TAO-only) — never a fabricated price.
"""
from __future__ import annotations

import logging
import time

import httpx

log = logging.getLogger("alpha.market")

_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd"
_CACHE: dict[str, float] = {}
_CACHE_TS = 0.0
_TTL = 300.0  # 5 minutes


def tao_usd() -> float | None:
    global _CACHE_TS
    now = time.time()
    if _CACHE.get("tao_usd") and (now - _CACHE_TS) < _TTL:
        return _CACHE["tao_usd"]
    try:
        with httpx.Client(timeout=10.0) as c:
            r = c.get(_URL)
            r.raise_for_status()
            price = float(r.json()["bittensor"]["usd"])
            _CACHE["tao_usd"] = price
            _CACHE_TS = now
            return price
    except Exception as exc:  # noqa: BLE001
        log.warning("tao_usd fetch failed: %s", exc)
        return _CACHE.get("tao_usd")  # stale-if-error
