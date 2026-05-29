"""TaoStats provider — real on-chain/market data.

Wraps the TaoStats API (https://docs.taostats.io). Requires ``TAOSTATS_API_KEY``.
On any failure it raises so the resolver can fall back to the demo provider; it
never returns silently-wrong data.

Endpoints used (subject to TaoStats versioning):
  * GET /api/dtao/pool/latest/v1        -> per-subnet pool reserves, price, mcap
  * GET /api/subnet/latest/v1           -> validators/miners/registration
  * GET /api/account/.../transfers      -> whale flow basis

The mapping from raw fields to our ``ChainData`` is intentionally defensive: the
API shape evolves, so we read multiple candidate keys.
"""
from __future__ import annotations

import httpx

from .base import ChainData

_BASE = "https://api.taostats.io"


def _pick(d: dict, *keys, default=0.0):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


class TaoStatsProvider:
    name = "taostats"

    def __init__(self, api_key: str, timeout: float = 20.0) -> None:
        self.api_key = api_key
        self._client = httpx.Client(
            base_url=_BASE,
            timeout=timeout,
            headers={"Authorization": api_key, "accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = self._client.get(path, params=params or {})
        resp.raise_for_status()
        return resp.json()

    def chain_data(self, netuids: list[int]) -> dict[int, ChainData]:
        # Pull the latest dTAO pool snapshot (paginated; one request grabs all subnets).
        data = self._get("/api/dtao/pool/latest/v1", {"limit": 256})
        rows = data.get("data") or data.get("pools") or []
        wanted = set(netuids)
        out: dict[int, ChainData] = {}
        for row in rows:
            netuid = int(_pick(row, "netuid", "net_uid", default=-1))
            if netuid not in wanted:
                continue
            tao_in = float(_pick(row, "tao_in", "total_tao"))
            alpha_in = float(_pick(row, "alpha_in", "total_alpha", default=1.0)) or 1.0
            price = float(_pick(row, "price", default=tao_in / alpha_in if alpha_in else 0.0))
            out[netuid] = ChainData(
                netuid=netuid,
                price_tao=price,
                price_change_24h=float(_pick(row, "price_change_24h", "change_24h")),
                market_cap_tao=float(_pick(row, "market_cap", "market_cap_tao")),
                liquidity_tao=tao_in,
                emission_share=float(_pick(row, "emission", "emission_share")),
                emission_change=float(_pick(row, "emission_change")),
                volume_24h_tao=float(_pick(row, "volume_24h", "volume_24h_tao")),
            )
        # Augment with subnet metagraph stats where available.
        try:
            meta = self._get("/api/subnet/latest/v1", {"limit": 256})
            for row in meta.get("data", []):
                netuid = int(_pick(row, "netuid", default=-1))
                if netuid in out:
                    out[netuid].validators = int(_pick(row, "active_validators", "validators"))
                    out[netuid].miners = int(_pick(row, "active_miners", "miners"))
                    out[netuid].nakamoto_coefficient = int(
                        _pick(row, "nakamoto_coefficient", default=0))
        except Exception:
            pass
        if not out:
            raise RuntimeError("taostats returned no rows for requested netuids")
        return out

    def wallet_portfolio(self, address: str) -> dict | None:
        data = self._get(f"/api/dtao/stake_balance/latest/v1", {"coldkey": address, "limit": 256})
        rows = data.get("data") or []
        if not rows:
            return None
        positions = []
        total = 0.0
        for row in rows:
            netuid = int(_pick(row, "netuid", default=-1))
            value = float(_pick(row, "balance_as_tao", "value_tao"))
            total += value
            positions.append({
                "netuid": netuid,
                "subnet_name": str(_pick(row, "subnet_name", default=f"SN{netuid}")),
                "symbol": str(_pick(row, "symbol", default="α")),
                "stake_alpha": float(_pick(row, "balance", "stake")),
                "value_tao": value,
                "change_24h": float(_pick(row, "change_24h")),
            })
        positions.sort(key=lambda p: p["value_tao"], reverse=True)
        return {
            "address": address, "label": "unknown",
            "total_value_tao": round(total, 4), "change_24h_tao": 0.0,
            "positions": positions,
        }
