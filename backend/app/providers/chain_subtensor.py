"""Subtensor chain provider — REAL on-chain data, no API key required.

Connects directly to a public Bittensor chain endpoint (default Finney) and reads
every subnet's economic + identity state via bulk ``query_map`` calls. This is the
ground-truth source: prices, liquidity, volume, emission weight, validator counts,
net capital flow, and the on-chain ``SubnetIdentitiesV3`` (name, symbol, GitHub repo,
discord, url, owner).

No synthetic data: if the chain is unreachable, this provider raises and the scan
fails loudly rather than fabricating numbers.

Units: chain stores balances in rao (1 TAO = 1e9 rao) and alpha in its own rao-equiv.
Price = SubnetTAO / SubnetAlphaIn (TAO per alpha).
"""
from __future__ import annotations

import logging

from substrateinterface import SubstrateInterface

from .base import ChainData

log = logging.getLogger("alpha.chain")

RAO = 1_000_000_000
DEFAULT_ENDPOINT = "wss://entrypoint-finney.opentensor.ai:443"

# Storage items we pull in bulk (one query_map each → all subnets).
_MAPS = [
    "SubnetTAO",
    "SubnetAlphaIn",
    "SubnetAlphaOut",
    "SubnetVolume",
    "SubnetMovingPrice",
    "SubnetTaoInEmission",
    "SubnetProtocolFlow",
    "SubnetworkN",
    "MaxAllowedValidators",
    "ValidatorPermit",
    "TokenSymbol",
    "SubnetIdentitiesV3",
    "SubnetOwner",
]


def _to_str(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray, list)):
        try:
            return bytes(v).decode("utf-8", "ignore") or None
        except Exception:
            return None
    s = str(v).strip()
    return s or None


class SubtensorChainProvider:
    name = "subtensor"

    def __init__(self, endpoint: str | None = None, timeout: float = 30.0) -> None:
        self.endpoint = endpoint or DEFAULT_ENDPOINT
        self._timeout = timeout
        self._substrate: SubstrateInterface | None = None

    def _connect(self) -> SubstrateInterface:
        if self._substrate is None:
            self._substrate = SubstrateInterface(url=self.endpoint)
        return self._substrate

    def close(self) -> None:
        if self._substrate is not None:
            try:
                self._substrate.close()
            except Exception:
                pass
            self._substrate = None

    def _map(self, s: SubstrateInterface, name: str) -> dict[int, object]:
        out: dict[int, object] = {}
        for k, v in s.query_map("SubtensorModule", name, page_size=300):
            try:
                out[int(k.value)] = v.value
            except Exception:
                continue
        return out

    def chain_data(self, netuids: list[int]) -> dict[int, ChainData]:
        s = self._connect()
        data = {name: self._map(s, name) for name in _MAPS}

        tao = data["SubnetTAO"]
        alpha_in = data["SubnetAlphaIn"]
        alpha_out = data["SubnetAlphaOut"]
        volume = data["SubnetVolume"]
        emission = data["SubnetTaoInEmission"]
        flow = data["SubnetProtocolFlow"]
        neurons = data["SubnetworkN"]
        permits = data["ValidatorPermit"]
        symbols = data["TokenSymbol"]
        idents = data["SubnetIdentitiesV3"]
        owners = data["SubnetOwner"]

        # First pass: spot prices for every subnet (needed for emission weight).
        prices: dict[int, float] = {}
        for n, t in tao.items():
            ai = alpha_in.get(n) or 0
            prices[n] = (t / ai) if ai else 0.0
        total_price = sum(prices.values()) or 1.0

        wanted = set(netuids)
        out: dict[int, ChainData] = {}
        for n in tao.keys():
            if wanted and n not in wanted:
                continue
            price = prices.get(n, 0.0)
            ai = (alpha_in.get(n) or 0) / RAO
            ao = (alpha_out.get(n) or 0) / RAO
            permit = permits.get(n) or []
            validators = sum(1 for p in permit if p) if isinstance(permit, list) else 0
            total_neurons = int(neurons.get(n) or (len(permit) if isinstance(permit, list) else 0))
            ident = idents.get(n) or {}
            gh = _to_str(ident.get("github_repo")) if isinstance(ident, dict) else None

            out[n] = ChainData(
                netuid=n,
                price_tao=price,
                price_change_24h=0.0,  # filled from history by the scan service
                market_cap_tao=round(price * (ai + ao), 4),
                liquidity_tao=round((tao.get(n) or 0) / RAO, 4),
                emission_share=round(price / total_price, 6),
                emission_change=0.0,  # filled from history
                volume_24h_tao=round((volume.get(n) or 0) / RAO, 4),
                validators=validators,
                miners=max(0, total_neurons - validators),
                nakamoto_coefficient=0,  # unknown without stake distribution; 0 = n/a
                net_tao_flow=round((flow.get(n) or 0) / RAO, 4),
                name=(_to_str(ident.get("subnet_name")) if isinstance(ident, dict) else None),
                symbol=_to_str(symbols.get(n)),
                github=gh,
                discord=(_to_str(ident.get("discord")) if isinstance(ident, dict) else None),
                url=(_to_str(ident.get("subnet_url")) if isinstance(ident, dict) else None),
                owner=_to_str(owners.get(n)),
                description=(_to_str(ident.get("description")) if isinstance(ident, dict) else None),
            )
        if not out:
            raise RuntimeError("subtensor chain returned no subnets")
        log.info("subtensor: fetched %d subnets from %s", len(out), self.endpoint)
        return out
