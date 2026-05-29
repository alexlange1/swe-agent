"""Provider interfaces and the normalised data shapes they emit.

A provider owns *one concern* (chain, dev, social, whale, wallet). The scan
service composes whichever concrete providers are available. There is no synthetic
fallback: a concern with no provider yields no data (honest n/a) rather than fake data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable


@dataclass
class ChainData:
    """On-chain / market truth for one subnet.

    Identity fields (name/symbol/github/...) come straight from the chain
    (``SubnetIdentitiesV3`` + ``TokenSymbol``) so they are authoritative.
    """
    netuid: int
    price_tao: float = 0.0
    price_change_24h: float = 0.0
    market_cap_tao: float = 0.0
    liquidity_tao: float = 0.0
    emission_share: float = 0.0
    emission_change: float = 0.0
    volume_24h_tao: float = 0.0
    validators: int = 0
    miners: int = 0
    max_validators: int = 0
    nakamoto_coefficient: int = 0
    net_tao_flow: float = 0.0  # SubnetProtocolFlow, TAO; positive = capital inflow

    # Lifecycle / economics
    registration_cost_tao: float = 0.0  # Burn(netuid)
    age_days: float = 0.0               # from NetworkRegisteredAt
    tempo: int = 0

    # On-chain identity
    name: str | None = None
    symbol: str | None = None
    github: str | None = None
    discord: str | None = None
    url: str | None = None
    owner: str | None = None
    description: str | None = None


@dataclass
class DevEvent:
    """A single development event worth surfacing (commit burst, release, model push)."""
    netuid: int
    title: str
    detail: str = ""
    source_url: str = ""
    occurred_at: datetime | None = None
    raw_text: str = ""  # fed to the AI summariser


@dataclass
class DevData:
    netuid: int
    commits_7d: int = 0
    contributors_7d: int = 0
    releases_30d: int = 0
    last_commit_at: datetime | None = None
    events: list[DevEvent] = field(default_factory=list)


@dataclass
class SocialData:
    netuid: int
    mentions_24h: int = 0
    engagement: int = 0
    heat_score: float = 0.0
    going_viral: bool = False


@dataclass
class WhaleEventData:
    netuid: int
    wallet: str
    wallet_label: str = "unknown"
    direction: str = "buy"
    amount_tao: float = 0.0


@dataclass
class WhaleData:
    netuid: int
    buy_sell_ratio: float = 1.0
    smart_money_flow: float = 0.0
    is_accumulating: bool = False
    events: list[WhaleEventData] = field(default_factory=list)


@runtime_checkable
class ChainProvider(Protocol):
    name: str
    def chain_data(self, netuids: list[int]) -> dict[int, ChainData]: ...


@runtime_checkable
class DevProvider(Protocol):
    name: str
    def dev_data(self, netuids: list[int]) -> dict[int, DevData]: ...


@runtime_checkable
class SocialProvider(Protocol):
    name: str
    def social_data(self, netuids: list[int]) -> dict[int, SocialData]: ...


@runtime_checkable
class WhaleProvider(Protocol):
    name: str
    def whale_data(self, netuids: list[int]) -> dict[int, WhaleData]: ...


@runtime_checkable
class WalletProvider(Protocol):
    name: str
    def wallet_portfolio(self, address: str) -> dict | None: ...
