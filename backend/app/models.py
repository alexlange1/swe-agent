"""Domain models for Alpha.

We keep two layers:
  * SQLModel tables  -> persisted scan snapshots
  * Pydantic schemas -> API response shapes (sometimes identical)

The Subnet snapshot is intentionally wide: it denormalises every pillar's
headline metrics so the leaderboard renders from a single row.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Persisted tables
# --------------------------------------------------------------------------- #
class SubnetSnapshot(SQLModel, table=True):
    __tablename__ = "subnet_snapshot"

    id: Optional[int] = Field(default=None, primary_key=True)
    netuid: int = Field(index=True)
    name: str
    symbol: str

    # On-chain / market
    price_tao: float = 0.0
    price_change_24h: float = 0.0
    market_cap_tao: float = 0.0
    liquidity_tao: float = 0.0
    emission_share: float = 0.0
    emission_change: float = 0.0
    volume_24h_tao: float = 0.0

    # Network
    validators: int = 0
    miners: int = 0
    nakamoto_coefficient: int = 0

    # Development
    commits_7d: int = 0
    contributors_7d: int = 0
    releases_30d: int = 0
    last_commit_at: Optional[datetime] = None

    # Social / awareness
    mentions_24h: int = 0
    social_engagement: int = 0
    heat_score: float = 0.0

    # Smart money
    buy_sell_ratio: float = 1.0
    smart_money_flow: float = 0.0
    is_whale_accumulating: bool = False

    # Scores
    agap_score: float = 0.0
    score_development: float = 0.0
    score_market_gap: float = 0.0
    score_awareness: float = 0.0
    score_smart_money: float = 0.0

    collected_at: datetime = Field(default_factory=_utcnow, index=True)


class Signal(SQLModel, table=True):
    __tablename__ = "signal"

    id: Optional[int] = Field(default=None, primary_key=True)
    netuid: int = Field(index=True)
    subnet_name: str
    kind: str = Field(index=True)  # development|whale|emission|discord|social|price
    title: str
    summary: str = ""
    # AI breakdown
    what_built: str = ""
    why_matters: str = ""
    simple_terms: str = ""
    alpha_take: str = ""
    signal_strength: int = 0  # 0-100
    source_url: str = ""
    created_at: datetime = Field(default_factory=_utcnow, index=True)


class WhaleEvent(SQLModel, table=True):
    __tablename__ = "whale_event"

    id: Optional[int] = Field(default=None, primary_key=True)
    netuid: int = Field(index=True)
    subnet_name: str
    wallet: str
    wallet_label: str = ""  # validator|founder|fund|exchange|unknown
    direction: str = "buy"  # buy|sell
    amount_tao: float = 0.0
    buy_sell_ratio: float = 1.0
    created_at: datetime = Field(default_factory=_utcnow, index=True)


# --------------------------------------------------------------------------- #
# API schemas
# --------------------------------------------------------------------------- #
class ScoreBreakdown(BaseModel):
    development: float
    market_gap: float
    awareness: float
    smart_money: float


class SubnetOut(BaseModel):
    netuid: int
    name: str
    symbol: str
    price_tao: float
    price_change_24h: float
    market_cap_tao: float
    emission_share: float
    emission_change: float
    volume_24h_tao: float
    validators: int
    miners: int
    nakamoto_coefficient: int
    commits_7d: int
    contributors_7d: int
    releases_30d: int
    mentions_24h: int
    heat_score: float
    buy_sell_ratio: float
    is_whale_accumulating: bool
    agap_score: float
    scores: ScoreBreakdown
    collected_at: datetime


class SignalOut(BaseModel):
    id: int
    netuid: int
    subnet_name: str
    kind: str
    title: str
    summary: str
    what_built: str
    why_matters: str
    simple_terms: str
    alpha_take: str
    signal_strength: int
    source_url: str
    created_at: datetime


class WhaleOut(BaseModel):
    netuid: int
    subnet_name: str
    wallet: str
    wallet_label: str
    direction: str
    amount_tao: float
    buy_sell_ratio: float
    created_at: datetime


class WalletPosition(BaseModel):
    netuid: int
    subnet_name: str
    symbol: str
    stake_alpha: float
    value_tao: float
    change_24h: float


class WalletPortfolio(BaseModel):
    address: str
    label: str
    total_value_tao: float
    change_24h_tao: float
    positions: list[WalletPosition]


class OracleQuery(BaseModel):
    question: str


class OracleAnswer(BaseModel):
    answer: str
    sources: list[str] = []
    grounded: bool = True


class HealthOut(BaseModel):
    status: str
    version: str
    providers: dict[str, bool]
    subnets_tracked: int
    last_scan: Optional[datetime]
