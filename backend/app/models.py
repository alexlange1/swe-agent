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

    net_tao_flow: float = 0.0

    # On-chain identity
    github: Optional[str] = None
    url: Optional[str] = None
    owner: Optional[str] = None
    description: Optional[str] = None

    # Network
    validators: int = 0
    miners: int = 0
    max_validators: int = 0
    nakamoto_coefficient: int = 0

    # Lifecycle / economics
    registration_cost_tao: float = 0.0
    age_days: float = 0.0
    tempo: int = 0

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
    awareness_available: bool = False

    collected_at: datetime = Field(default_factory=_utcnow, index=True)


class PriceHistory(SQLModel, table=True):
    """Rolling per-scan snapshot used to compute REAL 24h price/emission deltas."""
    __tablename__ = "price_history"

    id: Optional[int] = Field(default=None, primary_key=True)
    netuid: int = Field(index=True)
    price_tao: float = 0.0
    emission_share: float = 0.0
    agap_score: float = 0.0
    net_tao_flow: float = 0.0
    ts: datetime = Field(default_factory=_utcnow, index=True)


class Alert(SQLModel, table=True):
    """A user-defined alert rule evaluated against each scan."""
    __tablename__ = "alert"

    id: Optional[int] = Field(default=None, primary_key=True)
    metric: str = "agap_score"   # agap_score|price_change_24h|net_tao_flow|commits_7d|emission_change
    op: str = ">"                # > | <
    threshold: float = 70.0
    netuid: Optional[int] = None  # None = any subnet
    label: str = ""
    last_triggered_netuid: Optional[int] = None
    last_triggered_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=_utcnow)


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
    awareness: Optional[float] = None  # null when no social data source
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
    net_tao_flow: float
    github: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    validators: int
    miners: int
    max_validators: int = 0
    nakamoto_coefficient: int
    registration_cost_tao: float = 0.0
    age_days: float = 0.0
    tempo: int = 0
    commits_7d: int
    contributors_7d: int
    releases_30d: int
    mentions_24h: int
    heat_score: float
    buy_sell_ratio: float
    is_whale_accumulating: bool
    agap_score: float
    scores: ScoreBreakdown
    awareness_available: bool
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
    tao_usd: Optional[float] = None


class HistoryPoint(BaseModel):
    ts: datetime
    price_tao: float
    emission_share: float
    agap_score: float
    net_tao_flow: float


class SubnetHistory(BaseModel):
    netuid: int
    points: list[HistoryPoint]


class TopValidator(BaseModel):
    uid: int
    hotkey: str
    stake_alpha: float
    stake_pct: float


class Decentralization(BaseModel):
    netuid: int
    validators: int
    nakamoto_coefficient: int
    total_validator_stake_alpha: float
    top_validators: list[TopValidator]


class Mover(BaseModel):
    netuid: int
    name: str
    symbol: str
    agap_score: float
    agap_change: float
    price_tao: float
    price_change_24h: float


class Overview(BaseModel):
    subnets_tracked: int
    tao_usd: Optional[float]
    total_liquidity_tao: float
    total_market_cap_tao: float
    total_volume_tao: float
    total_commits_7d: int
    subnets_with_dev: int
    avg_agap: float
    net_inflow_subnets: int
    top_gainers: list[Mover]
    top_losers: list[Mover]
    last_scan: Optional[datetime]


class AlertIn(BaseModel):
    metric: str = "agap_score"
    op: str = ">"
    threshold: float = 70.0
    netuid: Optional[int] = None
    label: str = ""


class AlertOut(BaseModel):
    id: int
    metric: str
    op: str
    threshold: float
    netuid: Optional[int]
    label: str
    last_triggered_netuid: Optional[int]
    last_triggered_at: Optional[datetime]
    created_at: datetime


class AlertHit(BaseModel):
    alert_id: int
    label: str
    netuid: int
    subnet_name: str
    metric: str
    value: float
    message: str
