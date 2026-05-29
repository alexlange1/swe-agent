"""The aGap scoring engine — real data only.

One question: *is this subnet undervalued by the market right now?*

Pillars (each framed so higher = more bullish for the "alpha gap" thesis):
  * development   — how hard is the team shipping?            (GitHub, real)
  * market_gap    — has price NOT yet caught up to shipping?  (dev vs real price move)
  * awareness     — how *hidden* is it? low awareness = high  (social; OPTIONAL)
  * smart_money   — is capital quietly flowing in?            (on-chain net flow, real)

If a pillar has no data source (e.g. awareness without a social API key) it is marked
unavailable and EXCLUDED from the aGap blend, with the remaining weights renormalised.
Nothing is fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from .providers.base import ChainData, DevData, SocialData

BASE_WEIGHTS = {
    "development": 0.30,
    "market_gap": 0.25,
    "awareness": 0.20,
    "smart_money": 0.25,
}


@dataclass
class Scores:
    development: float
    market_gap: float
    smart_money: float
    awareness: float | None  # None => no social data source
    agap: float
    available: dict[str, bool] = field(default_factory=dict)


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _scale(value: float, soft_max: float) -> float:
    if soft_max <= 0:
        return 0.0
    ratio = value / soft_max
    if ratio <= 1.0:
        return _clamp(ratio * 85.0)
    return _clamp(85.0 + (ratio - 1.0) * 15.0)


def development_score(dev: DevData | None) -> float:
    if dev is None:
        return 0.0
    commits = _scale(dev.commits_7d, 40)
    contributors = _scale(dev.contributors_7d, 6)
    releases = _scale(dev.releases_30d, 3)
    recency = 0.0
    if dev.last_commit_at:
        last = dev.last_commit_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        recency = _clamp(100.0 - hours * 0.8)
    return _clamp(0.45 * commits + 0.2 * contributors + 0.15 * releases + 0.2 * recency)


def awareness_score(social: SocialData | None) -> float | None:
    """Hidden-ness. Low market awareness => high score. None when no social source."""
    if social is None:
        return None
    mentions = _scale(social.mentions_24h, 800)
    heat = _clamp(social.heat_score)
    level = _clamp(0.6 * mentions + 0.4 * heat)
    return _clamp(100.0 - level)


def market_gap_score(dev_score: float, chain: ChainData) -> float:
    """High when shipping but price hasn't run. Penalise recent appreciation."""
    appreciation_penalty = _clamp(max(0.0, chain.price_change_24h) * 2.2, 0.0, 60.0)
    return _clamp(dev_score - appreciation_penalty + 10.0)


def smart_money_score(chain: ChainData) -> float:
    """Real on-chain capital conviction.

    Combines net protocol flow relative to pool liquidity (immediate) with the
    emission-weight change over time (network rotating value in). Both are real.
    """
    liq = max(1.0, chain.liquidity_tao)
    flow_term = _clamp(chain.net_tao_flow / liq * 500.0, -45.0, 45.0)
    emission_term = _clamp(chain.emission_change, -10.0, 10.0)
    return _clamp(50.0 + flow_term + emission_term)


def compute_scores(
    chain: ChainData, dev: DevData | None, social: SocialData | None
) -> Scores:
    development = development_score(dev)
    market_gap = market_gap_score(development, chain)
    smart_money = smart_money_score(chain)
    awareness = awareness_score(social)

    available = {
        "development": True,
        "market_gap": True,
        "smart_money": True,
        "awareness": awareness is not None,
    }

    pillar_values = {
        "development": development,
        "market_gap": market_gap,
        "smart_money": smart_money,
        "awareness": awareness if awareness is not None else 0.0,
    }
    active_weight = sum(BASE_WEIGHTS[k] for k, ok in available.items() if ok)
    agap = 0.0
    for k, ok in available.items():
        if ok:
            agap += (BASE_WEIGHTS[k] / active_weight) * pillar_values[k]

    return Scores(
        development=round(development, 1),
        market_gap=round(market_gap, 1),
        smart_money=round(smart_money, 1),
        awareness=(round(awareness, 1) if awareness is not None else None),
        agap=round(_clamp(agap), 1),
        available=available,
    )
