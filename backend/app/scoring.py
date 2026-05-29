"""The aGap scoring engine.

One question: *is this subnet undervalued by the market right now?*

We compute four 0-100 pillar scores and blend them. Each pillar is framed so that
**higher = more bullish for the "alpha gap" thesis**:

  * development   — how hard is the team shipping?
  * market_gap    — how much has price NOT yet caught up to that shipping?
  * awareness     — how *hidden* is it? (low market awareness => high score)
  * smart_money   — are insiders / the network rotating value in quietly?

aGap = weighted blend. The weights are the product's opinion and live in one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .providers.base import ChainData, DevData, SocialData, WhaleData

WEIGHTS = {
    "development": 0.30,
    "market_gap": 0.25,
    "awareness": 0.20,
    "smart_money": 0.25,
}


@dataclass
class Scores:
    development: float
    market_gap: float
    awareness: float
    smart_money: float
    agap: float


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _scale(value: float, soft_max: float) -> float:
    """Map [0, soft_max] -> [0, 100] with gentle saturation past soft_max."""
    if soft_max <= 0:
        return 0.0
    ratio = value / soft_max
    if ratio <= 1.0:
        return ratio * 85.0
    # diminishing returns above the soft max
    return _clamp(85.0 + (ratio - 1.0) * 15.0, 0.0, 100.0)


def development_score(dev: DevData) -> float:
    commits = _scale(dev.commits_7d, 40)
    contributors = _scale(dev.contributors_7d, 6)
    releases = _scale(dev.releases_30d, 3)
    recency = 0.0
    if dev.last_commit_at:
        last = dev.last_commit_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        hours = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        recency = _clamp(100.0 - hours * 0.8)  # fresh commits score high
    return _clamp(0.45 * commits + 0.2 * contributors + 0.15 * releases + 0.2 * recency)


def awareness_level(social: SocialData) -> float:
    """Internal: how aware the market already is (0 = invisible, 100 = saturated)."""
    mentions = _scale(social.mentions_24h, 800)
    heat = _clamp(social.heat_score)
    return _clamp(0.6 * mentions + 0.4 * heat)


def awareness_score(social: SocialData) -> float:
    """Displayed pillar: hidden-ness. Low awareness => high score."""
    return _clamp(100.0 - awareness_level(social))


def market_gap_score(dev_score: float, chain: ChainData) -> float:
    """High when the team is shipping but price hasn't run yet.

    Start from the development score (the fundamental) and subtract recent price
    appreciation: the more price already moved, the more the gap has closed.
    """
    appreciation_penalty = _clamp(max(0.0, chain.price_change_24h) * 2.2, 0.0, 60.0)
    return _clamp(dev_score - appreciation_penalty + 10.0)


def smart_money_score(whale: WhaleData, chain: ChainData) -> float:
    ratio_component = _scale(max(0.0, whale.buy_sell_ratio - 1.0), 2.5)
    flow_component = _clamp(whale.smart_money_flow)
    # The network rotating emission toward a subnet is an insider/consensus signal.
    emission_component = _clamp(max(0.0, chain.emission_change) * 2.5, 0.0, 100.0)
    return _clamp(0.45 * ratio_component + 0.3 * flow_component + 0.25 * emission_component)


def compute_scores(
    chain: ChainData, dev: DevData, social: SocialData, whale: WhaleData
) -> Scores:
    development = development_score(dev)
    awareness = awareness_score(social)
    market_gap = market_gap_score(development, chain)
    smart_money = smart_money_score(whale, chain)
    agap = (
        WEIGHTS["development"] * development
        + WEIGHTS["market_gap"] * market_gap
        + WEIGHTS["awareness"] * awareness
        + WEIGHTS["smart_money"] * smart_money
    )
    return Scores(
        development=round(development, 1),
        market_gap=round(market_gap, 1),
        awareness=round(awareness, 1),
        smart_money=round(smart_money, 1),
        agap=round(_clamp(agap), 1),
    )
