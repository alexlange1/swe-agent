"""Deterministic demo provider.

Generates plausible, *stable* data for every subnet from a per-netuid seed so the
entire product is explorable with zero credentials. The numbers are correlated on
purpose: some subnets are engineered to be "alpha gaps" (high dev + low awareness +
smart-money inflow + lagging price) so the scoring engine has something to find.
"""
from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta, timezone

from ..registry import SUBNET_REGISTRY, get_registry_entry
from .base import (
    ChainData,
    DevData,
    DevEvent,
    SocialData,
    WhaleData,
    WhaleEventData,
)

_LABELS = ["validator", "founder", "fund", "exchange", "unknown", "unknown"]

_DEV_TEMPLATES = [
    ("Shipped new inference API", "Deployed a v{n} inference endpoint with {x}% lower latency.",
     "perf"),
    ("Merged validator reward refactor", "Reworked the incentive mechanism; PR #{n} merged by {c} contributors.",
     "incentive"),
    ("Released model checkpoint", "Pushed a {x}B-param checkpoint to HuggingFace.",
     "model"),
    ("Testnet v{n} launch", "Opened testnet for the upcoming mainnet upgrade.",
     "testnet"),
    ("Optimised miner pipeline", "Cut miner cold-start time by {x}% in {n} commits.",
     "perf"),
]


def _seed(netuid: int, salt: str = "") -> random.Random:
    h = hashlib.sha256(f"{netuid}:{salt}".encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def _archetype(netuid: int) -> str:
    """Assign each subnet a stable behavioural archetype."""
    r = _seed(netuid, "archetype").random()
    if r < 0.18:
        return "alpha_gap"     # high dev, low awareness -> our target
    if r < 0.33:
        return "hyped"         # high awareness, modest dev
    if r < 0.45:
        return "whale_target"  # smart money moving in quietly
    if r < 0.6:
        return "dormant"       # little of anything
    return "balanced"


class DemoProvider:
    name = "demo"

    def __init__(self, netuid_max: int = 128) -> None:
        self.netuid_max = netuid_max

    # -- chain ----------------------------------------------------------------
    def chain_data(self, netuids: list[int]) -> dict[int, ChainData]:
        out: dict[int, ChainData] = {}
        for n in netuids:
            r = _seed(n, "chain")
            arc = _archetype(n)
            base_price = round(0.002 + r.random() * 0.4, 5)
            # alpha gaps under-priced; hyped over-priced momentum
            change = r.uniform(-12, 12)
            if arc == "alpha_gap":
                change = r.uniform(-6, 2)
            elif arc == "hyped":
                change = r.uniform(4, 28)
            elif arc == "whale_target":
                change = r.uniform(-2, 6)
            liquidity = round(2000 + r.random() * 60000, 2)
            emission = round(r.random() * 0.04, 5)
            emission_change = r.uniform(-20, 20)
            if arc in ("alpha_gap", "whale_target"):
                emission_change = r.uniform(2, 31)
            out[n] = ChainData(
                netuid=n,
                price_tao=base_price,
                price_change_24h=round(change, 2),
                market_cap_tao=round(base_price * (1_000_000 + r.random() * 9_000_000), 2),
                liquidity_tao=liquidity,
                emission_share=emission,
                emission_change=round(emission_change, 2),
                volume_24h_tao=round(liquidity * r.uniform(0.05, 0.9), 2),
                validators=int(40 + r.random() * 180),
                miners=int(60 + r.random() * 200),
                nakamoto_coefficient=max(1, int(r.random() * 12)),
            )
        return out

    # -- dev ------------------------------------------------------------------
    def dev_data(self, netuids: list[int]) -> dict[int, DevData]:
        out: dict[int, DevData] = {}
        now = datetime.now(timezone.utc)
        for n in netuids:
            r = _seed(n, "dev")
            arc = _archetype(n)
            intensity = {
                "alpha_gap": 1.0, "hyped": 0.35, "whale_target": 0.7,
                "dormant": 0.1, "balanced": 0.5,
            }[arc]
            commits = int(r.random() * 60 * intensity)
            contributors = max(0, int(r.random() * 8 * intensity))
            releases = int(r.random() * 4 * intensity)
            last_commit = now - timedelta(hours=r.random() * (8 if intensity > 0.6 else 240))
            events: list[DevEvent] = []
            n_events = 0 if intensity < 0.2 else r.randint(1, 3)
            for _ in range(n_events):
                title_t, detail_t, _kind = r.choice(_DEV_TEMPLATES)
                title = title_t.format(n=r.randint(1, 4))
                detail = detail_t.format(n=r.randint(1, 40), x=r.randint(8, 70),
                                         c=r.randint(1, 5))
                events.append(DevEvent(
                    netuid=n, title=title, detail=detail,
                    source_url="", occurred_at=now - timedelta(hours=r.random() * 72),
                    raw_text=f"{title}. {detail}",
                ))
            out[n] = DevData(
                netuid=n, commits_7d=commits, contributors_7d=contributors,
                releases_30d=releases, last_commit_at=last_commit, events=events,
            )
        return out

    # -- social ---------------------------------------------------------------
    def social_data(self, netuids: list[int]) -> dict[int, SocialData]:
        out: dict[int, SocialData] = {}
        for n in netuids:
            r = _seed(n, "social")
            arc = _archetype(n)
            awareness = {
                "alpha_gap": 0.15, "hyped": 1.0, "whale_target": 0.3,
                "dormant": 0.1, "balanced": 0.5,
            }[arc]
            mentions = int(r.random() * 1500 * awareness)
            engagement = int(mentions * r.uniform(2, 12))
            heat = round(min(100.0, mentions * 0.08 + r.random() * 20 * awareness), 1)
            out[n] = SocialData(
                netuid=n, mentions_24h=mentions, engagement=engagement,
                heat_score=heat, going_viral=(awareness >= 0.9 and r.random() > 0.6),
            )
        return out

    # -- whale ----------------------------------------------------------------
    def whale_data(self, netuids: list[int]) -> dict[int, WhaleData]:
        out: dict[int, WhaleData] = {}
        for n in netuids:
            r = _seed(n, "whale")
            arc = _archetype(n)
            flow = {
                "alpha_gap": 0.7, "hyped": 0.4, "whale_target": 1.0,
                "dormant": 0.1, "balanced": 0.45,
            }[arc]
            ratio = round(1.0 + r.random() * 2.6 * flow, 2)
            accumulating = ratio > 1.8
            events: list[WhaleEventData] = []
            if accumulating:
                meta = get_registry_entry(n)
                for _ in range(r.randint(1, 3)):
                    events.append(WhaleEventData(
                        netuid=n,
                        wallet="5" + hashlib.sha256(f"{n}{r.random()}".encode()).hexdigest()[:46],
                        wallet_label=r.choice(_LABELS),
                        direction="buy",
                        amount_tao=round(r.uniform(80, 1200), 2),
                    ))
            out[n] = WhaleData(
                netuid=n, buy_sell_ratio=ratio,
                smart_money_flow=round(flow * 100 * r.uniform(0.6, 1.0), 1),
                is_accumulating=accumulating, events=events,
            )
        return out

    # -- wallet ---------------------------------------------------------------
    def wallet_portfolio(self, address: str) -> dict | None:
        r = random.Random(int(hashlib.sha256(address.encode()).hexdigest()[:12], 16))
        n_positions = r.randint(2, 7)
        netuids = r.sample(list(SUBNET_REGISTRY.keys()), n_positions)
        positions = []
        total = 0.0
        change = 0.0
        for n in netuids:
            meta = get_registry_entry(n)
            value = round(r.uniform(20, 4000), 2)
            ch = round(r.uniform(-15, 25), 2)
            total += value
            change += value * ch / 100
            positions.append({
                "netuid": n, "subnet_name": meta.name, "symbol": meta.symbol,
                "stake_alpha": round(r.uniform(100, 50000), 2),
                "value_tao": value, "change_24h": ch,
            })
        positions.sort(key=lambda p: p["value_tao"], reverse=True)
        label = "unknown"
        if r.random() > 0.7:
            label = r.choice(["validator", "founder", "fund", "exchange"])
        return {
            "address": address,
            "label": label,
            "total_value_tao": round(total, 2),
            "change_24h_tao": round(change, 2),
            "positions": positions,
        }
