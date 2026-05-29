"""Scan orchestration.

A scan: pull every concern from the best available provider, merge real over demo
per subnet, compute scores, and persist snapshots + signals + whale events.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlmodel import Session, delete

from ..ai import summarise_event
from ..config import Settings, get_settings
from ..db import engine
from ..models import Signal, SubnetSnapshot, WhaleEvent
from ..providers import resolve_providers
from ..providers.base import ChainData, DevData, SocialData, WhaleData
from ..registry import SUBNET_REGISTRY, get_registry_entry
from ..scoring import compute_scores

log = logging.getLogger("alpha.scan")

_LAST_SCAN: datetime | None = None


def last_scan_time() -> datetime | None:
    return _LAST_SCAN


def _merge(real: dict, demo: dict) -> dict:
    """Real data wins per-netuid; demo fills the gaps."""
    merged = dict(demo)
    merged.update(real)
    return merged


def _safe(provider, method: str, netuids: list[int]) -> dict:
    if provider is None:
        return {}
    try:
        return getattr(provider, method)(netuids) or {}
    except Exception as exc:  # noqa: BLE001
        log.warning("provider %s.%s failed: %s", getattr(provider, "name", "?"), method, exc)
        return {}


def run_scan(settings: Settings | None = None, max_signals_with_ai: int = 12) -> int:
    """Execute one full scan. Returns the number of subnets written."""
    global _LAST_SCAN
    settings = settings or get_settings()
    netuids = list(SUBNET_REGISTRY.keys())[: settings.subnet_count]
    providers = resolve_providers(settings)

    chain: dict[int, ChainData] = _merge(
        _safe(providers.chain, "chain_data", netuids), providers.demo.chain_data(netuids))
    dev: dict[int, DevData] = _merge(
        _safe(providers.dev, "dev_data", netuids), providers.demo.dev_data(netuids))
    social: dict[int, SocialData] = _merge(
        _safe(providers.social, "social_data", netuids), providers.demo.social_data(netuids))
    whale: dict[int, WhaleData] = _merge(
        _safe(providers.whale, "whale_data", netuids), providers.demo.whale_data(netuids))

    snapshots: list[SubnetSnapshot] = []
    candidate_signals: list[tuple[int, str, object]] = []  # (strength, kind, event-ish)
    whale_rows: list[WhaleEvent] = []

    for n in netuids:
        meta = get_registry_entry(n)
        c, d, s, w = chain[n], dev[n], social[n], whale[n]
        scores = compute_scores(c, d, s, w)
        snapshots.append(SubnetSnapshot(
            netuid=n, name=meta.name, symbol=meta.symbol,
            price_tao=c.price_tao, price_change_24h=c.price_change_24h,
            market_cap_tao=c.market_cap_tao, liquidity_tao=c.liquidity_tao,
            emission_share=c.emission_share, emission_change=c.emission_change,
            volume_24h_tao=c.volume_24h_tao, validators=c.validators, miners=c.miners,
            nakamoto_coefficient=c.nakamoto_coefficient,
            commits_7d=d.commits_7d, contributors_7d=d.contributors_7d,
            releases_30d=d.releases_30d, last_commit_at=d.last_commit_at,
            mentions_24h=s.mentions_24h, social_engagement=s.engagement,
            heat_score=s.heat_score, buy_sell_ratio=w.buy_sell_ratio,
            smart_money_flow=w.smart_money_flow, is_whale_accumulating=w.is_accumulating,
            agap_score=scores.agap, score_development=scores.development,
            score_market_gap=scores.market_gap, score_awareness=scores.awareness,
            score_smart_money=scores.smart_money,
        ))

        # Dev events -> candidate development signals.
        for ev in d.events:
            strength = int(min(100, scores.development * 0.6 + 30))
            candidate_signals.append((strength, "development", (n, meta.name, ev)))

        # Emission spike signal.
        if c.emission_change >= 18:
            candidate_signals.append((
                int(min(100, 50 + c.emission_change)), "emission", (n, meta.name, c)))

        # Going viral / social signal.
        if s.going_viral:
            candidate_signals.append((int(min(100, s.heat_score)), "social", (n, meta.name, s)))

        # Whale events.
        if w.is_accumulating:
            candidate_signals.append((
                int(min(100, 50 + w.buy_sell_ratio * 12)), "whale", (n, meta.name, w)))
            for we in w.events:
                whale_rows.append(WhaleEvent(
                    netuid=n, subnet_name=meta.name, wallet=we.wallet,
                    wallet_label=we.wallet_label, direction=we.direction,
                    amount_tao=we.amount_tao, buy_sell_ratio=w.buy_sell_ratio,
                ))

    candidate_signals.sort(key=lambda t: t[0], reverse=True)

    signals: list[Signal] = []
    ai_budget = max_signals_with_ai
    for strength, kind, payload in candidate_signals[:60]:
        n, name, obj = payload
        if kind == "development":
            ev = obj
            if ai_budget > 0:
                breakdown = summarise_event(settings, ev, name)
                ai_budget -= 1
            else:
                breakdown = {
                    "what_built": ev.title, "why_matters": ev.detail,
                    "simple_terms": f"The {name} team shipped something new.",
                    "alpha_take": "Dev activity often precedes a re-rating.",
                }
            signals.append(Signal(
                netuid=n, subnet_name=name, kind=kind, title=ev.title,
                summary=ev.detail, source_url=ev.source_url, signal_strength=strength,
                **breakdown,
            ))
        elif kind == "emission":
            c = obj
            signals.append(Signal(
                netuid=n, subnet_name=name, kind=kind,
                title=f"Emission share up {c.emission_change:.0f}%",
                summary="Network weight rotating in.", signal_strength=strength,
                why_matters="Validators rotating emission toward a subnet is a leading fundamental signal.",
                simple_terms="The network is paying this subnet more — insiders may be confident.",
                alpha_take="Emission shifts lead price; watch for accumulation.",
            ))
        elif kind == "social":
            s = obj
            signals.append(Signal(
                netuid=n, subnet_name=name, kind=kind,
                title=f"Going viral · heat {s.heat_score:.0f}",
                summary=f"{s.mentions_24h} mentions in 24h.", signal_strength=strength,
                why_matters="Rapid social acceleration can front-run a price move.",
                simple_terms="People are suddenly talking about this subnet a lot.",
                alpha_take="If dev backs the hype it can run; if not, fade it.",
            ))
        elif kind == "whale":
            w = obj
            signals.append(Signal(
                netuid=n, subnet_name=name, kind=kind,
                title=f"Whale accumulation · {w.buy_sell_ratio:.1f}x",
                summary="Large wallets buying.", signal_strength=strength,
                why_matters="Smart money accumulating before retail is a classic edge.",
                simple_terms="Big wallets are quietly buying this subnet.",
                alpha_take="Follow the smart money — but size for volatility.",
            ))

    with Session(engine) as session:
        session.exec(delete(SubnetSnapshot))
        session.exec(delete(Signal))
        session.exec(delete(WhaleEvent))
        session.add_all(snapshots)
        session.add_all(signals)
        session.add_all(whale_rows)
        session.commit()

    _LAST_SCAN = datetime.now(timezone.utc)
    log.info("scan complete: %d subnets, %d signals, %d whale events",
             len(snapshots), len(signals), len(whale_rows))
    return len(snapshots)
