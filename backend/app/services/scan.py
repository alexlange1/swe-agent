"""Scan orchestration — REAL data only.

A scan:
  1. pull every subnet's on-chain state from the Subtensor chain (required),
  2. pull real development activity from GitHub for on-chain repos (best-effort),
  3. compute REAL 24h price/emission deltas from stored history,
  4. score with the aGap engine (awareness pillar excluded unless a social source exists),
  5. persist snapshots, history, and signals derived from real events.

No synthetic data: if the chain is unreachable the scan raises and the previous
snapshot is left untouched.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlmodel import Session, col, delete, select

from ..ai import summarise_event
from ..config import Settings, get_settings
from ..db import engine
from ..models import PriceHistory, Signal, SubnetSnapshot
from ..providers import resolve_providers
from ..providers.base import ChainData, DevData
from ..scoring import compute_scores

log = logging.getLogger("alpha.scan")

_LAST_SCAN: datetime | None = None
_HISTORY_RETENTION_DAYS = 3
_DELTA_WINDOW_HOURS = 24


def last_scan_time() -> datetime | None:
    return _LAST_SCAN


def _compute_deltas(
    session: Session, netuids: list[int]
) -> dict[int, tuple[float, float]]:
    """Return netuid -> (price_change_pct, emission_change_pct) from stored history."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=_HISTORY_RETENTION_DAYS)
    rows = session.exec(
        select(PriceHistory).where(PriceHistory.ts >= cutoff).order_by(PriceHistory.ts)
    ).all()
    by_netuid: dict[int, list[PriceHistory]] = {}
    for r in rows:
        by_netuid.setdefault(r.netuid, []).append(r)

    target = datetime.now(timezone.utc) - timedelta(hours=_DELTA_WINDOW_HOURS)
    out: dict[int, tuple[float, float]] = {}
    for n in netuids:
        hist = by_netuid.get(n)
        if not hist:
            out[n] = (0.0, 0.0)
            continue
        # closest row at or before the 24h mark; else the oldest we have
        ref = None
        for r in hist:
            if r.ts <= target:
                ref = r
        ref = ref or hist[0]
        price_chg = (
            ((hist[-1].price_tao - ref.price_tao) / ref.price_tao * 100.0)
            if ref.price_tao else 0.0
        )
        emis_chg = (
            ((hist[-1].emission_share - ref.emission_share) / ref.emission_share * 100.0)
            if ref.emission_share else 0.0
        )
        out[n] = (round(price_chg, 2), round(emis_chg, 2))
    return out


def run_scan(settings: Settings | None = None, max_signals_with_ai: int = 12) -> int:
    global _LAST_SCAN
    settings = settings or get_settings()
    providers = resolve_providers(settings)

    # 1. On-chain truth (required).
    netuid_range = list(range(1, settings.subnet_count + 1))
    chain: dict[int, ChainData] = providers.chain.chain_data(netuid_range)
    chain = {n: c for n, c in chain.items() if n >= 1}
    netuids = sorted(chain.keys())

    # 2. Real development activity for on-chain GitHub repos.
    dev: dict[int, DevData] = {}
    if providers.dev is not None:
        repos = {n: c.github for n, c in chain.items() if c.github}
        try:
            dev = providers.dev.dev_data_for_repos(repos)
        except Exception as exc:  # noqa: BLE001
            log.warning("github dev scan failed: %s", exc)

    social = {}  # awareness pillar: only when a social provider is configured

    with Session(engine) as session:
        deltas = _compute_deltas(session, netuids)

        snapshots: list[SubnetSnapshot] = []
        history_rows: list[PriceHistory] = []
        candidate_signals: list[tuple[int, str, object]] = []

        for n in netuids:
            c = chain[n]
            price_chg, emis_chg = deltas.get(n, (0.0, 0.0))
            c.price_change_24h = price_chg
            c.emission_change = emis_chg
            d = dev.get(n)
            sc = compute_scores(c, d, social.get(n))

            is_inflow = c.net_tao_flow > 0 and sc.smart_money >= 60

            snapshots.append(SubnetSnapshot(
                netuid=n, name=c.name or f"Subnet {n}", symbol=c.symbol or "α",
                price_tao=c.price_tao, price_change_24h=price_chg,
                market_cap_tao=c.market_cap_tao, liquidity_tao=c.liquidity_tao,
                emission_share=c.emission_share, emission_change=emis_chg,
                volume_24h_tao=c.volume_24h_tao, net_tao_flow=c.net_tao_flow,
                github=c.github, url=c.url, owner=c.owner, description=c.description,
                validators=c.validators, miners=c.miners,
                nakamoto_coefficient=c.nakamoto_coefficient,
                commits_7d=(d.commits_7d if d else 0),
                contributors_7d=(d.contributors_7d if d else 0),
                releases_30d=(d.releases_30d if d else 0),
                last_commit_at=(d.last_commit_at if d else None),
                mentions_24h=0, social_engagement=0, heat_score=0.0,
                buy_sell_ratio=0.0, smart_money_flow=c.net_tao_flow,
                is_whale_accumulating=is_inflow,
                agap_score=sc.agap, score_development=sc.development,
                score_market_gap=sc.market_gap,
                score_awareness=(sc.awareness or 0.0),
                score_smart_money=sc.smart_money,
                awareness_available=sc.available["awareness"],
            ))
            history_rows.append(PriceHistory(
                netuid=n, price_tao=c.price_tao, emission_share=c.emission_share,
            ))

            if d:
                for ev in d.events:
                    strength = int(min(100, sc.development * 0.6 + 30))
                    candidate_signals.append((strength, "development", (n, c.name or f"SN{n}", ev)))
            if emis_chg >= 10:
                candidate_signals.append((int(min(100, 50 + emis_chg)), "emission", (n, c.name or f"SN{n}", c)))
            if is_inflow and c.net_tao_flow > 0:
                strength = int(min(100, 50 + sc.smart_money * 0.4))
                candidate_signals.append((strength, "flow", (n, c.name or f"SN{n}", c)))

        candidate_signals.sort(key=lambda t: t[0], reverse=True)
        signals = _build_signals(settings, candidate_signals, max_signals_with_ai)

        # Persist (replace current snapshot + signals; APPEND history).
        session.exec(delete(SubnetSnapshot))
        session.exec(delete(Signal))
        session.add_all(snapshots)
        session.add_all(signals)
        session.add_all(history_rows)
        # prune old history
        cutoff = datetime.now(timezone.utc) - timedelta(days=_HISTORY_RETENTION_DAYS)
        session.exec(delete(PriceHistory).where(col(PriceHistory.ts) < cutoff))
        session.commit()

    _LAST_SCAN = datetime.now(timezone.utc)
    log.info("scan complete: %d subnets, %d dev repos, %d signals",
             len(snapshots), len(dev), len(signals))
    return len(snapshots)


def _build_signals(settings, candidates, ai_budget: int) -> list[Signal]:
    signals: list[Signal] = []
    for strength, kind, payload in candidates[:60]:
        n, name, obj = payload
        if kind == "development":
            ev = obj
            if ai_budget > 0:
                breakdown = summarise_event(settings, ev, name)
                ai_budget -= 1
            else:
                breakdown = {
                    "what_built": ev.title, "why_matters": ev.detail,
                    "simple_terms": f"The {name} team pushed new code.",
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
                title=f"Emission weight up {c.emission_change:.0f}%",
                summary="Network value rotating in.", signal_strength=strength,
                why_matters="A rising emission/price weight means the network is allocating more value to this subnet.",
                simple_terms="The network is paying this subnet more than before.",
                alpha_take="Emission shifts lead price; watch for follow-through.",
            ))
        elif kind == "flow":
            c = obj
            signals.append(Signal(
                netuid=n, subnet_name=name, kind="whale",
                title=f"Net capital inflow · {c.net_tao_flow:+.2f} τ",
                summary="Positive net TAO flow into the subnet pool.",
                signal_strength=strength,
                why_matters="Net capital flowing into the pool is real, on-chain accumulation.",
                simple_terms="More TAO is flowing into this subnet than out.",
                alpha_take="Follow the flow — capital often moves before narrative.",
            ))
    return signals
