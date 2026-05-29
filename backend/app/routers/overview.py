"""Network overview + movers — aggregate real stats across all subnets."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from ..db import get_session
from ..market import tao_usd
from ..models import Mover, Overview, PriceHistory, SubnetSnapshot
from ..services.scan import last_scan_time

router = APIRouter(prefix="/api", tags=["overview"])


def _movers(session: Session) -> tuple[list[Mover], list[Mover]]:
    """aGap change over ~24h from history."""
    snaps = {s.netuid: s for s in session.exec(select(SubnetSnapshot)).all()}
    cutoff = datetime.now(timezone.utc) - timedelta(days=3)
    target = datetime.now(timezone.utc) - timedelta(hours=24)
    rows = session.exec(
        select(PriceHistory).where(PriceHistory.ts >= cutoff).order_by(PriceHistory.ts)
    ).all()
    by_netuid: dict[int, list[PriceHistory]] = {}
    for r in rows:
        by_netuid.setdefault(r.netuid, []).append(r)

    movers: list[Mover] = []
    for netuid, hist in by_netuid.items():
        s = snaps.get(netuid)
        if not s or len(hist) < 2:
            continue
        ref = None
        for r in hist:
            if r.ts <= target:
                ref = r
        ref = ref or hist[0]
        change = round(s.agap_score - ref.agap_score, 1)
        if change == 0:
            continue
        movers.append(Mover(
            netuid=netuid, name=s.name, symbol=s.symbol, agap_score=s.agap_score,
            agap_change=change, price_tao=s.price_tao, price_change_24h=s.price_change_24h,
        ))
    gainers = sorted(movers, key=lambda m: m.agap_change, reverse=True)[:6]
    losers = sorted(movers, key=lambda m: m.agap_change)[:6]
    return gainers, losers


@router.get("/overview", response_model=Overview)
def overview(session: Session = Depends(get_session)) -> Overview:
    rows = session.exec(select(SubnetSnapshot)).all()
    n = len(rows) or 1
    gainers, losers = _movers(session)
    return Overview(
        subnets_tracked=len(rows),
        tao_usd=tao_usd(),
        total_liquidity_tao=round(sum(s.liquidity_tao for s in rows), 2),
        total_market_cap_tao=round(sum(s.market_cap_tao for s in rows), 2),
        total_volume_tao=round(sum(s.volume_24h_tao for s in rows), 2),
        total_commits_7d=sum(s.commits_7d for s in rows),
        subnets_with_dev=sum(1 for s in rows if s.commits_7d > 0),
        avg_agap=round(sum(s.agap_score for s in rows) / n, 1),
        net_inflow_subnets=sum(1 for s in rows if s.net_tao_flow > 0),
        top_gainers=gainers,
        top_losers=losers,
        last_scan=last_scan_time(),
    )
