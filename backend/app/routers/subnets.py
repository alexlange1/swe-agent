"""Subnet leaderboard + detail endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select

from ..db import get_session
from ..models import ScoreBreakdown, SubnetOut, SubnetSnapshot

router = APIRouter(prefix="/api/subnets", tags=["subnets"])

_SORT_FIELDS = {
    "agap": SubnetSnapshot.agap_score,
    "price_change": SubnetSnapshot.price_change_24h,
    "emission": SubnetSnapshot.emission_share,
    "emission_change": SubnetSnapshot.emission_change,
    "dev": SubnetSnapshot.score_development,
    "heat": SubnetSnapshot.heat_score,
    "market_cap": SubnetSnapshot.market_cap_tao,
}


def _to_out(s: SubnetSnapshot) -> SubnetOut:
    return SubnetOut(
        netuid=s.netuid, name=s.name, symbol=s.symbol, price_tao=s.price_tao,
        price_change_24h=s.price_change_24h, market_cap_tao=s.market_cap_tao,
        emission_share=s.emission_share, emission_change=s.emission_change,
        volume_24h_tao=s.volume_24h_tao, validators=s.validators, miners=s.miners,
        nakamoto_coefficient=s.nakamoto_coefficient, commits_7d=s.commits_7d,
        contributors_7d=s.contributors_7d, releases_30d=s.releases_30d,
        mentions_24h=s.mentions_24h, heat_score=s.heat_score,
        buy_sell_ratio=s.buy_sell_ratio, is_whale_accumulating=s.is_whale_accumulating,
        agap_score=s.agap_score,
        scores=ScoreBreakdown(
            development=s.score_development, market_gap=s.score_market_gap,
            awareness=s.score_awareness, smart_money=s.score_smart_money,
        ),
        collected_at=s.collected_at,
    )


@router.get("", response_model=list[SubnetOut])
def list_subnets(
    sort: str = Query("agap"),
    order: str = Query("desc"),
    limit: int = Query(200, le=512),
    whales_only: bool = Query(False),
    session: Session = Depends(get_session),
) -> list[SubnetOut]:
    field = _SORT_FIELDS.get(sort, SubnetSnapshot.agap_score)
    stmt = select(SubnetSnapshot)
    if whales_only:
        stmt = stmt.where(SubnetSnapshot.is_whale_accumulating == True)  # noqa: E712
    stmt = stmt.order_by(field.asc() if order == "asc" else field.desc()).limit(limit)
    return [_to_out(s) for s in session.exec(stmt).all()]


@router.get("/{netuid}", response_model=SubnetOut)
def get_subnet(netuid: int, session: Session = Depends(get_session)) -> SubnetOut:
    s = session.exec(
        select(SubnetSnapshot).where(SubnetSnapshot.netuid == netuid)
    ).first()
    if not s:
        raise HTTPException(status_code=404, detail="subnet not scanned yet")
    return _to_out(s)
