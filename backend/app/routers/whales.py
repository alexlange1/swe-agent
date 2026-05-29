"""Whale detection endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from ..db import get_session
from ..models import WhaleEvent, WhaleOut

router = APIRouter(prefix="/api/whales", tags=["whales"])


@router.get("", response_model=list[WhaleOut])
def list_whale_events(
    limit: int = Query(50, le=200),
    session: Session = Depends(get_session),
) -> list[WhaleOut]:
    rows = session.exec(
        select(WhaleEvent).order_by(WhaleEvent.amount_tao.desc()).limit(limit)
    ).all()
    return [
        WhaleOut(
            netuid=r.netuid, subnet_name=r.subnet_name, wallet=r.wallet,
            wallet_label=r.wallet_label, direction=r.direction,
            amount_tao=r.amount_tao, buy_sell_ratio=r.buy_sell_ratio,
            created_at=r.created_at,
        )
        for r in rows
    ]
