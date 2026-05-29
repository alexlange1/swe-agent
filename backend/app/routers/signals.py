"""AI intelligence feed endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from ..db import get_session
from ..models import Signal, SignalOut

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("", response_model=list[SignalOut])
def list_signals(
    kind: str | None = Query(None),
    netuid: int | None = Query(None),
    limit: int = Query(50, le=200),
    session: Session = Depends(get_session),
) -> list[SignalOut]:
    stmt = select(Signal)
    if kind:
        stmt = stmt.where(Signal.kind == kind)
    if netuid is not None:
        stmt = stmt.where(Signal.netuid == netuid)
    stmt = stmt.order_by(Signal.signal_strength.desc(), Signal.created_at.desc()).limit(limit)
    rows = session.exec(stmt).all()
    return [
        SignalOut(
            id=r.id or 0, netuid=r.netuid, subnet_name=r.subnet_name, kind=r.kind,
            title=r.title, summary=r.summary, what_built=r.what_built,
            why_matters=r.why_matters, simple_terms=r.simple_terms,
            alpha_take=r.alpha_take, signal_strength=r.signal_strength,
            source_url=r.source_url, created_at=r.created_at,
        )
        for r in rows
    ]
