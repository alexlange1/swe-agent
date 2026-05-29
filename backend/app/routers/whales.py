"""Capital-flow / accumulation endpoint.

Per-wallet whale labelling needs an on-chain transfer indexer (TaoStats). Until that
is configured we surface the REAL on-chain signal we do have: net TAO flow into each
subnet pool (``SubnetProtocolFlow``). Subnets with the strongest positive inflow are
where capital is actually accumulating.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, select

from ..db import get_session
from ..models import SubnetSnapshot
from .subnets import _to_out

router = APIRouter(prefix="/api/whales", tags=["whales"])


@router.get("")
def capital_flows(
    limit: int = Query(40, le=128),
    session: Session = Depends(get_session),
):
    rows = session.exec(
        select(SubnetSnapshot)
        .where(SubnetSnapshot.net_tao_flow > 0)
        .order_by(SubnetSnapshot.net_tao_flow.desc())
        .limit(limit)
    ).all()
    return {
        "signal": "net_tao_flow",
        "note": "Real on-chain net capital flow into each subnet pool. Per-wallet labelling requires a TaoStats key.",
        "subnets": [_to_out(s) for s in rows],
    }
