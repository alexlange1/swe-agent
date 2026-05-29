"""TAO Oracle — AI chat grounded on the latest scan."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from ..ai import answer_question
from ..config import get_settings
from ..db import get_session
from ..models import OracleAnswer, OracleQuery, Signal, SubnetSnapshot

router = APIRouter(prefix="/api/oracle", tags=["oracle"])


def _build_context(session: Session, limit: int = 20) -> tuple[str, list[str]]:
    """Retrieve the most relevant scan data as grounding context."""
    top = session.exec(
        select(SubnetSnapshot).order_by(SubnetSnapshot.agap_score.desc()).limit(limit)
    ).all()
    whales = session.exec(
        select(SubnetSnapshot).where(SubnetSnapshot.net_tao_flow > 0)
        .order_by(SubnetSnapshot.net_tao_flow.desc()).limit(8)
    ).all()
    signals = session.exec(
        select(Signal).order_by(Signal.signal_strength.desc()).limit(10)
    ).all()

    lines: list[str] = ["TOP ALPHA GAPS (by aGap score):"]
    sources: list[str] = []
    for s in top:
        hidden = f", hidden {s.score_awareness}" if s.awareness_available else ""
        lines.append(
            f"  SN{s.netuid} {s.name}: aGap {s.agap_score} "
            f"(dev {s.score_development}, gap {s.score_market_gap}{hidden}, "
            f"smart-money {s.score_smart_money}), "
            f"price {s.price_tao:.4f} TAO ({s.price_change_24h:+.1f}% 24h), "
            f"commits7d {s.commits_7d}, net-flow {s.net_tao_flow:+.2f} TAO, "
            f"emission-share {s.emission_share:.4f}"
        )
        sources.append(f"SN{s.netuid}")
    lines.append("\nNET CAPITAL INFLOW (on-chain):")
    for s in whales:
        lines.append(f"  SN{s.netuid} {s.name}: net flow {s.net_tao_flow:+.2f} TAO")
    lines.append("\nLATEST SIGNALS:")
    for sig in signals:
        lines.append(f"  [{sig.kind}] SN{sig.netuid} {sig.subnet_name}: {sig.title} "
                     f"(strength {sig.signal_strength})")
    return "\n".join(lines), sources


@router.post("", response_model=OracleAnswer)
def ask(query: OracleQuery, session: Session = Depends(get_session)) -> OracleAnswer:
    context, sources = _build_context(session)
    result = answer_question(get_settings(), query.question, context, sources[:6])
    return OracleAnswer(**result)
