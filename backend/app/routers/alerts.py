"""Alert rules CRUD + Telegram test."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..config import get_settings
from ..db import get_session
from ..models import Alert, AlertIn, AlertOut
from ..services.alerts import send_telegram

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

_ALLOWED_METRICS = {"agap_score", "price_change_24h", "net_tao_flow", "commits_7d", "emission_change"}


def _to_out(a: Alert) -> AlertOut:
    return AlertOut(
        id=a.id or 0, metric=a.metric, op=a.op, threshold=a.threshold, netuid=a.netuid,
        label=a.label, last_triggered_netuid=a.last_triggered_netuid,
        last_triggered_at=a.last_triggered_at, created_at=a.created_at,
    )


@router.get("", response_model=list[AlertOut])
def list_alerts(session: Session = Depends(get_session)) -> list[AlertOut]:
    return [_to_out(a) for a in session.exec(select(Alert).order_by(Alert.created_at)).all()]


@router.post("", response_model=AlertOut)
def create_alert(body: AlertIn, session: Session = Depends(get_session)) -> AlertOut:
    if body.metric not in _ALLOWED_METRICS:
        raise HTTPException(status_code=400, detail=f"metric must be one of {_ALLOWED_METRICS}")
    if body.op not in (">", "<"):
        raise HTTPException(status_code=400, detail="op must be '>' or '<'")
    alert = Alert(metric=body.metric, op=body.op, threshold=body.threshold,
                  netuid=body.netuid, label=body.label)
    session.add(alert)
    session.commit()
    session.refresh(alert)
    return _to_out(alert)


@router.delete("/{alert_id}")
def delete_alert(alert_id: int, session: Session = Depends(get_session)) -> dict:
    alert = session.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="alert not found")
    session.delete(alert)
    session.commit()
    return {"deleted": alert_id}


@router.get("/status")
def telegram_status() -> dict:
    return {"telegram_configured": get_settings().has_telegram}


@router.post("/test")
def test_telegram() -> dict:
    settings = get_settings()
    if not settings.has_telegram:
        raise HTTPException(
            status_code=503,
            detail="Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.",
        )
    ok = send_telegram(settings, "✅ Alpha alerts connected. You'll get pings here.")
    return {"sent": ok}
