"""Alert engine.

Evaluates user-defined alert rules against the latest snapshot after each scan and,
when configured, pushes notifications to Telegram. Rules live in the DB; delivery is
optional (only fires when TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID are set).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx
from sqlmodel import Session, select

from ..config import Settings
from ..models import Alert, AlertHit, SubnetSnapshot

log = logging.getLogger("alpha.alerts")

_METRIC_FIELDS = {
    "agap_score": "agap_score",
    "price_change_24h": "price_change_24h",
    "net_tao_flow": "net_tao_flow",
    "commits_7d": "commits_7d",
    "emission_change": "emission_change",
}


def _matches(value: float, op: str, threshold: float) -> bool:
    return value > threshold if op == ">" else value < threshold


def evaluate_alerts(session: Session, settings: Settings) -> list[AlertHit]:
    alerts = session.exec(select(Alert)).all()
    if not alerts:
        return []
    snapshots = session.exec(select(SubnetSnapshot)).all()
    by_netuid = {s.netuid: s for s in snapshots}

    hits: list[AlertHit] = []
    for alert in alerts:
        field = _METRIC_FIELDS.get(alert.metric)
        if not field:
            continue
        targets = (
            [by_netuid[alert.netuid]] if alert.netuid and alert.netuid in by_netuid
            else snapshots
        )
        best: tuple[SubnetSnapshot, float] | None = None
        for s in targets:
            val = float(getattr(s, field, 0.0))
            if _matches(val, alert.op, alert.threshold):
                if best is None or (alert.op == ">" and val > best[1]) or (
                    alert.op == "<" and val < best[1]
                ):
                    best = (s, val)
        if best is None:
            continue
        s, val = best
        # de-dupe: skip if same subnet already the last trigger
        if alert.last_triggered_netuid == s.netuid:
            continue
        msg = (
            f"🔔 {alert.label or alert.metric}: SN{s.netuid} {s.name} "
            f"{alert.metric} = {val:.2f} ({alert.op} {alert.threshold})"
        )
        hits.append(AlertHit(
            alert_id=alert.id or 0, label=alert.label, netuid=s.netuid,
            subnet_name=s.name, metric=alert.metric, value=round(val, 2), message=msg,
        ))
        alert.last_triggered_netuid = s.netuid
        alert.last_triggered_at = datetime.now(timezone.utc)
        session.add(alert)
    session.commit()

    if hits and settings.has_telegram:
        for h in hits:
            send_telegram(settings, h.message)
    return hits


def send_telegram(settings: Settings, text: str) -> bool:
    if not settings.has_telegram:
        return False
    try:
        with httpx.Client(timeout=10.0) as c:
            r = c.post(
                f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
                json={"chat_id": settings.telegram_chat_id, "text": text},
            )
            return r.status_code == 200
    except Exception as exc:  # noqa: BLE001
        log.warning("telegram send failed: %s", exc)
        return False
