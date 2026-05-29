"""Health, status, and manual scan trigger."""
from __future__ import annotations

from fastapi import APIRouter
from sqlmodel import Session, func, select

from .. import __version__
from ..config import get_settings
from ..db import engine
from ..market import tao_usd
from ..models import HealthOut, SubnetSnapshot
from ..providers import resolve_providers
from ..services.scan import last_scan_time, run_scan

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    settings = get_settings()
    providers = resolve_providers(settings)
    with Session(engine) as session:
        count = session.exec(select(func.count()).select_from(SubnetSnapshot)).one()
    status = providers.status()
    status["llm"] = settings.has_llm
    status["telegram"] = settings.has_telegram
    return HealthOut(
        status="ok", version=__version__, providers=status,
        subnets_tracked=int(count), last_scan=last_scan_time(), tao_usd=tao_usd(),
    )


@router.post("/scan")
def trigger_scan() -> dict:
    n = run_scan()
    return {"scanned": n, "last_scan": last_scan_time()}
