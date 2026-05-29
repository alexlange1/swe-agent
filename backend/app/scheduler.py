"""Background scan scheduler."""
from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from .config import Settings
from .services.scan import run_scan

log = logging.getLogger("alpha.scheduler")
_scheduler: BackgroundScheduler | None = None


def start_scheduler(settings: Settings) -> None:
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        lambda: run_scan(settings),
        "interval",
        minutes=settings.scan_interval_minutes,
        id="periodic_scan",
        max_instances=1,
        coalesce=True,
    )
    _scheduler.start()
    log.info("scheduler started: every %d min", settings.scan_interval_minutes)


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
