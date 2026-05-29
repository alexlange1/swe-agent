"""Alpha backend entrypoint."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session, func, select

from . import __version__
from .config import get_settings
from .db import engine, init_db
from .models import SubnetSnapshot
from .routers import meta, oracle, signals, subnets, wallets, whales
from .scheduler import start_scheduler, stop_scheduler
from .services.scan import run_scan

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("alpha")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    with Session(engine) as session:
        count = session.exec(select(func.count()).select_from(SubnetSnapshot)).one()
    if settings.scan_on_startup and int(count) == 0:
        log.info("empty store — running initial scan")
        try:
            run_scan(settings)
        except Exception as exc:  # noqa: BLE001
            log.error("initial scan failed: %s", exc)
    start_scheduler(settings)
    yield
    stop_scheduler()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Alpha — Bittensor Subnet Intelligence",
        version=__version__,
        description="Find the alpha gap before everyone else.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(meta.router)
    app.include_router(subnets.router)
    app.include_router(signals.router)
    app.include_router(whales.router)
    app.include_router(wallets.router)
    app.include_router(oracle.router)

    @app.get("/")
    def root() -> dict:
        return {"name": "Alpha", "version": __version__, "docs": "/docs"}

    return app


app = create_app()
