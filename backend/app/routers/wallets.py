"""Wallet tracker endpoint.

Resolving a coldkey's full cross-subnet stake portfolio requires an on-chain stake
indexer. We use TaoStats when ``TAOSTATS_API_KEY`` is configured. Without it we return
a clear 503 rather than fabricating positions — no synthetic data.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..models import WalletPortfolio, WalletPosition
from ..providers import resolve_providers

router = APIRouter(prefix="/api/wallets", tags=["wallets"])


@router.get("/{address}", response_model=WalletPortfolio)
def wallet_portfolio(address: str) -> WalletPortfolio:
    if len(address) < 8:
        raise HTTPException(status_code=400, detail="address looks invalid")
    settings = get_settings()
    providers = resolve_providers(settings)

    if providers.wallet is None:
        raise HTTPException(
            status_code=503,
            detail="Wallet tracking requires an on-chain stake indexer. Set TAOSTATS_API_KEY to enable it.",
        )
    try:
        data = providers.wallet.wallet_portfolio(address)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"wallet lookup failed: {exc}") from exc
    if not data:
        raise HTTPException(status_code=404, detail="no positions found for that address")

    return WalletPortfolio(
        address=data["address"], label=data["label"],
        total_value_tao=data["total_value_tao"], change_24h_tao=data["change_24h_tao"],
        positions=[WalletPosition(**p) for p in data["positions"]],
    )
