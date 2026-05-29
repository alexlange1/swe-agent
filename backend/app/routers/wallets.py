"""Wallet tracker endpoints."""
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

    data = None
    if providers.wallet is not None:
        try:
            data = providers.wallet.wallet_portfolio(address)
        except Exception:
            data = None
    if data is None:
        data = providers.demo.wallet_portfolio(address)
    if data is None:
        raise HTTPException(status_code=404, detail="no positions found")

    return WalletPortfolio(
        address=data["address"], label=data["label"],
        total_value_tao=data["total_value_tao"], change_24h_tao=data["change_24h_tao"],
        positions=[WalletPosition(**p) for p in data["positions"]],
    )
