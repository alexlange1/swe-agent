"""Resolve concrete providers. REAL data only — there is no synthetic fallback.

  * chain   — SubtensorChainProvider (free, no key). The required ground-truth source.
  * dev     — GitHubProvider (works anonymously but rate-limited; GITHUB_TOKEN lifts it).
  * social  — only when X_BEARER_TOKEN is configured (awareness pillar). Else None.
  * wallet  — TaoStats when TAOSTATS_API_KEY is configured. Else None (gated, not faked).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from .chain_subtensor import SubtensorChainProvider


@dataclass
class ProviderSet:
    chain: SubtensorChainProvider
    dev: object | None = None
    social: object | None = None
    wallet: object | None = None

    def status(self) -> dict[str, bool]:
        return {
            "chain": True,
            "dev_github": self.dev is not None,
            "social": self.social is not None,
            "wallet_lookup": self.wallet is not None,
        }


def resolve_providers(settings: Settings) -> ProviderSet:
    chain = SubtensorChainProvider(endpoint=settings.subtensor_endpoint)
    ps = ProviderSet(chain=chain)

    try:
        from .github import GitHubProvider
        ps.dev = GitHubProvider(settings.github_token)
    except Exception:
        ps.dev = None

    if settings.taostats_api_key:
        try:
            from .taostats import TaoStatsProvider
            ps.wallet = TaoStatsProvider(settings.taostats_api_key)
        except Exception:
            ps.wallet = None

    # Social/awareness attaches here when X_BEARER_TOKEN is implemented.
    return ps
