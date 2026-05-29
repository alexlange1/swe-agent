"""Resolve which concrete provider serves each concern.

Strategy: prefer a real provider when its credential is present, but ALWAYS keep
the demo provider as a fallback so a missing key or an upstream error degrades a
single pillar instead of breaking the scan. The scan service merges real data over
demo data per subnet, so partial real coverage (e.g. GitHub only for curated
subnets) is fully supported.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from .demo import DemoProvider


@dataclass
class ProviderSet:
    demo: DemoProvider
    chain: object | None = None
    dev: object | None = None
    social: object | None = None
    whale: object | None = None
    wallet: object | None = None

    def status(self) -> dict[str, bool]:
        return {
            "chain_real": self.chain is not None,
            "dev_real": self.dev is not None,
            "social_real": self.social is not None,
            "whale_real": self.whale is not None,
            "wallet_real": self.wallet is not None,
        }


def resolve_providers(settings: Settings) -> ProviderSet:
    demo = DemoProvider(netuid_max=settings.subnet_count)
    ps = ProviderSet(demo=demo)

    if settings.taostats_api_key:
        try:
            from .taostats import TaoStatsProvider
            ts = TaoStatsProvider(settings.taostats_api_key)
            ps.chain = ts
            ps.wallet = ts
        except Exception:
            ps.chain = None

    # GitHub works (rate-limited) even without a token; only wire it when a token is
    # present to avoid hammering the anonymous limit during scans.
    if settings.github_token:
        try:
            from .github import GitHubProvider
            ps.dev = GitHubProvider(settings.github_token)
        except Exception:
            ps.dev = None

    # Social/X provider would attach here when X_BEARER_TOKEN is configured.
    # (Left as demo for now; the interface is ready in base.py.)

    return ps
