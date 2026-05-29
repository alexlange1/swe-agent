"""Subnet registry: the netuid -> {name, symbol, github, hf, x} mapping.

This is the single hardest and most valuable piece of the whole system: linking a
netuid to the *places its team actually ships*. We seed it with a curated set of
well-known subnets and their public handles, then synthesise plausible entries for
the remaining netuids so the full 128-subnet surface is navigable. Real entries are
marked ``curated=True`` and take precedence whenever a real provider is queried.

Refine this file as the canonical source of truth; everything downstream (GitHub
scans, HF scans, social search terms) keys off it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SubnetMeta:
    netuid: int
    name: str
    symbol: str
    github: str | None = None          # "owner/repo" or "owner"
    huggingface: str | None = None     # HF org/author
    x_handle: str | None = None        # without leading @
    discord: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    curated: bool = False


# Curated, real-world-ish seed set. Handles reflect publicly known projects; treat
# them as a starting point to be verified/expanded by an operator.
_CURATED: list[SubnetMeta] = [
    SubnetMeta(1, "Apex", "α", "opentensor/prompting", "opentensor", "opentensor",
               tags=("llm", "inference"), curated=True),
    SubnetMeta(2, "Omron", "β", "inference-labs/omron", None, "omron_ai",
               tags=("zkml",), curated=True),
    SubnetMeta(3, "Templar", "γ", "tplr-ai/templar", None, "tplr_ai",
               tags=("pretraining",), curated=True),
    SubnetMeta(4, "Targon", "δ", "manifold-inc/targon", None, "manifoldlabs",
               tags=("inference", "llm"), curated=True),
    SubnetMeta(5, "OpenKaito", "ε", "OpenKaito/openkaito", None, "_kaitoai",
               tags=("search",), curated=True),
    SubnetMeta(6, "Infinite Games", "ζ", "amedeo-gigaver/infinite_games", None, None,
               tags=("forecasting",), curated=True),
    SubnetMeta(7, "Subvortex", "η", "eclipsevortex/SubVortex", None, "subvortex",
               tags=("infra",), curated=True),
    SubnetMeta(8, "Proprietary Trading", "θ", "taoshidev/proprietary-trading-network",
               None, "taoshiio", tags=("finance", "trading"), curated=True),
    SubnetMeta(9, "Pretraining", "ι", "macrocosm-os/pretraining", "macrocosm-os",
               "MacrocosmosAI", tags=("pretraining",), curated=True),
    SubnetMeta(10, "Sturdy", "κ", "Sturdy-Subnet/sturdy-subnet", None, "sturdyfinance",
               tags=("defi",), curated=True),
    SubnetMeta(11, "Dippy Roleplay", "λ", "impel-intelligence/dippy-bittensor-subnet",
               None, "dippyai", tags=("llm", "roleplay"), curated=True),
    SubnetMeta(12, "ComputeHorde", "μ", "backend-developers-ltd/ComputeHorde", None,
               None, tags=("compute",), curated=True),
    SubnetMeta(13, "Datauniverse", "ν", "macrocosm-os/data-universe", "macrocosm-os",
               "MacrocosmosAI", tags=("data",), curated=True),
    SubnetMeta(14, "TAOHash", "ξ", "latent-to/TAOHash", None, None,
               tags=("bitcoin", "hashrate"), curated=True),
    SubnetMeta(15, "De-Val", "ο", "deval-core/De-Val", None, None,
               tags=("evaluation",), curated=True),
    SubnetMeta(16, "BitAds", "π", "eseckft/BitAds.ai", None, None,
               tags=("ads",), curated=True),
    SubnetMeta(17, "Three Gen", "ρ", "404-Repo/three-gen-subnet", None, None,
               tags=("3d", "generative"), curated=True),
    SubnetMeta(18, "Cortex.t", "σ", "corcel-api/cortex.t", None, "corcel_io",
               tags=("inference",), curated=True),
    SubnetMeta(19, "Nineteen", "τ", "rayonlabs/nineteen", None, "rayon_labs",
               tags=("inference",), curated=True),
    SubnetMeta(20, "BitAgent", "υ", "RogueTensor/bitagent_subnet", None, None,
               tags=("agents",), curated=True),
    SubnetMeta(21, "Any-to-Any", "φ", "omegalabsinc/omegalabs-anytoany-bittensor",
               "omegalabsinc", "omegalabsai", tags=("multimodal",), curated=True),
    SubnetMeta(22, "Desearch", "χ", "Datura-ai/desearch", None, "datura_ai",
               tags=("search",), curated=True),
    SubnetMeta(23, "SocialTensor", "ψ", "NicheTensor/NicheImage", None, None,
               tags=("image",), curated=True),
    SubnetMeta(24, "Omega", "ω", "omegalabsinc/omegalabs-bittensor-subnet",
               "omegalabsinc", "omegalabsai", tags=("video", "data"), curated=True),
    SubnetMeta(25, "Mainframe", "α₂", "macrocosm-os/folding", "macrocosm-os",
               "MacrocosmosAI", tags=("science", "protein"), curated=True),
    SubnetMeta(27, "Neural Internet", "γ₂", "neuralinternet/compute-subnet", None,
               "neural_internet", tags=("compute",), curated=True),
    SubnetMeta(64, "Chutes", "β₃", "rayonlabs/chutes", None, "rayon_labs",
               tags=("compute", "serverless"), curated=True),
    SubnetMeta(77, "Liquidity", "δ₃", None, None, None,
               tags=("defi",), curated=True),
]

_SYMBOL_POOL = [
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π",
    "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω",
]
_THEMES = [
    "Inference", "Pretraining", "Compute", "Storage", "Vision", "Audio", "Agents",
    "Finance", "Forecasting", "Search", "Data", "Science", "Robotics", "Security",
]


def _synthetic(netuid: int) -> SubnetMeta:
    theme = _THEMES[netuid % len(_THEMES)]
    symbol = _SYMBOL_POOL[netuid % len(_SYMBOL_POOL)]
    return SubnetMeta(
        netuid=netuid,
        name=f"{theme} Net {netuid}",
        symbol=symbol,
        github=None,
        huggingface=None,
        x_handle=None,
        tags=(theme.lower(),),
        curated=False,
    )


def _build_registry(max_netuid: int = 128) -> dict[int, SubnetMeta]:
    reg: dict[int, SubnetMeta] = {m.netuid: m for m in _CURATED}
    for netuid in range(1, max_netuid + 1):
        if netuid not in reg:
            reg[netuid] = _synthetic(netuid)
    return dict(sorted(reg.items()))


SUBNET_REGISTRY: dict[int, SubnetMeta] = _build_registry()


def get_registry_entry(netuid: int) -> SubnetMeta:
    return SUBNET_REGISTRY.get(netuid) or _synthetic(netuid)
