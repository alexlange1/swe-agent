<div align="center">

# α  Alpha

### Bittensor Subnet Intelligence

**Find the alpha gap before everyone else.**

Alpha scans thousands of data points across the entire Bittensor ecosystem — development
activity, on-chain emissions, whale flows, and social velocity — and surfaces
**undervalued subnets before the market catches on.**

</div>

---

## What is "the alpha gap"?

There is a measurable lag between a subnet team **shipping** something meaningful and the
**market pricing it in**. Alpha instruments both sides — *development reality* vs *market
awareness* — and computes the **aGap score**: a single 0–100 answer to *"is this subnet
undervalued right now?"*

The score blends four pillars (each framed so higher = more bullish for the gap thesis):

| Pillar | Question | Weight |
|---|---|---|
| **Development** | How hard is the team shipping? | 30% |
| **Market Gap** | Has price NOT yet caught up to that shipping? | 25% |
| **Awareness** | How *hidden* is it? (low market awareness ⇒ high score) | 20% |
| **Smart Money** | Are insiders / the network rotating value in quietly? | 25% |

High dev + lagging price + low awareness + smart-money inflow = a high aGap = an alpha gap.

## How we get all the info

The full data-acquisition strategy is documented in
**[`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)** — the core "deep thinking" of the project.
In short, **the data is real**:

- **On-chain truth (free, no key)** — Alpha talks Substrate JSON-RPC directly to the
  Bittensor chain (Finney) via `substrate-interface`. Bulk `query_map` calls read every
  subnet's pool reserves → **price**, **market cap**, **liquidity**, **volume**,
  **emission weight**, **validator/miner counts**, and **net capital flow**
  (`SubnetProtocolFlow`). Subnet **identity** — name, symbol, GitHub repo, Discord,
  website, owner — comes from `SubnetIdentitiesV3` on-chain (no hand-curated registry).
- **Development signal** (the leading indicator) — real GitHub commits/contributors/
  releases for each subnet's **on-chain** GitHub repo.
- **Awareness signal** — X/Twitter + Discord velocity. Gated behind a key; until then the
  awareness pillar is honestly marked `n/a` and excluded from the score.
- **Smart money** — real on-chain net TAO flow into each subnet pool now; per-wallet whale
  labelling unlocks with a TaoStats key.
- **AI layer** — an LLM turns raw commits/metrics into plain-English breakdowns and powers
  the **TAO Oracle** chat, grounded on the collected data.

**There is no synthetic data.** With zero credentials the stack runs on live on-chain data
for the core economic + identity signals; pillars that need an external feed are marked
`n/a`/gated rather than fabricated, and light up as keys are added. See the
[live-vs-gated table](docs/DATA_SOURCES.md#8-whats-live-now-vs-gated).

## Architecture

```
                 ┌───────────────────────────┐
   data sources  │  scan engine (scheduler)  │   every N minutes
   ┌──────────┐  │  chain · dev · social ·   │
   │ taostats │─►│  whale → normalise →      │
   │ github   │─►│  aGap scoring → AI summary │
   │ x/discord│─►│  → SQLite store           │
   └──────────┘  └────────────┬──────────────┘
                              │  FastAPI REST
                              ▼
                   ┌────────────────────┐
                   │  Next.js frontend  │  landing + dashboard
                   └────────────────────┘
```

- **`backend/`** — FastAPI + SQLModel. aGap scoring engine, pluggable data providers,
  scan scheduler, AI/Oracle layer, REST API. See [`backend/README.md`](backend/README.md).
- **`frontend/`** — Next.js 14 + Tailwind. Marketing site replica + a live intelligence
  dashboard: **network overview** (aggregates, TAO/USD, 24h aGap movers), leaderboard with
  **watchlist**, AI feed, capital-flow detection, wallet tracker, **alerts** (Telegram),
  TAO Oracle, and per-subnet **detail pages** with price/aGap **charts** and live
  **decentralization** analysis (Nakamoto coefficient + top validators).
- **`docs/`** — data-acquisition strategy.

## Quick start

Two terminals (or use Docker below).

**Backend**

```bash
cd backend
virtualenv .venv && source .venv/bin/activate   # or python -m venv .venv
pip install -r requirements.txt
cp .env.example .env        # optional — add keys to enable real data
uvicorn app.main:app --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
API_BASE_URL=http://localhost:8000 npm run dev   # http://localhost:3000
```

Open <http://localhost:3000>. On first boot the backend connects to the Bittensor chain
and runs an initial **real** scan (~8s), so the dashboard shows live subnet data
immediately. (Requires outbound access to the chain endpoint.)

### Docker

```bash
docker compose up --build
# frontend → http://localhost:3000   backend → http://localhost:8000/docs
```

## Unlocking the remaining feeds

Core economic + identity data is live with no keys. These unlock the rest (drop into
`backend/.env`):

| Env var | Lights up |
|---|---|
| `GITHUB_TOKEN` | Full-coverage development signal (all ~107 on-chain repos, not ~20) |
| `X_BEARER_TOKEN` / `DISCORD_BOT_TOKEN` | The awareness pillar (currently `n/a`) |
| `TAOSTATS_API_KEY` | Wallet tracker + per-wallet whale labelling |
| `OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY` | LLM feed summaries + Oracle answers |
| `SUBTENSOR_ENDPOINT` | Use a different chain endpoint (default Finney) |

Nothing is faked: a feed without its key is reported as `n/a`/gated, never synthesised.

## Disclaimer

Alpha is an analytics tool, **not financial advice**. Crypto assets are volatile. The demo
provider generates synthetic numbers for exploration; verify with real feeds before acting.
