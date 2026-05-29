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
In short:

- **On-chain truth** — TaoStats API / Subtensor RPC for price, emissions, validators,
  Nakamoto coefficient, and wallet/transfer flows (dTAO pools give per-subnet price).
- **Development signal** (the leading indicator) — GitHub commits/PRs/releases and
  HuggingFace model pushes, mapped to subnets via a curated registry.
- **Awareness signal** — X/Twitter + Discord velocity to measure how aware the market is.
- **Smart money** — classify on-chain transfers by size to detect whale accumulation.
- **AI layer** — an LLM turns raw commits/metrics into plain-English breakdowns and powers
  the **TAO Oracle** chat, grounded on the collected data.

**Every provider has a deterministic demo fallback, so the entire stack runs end-to-end
with zero credentials** and progressively lights up real feeds as keys are added.

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
  dashboard (leaderboard, AI feed, whale detection, wallet tracker, TAO Oracle).
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

Open <http://localhost:3000>. The backend auto-runs an initial demo scan on first boot,
so the dashboard has data immediately.

### Docker

```bash
docker compose up --build
# frontend → http://localhost:3000   backend → http://localhost:8000/docs
```

## Enabling real data

Drop any of these into `backend/.env` (all optional):

| Env var | Lights up |
|---|---|
| `TAOSTATS_API_KEY` | Real price / emissions / validators / wallet flows |
| `GITHUB_TOKEN` | Real development signal for curated subnets |
| `X_BEARER_TOKEN` / `DISCORD_BOT_TOKEN` | Real social / awareness signal |
| `OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY` | AI feed summaries + live TAO Oracle |

Real data is merged *over* demo data per subnet, so partial coverage degrades gracefully.

## Disclaimer

Alpha is an analytics tool, **not financial advice**. Crypto assets are volatile. The demo
provider generates synthetic numbers for exploration; verify with real feeds before acting.
