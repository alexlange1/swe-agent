# Alpha — Backend

FastAPI service that scans the Bittensor ecosystem, scores subnets with the **aGap**
engine, and serves the intelligence API. See [`../docs/DATA_SOURCES.md`](../docs/DATA_SOURCES.md)
for the full data-acquisition strategy.

## Run

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional — fill in keys to light up real data
uvicorn app.main:app --reload --port 8000
```

With no credentials the service boots, connects to the Bittensor chain, and runs an
initial **real** scan immediately (needs outbound access to the chain endpoint). The core
economic + identity signals are live key-free; add keys (`GITHUB_TOKEN`, `TAOSTATS_API_KEY`,
`OPENROUTER_API_KEY`, ...) to unlock the remaining pillars. **No data is ever synthesised.**

## API

| Endpoint | Description |
|---|---|
| `GET /api/health` | Status, provider wiring, last scan time |
| `POST /api/scan` | Trigger a scan now |
| `GET /api/subnets` | Leaderboard (`?sort=agap&order=desc&whales_only=true`) |
| `GET /api/subnets/{netuid}` | Single subnet detail |
| `GET /api/signals` | AI intelligence feed (`?kind=development&netuid=19`) |
| `GET /api/whales` | Whale accumulation events |
| `GET /api/wallets/{address}` | Cross-subnet wallet portfolio |
| `POST /api/oracle` | TAO Oracle Q&A (`{"question": "..."}`) |

Interactive docs at `/docs`.

## Layout

```
app/
  config.py        settings / credentials
  models.py        SQLModel tables (snapshots + price history) + API schemas
  scoring.py       the aGap composite engine (pillars + availability)
  ai.py            LLM feed summaries + Oracle (with deterministic fallback)
  providers/
    chain_subtensor.py  REAL on-chain data via Substrate JSON-RPC (no key)
    github.py           REAL dev signal for on-chain repos
    taostats.py         optional wallet/transfer data (key)
  services/scan.py scan orchestration + real 24h deltas from price history
  routers/         REST endpoints
```
