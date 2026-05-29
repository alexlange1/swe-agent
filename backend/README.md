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

With no credentials the service boots, runs an initial scan on the deterministic demo
provider, and serves data immediately. Add keys (`TAOSTATS_API_KEY`, `GITHUB_TOKEN`,
`OPENROUTER_API_KEY`, ...) to progressively replace demo data with real feeds.

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
  models.py        SQLModel tables + API schemas
  scoring.py       the aGap composite engine
  ai.py            LLM feed summaries + Oracle (with deterministic fallback)
  registry/        netuid -> github/hf/social mapping
  providers/       chain/dev/social/whale adapters (+ demo fallback)
  services/scan.py scan orchestration
  routers/         REST endpoints
```
