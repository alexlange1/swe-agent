# Alpha — Data Acquisition Strategy

> "Deep thinking of how we get all this info."

Alpha is a **Bittensor subnet intelligence** engine. The entire product is downstream
of one question: *how do we observe everything 128+ subnet teams are doing, faster
than the market does?* This document is the map of every signal we collect, where it
comes from, how reliable it is, and how it feeds the **aGap score**.

The guiding insight (the "alpha gap"): there is a measurable lag between a team
**shipping** something meaningful and the **market pricing it in**. We instrument both
sides — *development reality* vs *market awareness* — and surface the gap.

---

## 1. The four pillars

Every subnet is scored across four orthogonal dimensions. Each pillar is fed by
specific, independently-sourced data so a weakness in one feed does not poison the score.

| Pillar | Question it answers | Primary sources |
|---|---|---|
| **Development** | How hard is the team shipping? | GitHub, HuggingFace, release feeds |
| **Market Gap** | Has price caught up to fundamentals? | On-chain price/emission vs dev momentum |
| **Awareness** | Does the market even know? | X/Twitter, Discord, news velocity |
| **Smart Money** | Are insiders accumulating? | On-chain wallet flows, stake deltas |

The **aGap score** is a weighted composite (see `backend/app/scoring.py`). High dev +
low awareness + smart-money inflow + lagging price = high aGap = *undervalued*.

---

## 2. On-chain data (the ground truth)

Bittensor is a Substrate chain. Post-dTAO, every subnet has its own alpha token with a
liquidity pool against TAO, so price/emission/market-cap are all *on-chain and public*.

### 2.1 Sources, in priority order

1. **TaoStats API** (`https://api.taostats.io`) — the fastest path. Indexed subnet
   metadata, prices, emissions, validators, historical series, wallet balances and
   transfers. Requires an API key (`TAOSTATS_API_KEY`). This is our default real
   provider because it is already indexed and rate-friendly.
2. **Direct Subtensor RPC** via the `bittensor` Python SDK — ground truth, no
   intermediary. We read `metagraph(netuid)` for stake/weights/validators, and the
   subnet pool reserves (`tao_in`, `alpha_in`) to derive price. Heavier to run (needs
   websocket access to a chain endpoint, e.g. `wss://entrypoint-finney.opentensor.ai`).
   Used as a verification/fallback layer.
3. **Demo provider** — a deterministic synthetic generator (seeded per netuid) so the
   whole stack runs locally with zero credentials. Numbers are *plausible*, not real.

### 2.2 What on-chain gives us

- **Price** of each alpha token in TAO (from pool reserves: `price = tao_in / alpha_in`).
- **Emission share** per subnet (the network's revealed preference — validators rotating
  weight onto a subnet is a *leading* fundamental signal).
- **Market cap & liquidity** per subnet.
- **Metagraph**: validator/miner counts, stake distribution, **Nakamoto coefficient**
  (centralisation red flag — a coefficient of 1 means one validator controls consensus).
- **Wallet flows**: transfers and stake deltas → the basis for whale detection.

---

## 3. Development signal (the leading indicator)

This is the highest-alpha feed because it moves *before* price. The hard part is the
**mapping**: which GitHub org / HF org / repo belongs to which netuid. We maintain a
curated registry (`backend/app/registry/subnets.py`) seeded from the public subnet
directory and refined over time.

### 3.1 GitHub

- **API**: REST `https://api.github.com` (+ optional GraphQL for efficiency). Auth with
  `GITHUB_TOKEN` to lift the rate limit from 60/h to 5000/h.
- **What we pull per repo**: commits (last 7/30d), distinct contributors, opened/merged
  PRs, tags/releases, additions/deletions. We also fetch the **commit messages and
  release notes** as raw text for the AI layer to summarise.
- **Signal**: a burst of commits + new contributors + a tagged release is a strong
  "shipping" signal. We normalise per-repo baselines so a sleepy repo waking up scores
  higher than a perpetually-busy one.

### 3.2 HuggingFace

- **API**: `https://huggingface.co/api/models?author=<org>` and per-model `lastModified`,
  downloads, likes. New model uploads / large checkpoint pushes = a deployment event.

### 3.3 Release & changelog feeds

- GitHub Releases Atom feeds and `CHANGELOG.md` diffs as a backstop when teams tag
  releases but don't push noisy commits.

---

## 4. Social & awareness signal

To measure the *gap* we need the "market awareness" side. Awareness that is **low while
dev is high** is exactly the opportunity.

### 4.1 X / Twitter

- **API**: X API v2 recent search (`/2/tweets/search/recent`) with `X_BEARER_TOKEN`,
  filtered by subnet name, `$symbol`, netuid, and a curated KOL author list.
- **Metrics**: mention count (24h), engagement (likes+RT+replies), velocity (rate of
  change), KOL participation. Velocity matters more than volume — *acceleration* is the
  tell that something is about to go viral.
- **Fallback / scale option**: a scraping worker (e.g. `snscrape`-style) when API quota
  is constrained.

### 4.2 Discord

- A read-only **bot** joined to public subnet servers monitors announcement and
  high-traffic channels. We score posts by author role (team/mod) and reaction velocity.
  Stored as `discord_buzz` events. (Requires per-server invite + `DISCORD_BOT_TOKEN`.)

### 4.3 News / aggregators

- TAO-focused newsletters, the official subnet directory changelog, and aggregators are
  polled as low-frequency awareness inputs.

---

## 5. Smart-money & whale detection

The "is someone who knows something acting on it?" signal.

- **Source**: on-chain transfers + stake-change events (TaoStats transfers endpoint or
  Subtensor events).
- **Method**: classify transactions by size. Compute a **buy/sell ratio** = (avg large
  buy size) / (avg retail sell size) over a rolling window. When large wallets accumulate
  while retail is flat/selling, we flag a 🐋. We maintain a **known-wallet labelset**
  (validators, founders, funds, exchanges) so labelled accumulation is weighted higher.
- **Wallet tracker**: for any coldkey/hotkey we resolve the full cross-subnet stake
  portfolio and 24h movement.

---

## 6. The AI layer (turning data into English)

Raw commits and on-chain numbers are useless to most investors. The **Oracle** and the
**Intelligence Feed** use an LLM (`OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY`) to:

- Summarise each meaningful dev event into 4 sections:
  **What they built · Why it matters · In simple terms · The Alpha take**.
- Power a chat ("TAO Oracle") that answers questions grounded *only* in the data we've
  collected (retrieval over the scan store), so answers cite real scores/signals.

If no LLM key is present we fall back to deterministic template summaries so the feature
still renders.

---

## 7. Collection cadence & architecture

```
            ┌─────────── scheduler (APScheduler) ───────────┐
            │  every N min: scan subnets in priority order   │
            └───────────────────────────────────────────────┘
                                │
        ┌───────────────┬───────┴────────┬────────────────┐
        ▼               ▼                ▼                ▼
   on-chain        development        social           whale
  (taostats/      (github/hf)       (x/discord)      (transfers)
   subtensor)
        └───────────────┴───────┬────────┴────────────────┘
                                ▼
                    normalise → store (SQLite)
                                ▼
                   scoring engine → aGap + sub-scores
                                ▼
                 AI layer → feed summaries + Oracle index
                                ▼
                      REST API  ──►  Next.js frontend
```

- **Storage**: SQLite via SQLModel (zero-config; swappable for Postgres in prod).
- **Resilience**: every provider has a typed interface and a demo fallback, so a single
  upstream outage degrades gracefully instead of breaking the score.
- **Freshness**: each record carries `collected_at`; the UI shows data age.

---

## 8. Credentials summary

| Env var | Unlocks | Required? |
|---|---|---|
| `TAOSTATS_API_KEY` | Real on-chain price/emission/validators/transfers | Recommended |
| `GITHUB_TOKEN` | High-rate GitHub dev signal | Recommended |
| `X_BEARER_TOKEN` | X/Twitter awareness signal | Optional |
| `DISCORD_BOT_TOKEN` | Discord buzz | Optional |
| `OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY` | AI feed summaries + Oracle | Optional |
| `SUBTENSOR_ENDPOINT` | Direct-chain verification via `bittensor` SDK | Optional |

**With zero credentials the stack runs fully on the deterministic demo provider** so the
product is explorable end-to-end, then progressively lights up real feeds as keys are added.
