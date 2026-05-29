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

1. **Direct Subtensor RPC (default, free, no key)** — ground truth, no intermediary.
   The chain answers Substrate JSON-RPC over its public endpoint
   (`wss://entrypoint-finney.opentensor.ai:443`). We use `substrate-interface` and a
   handful of bulk `query_map` calls (one round-trip per storage item → all subnets in
   ~1s) to read everything below. **This is what the product actually runs on.**
2. **TaoStats API** (`https://api.taostats.io`) — optional, key-gated
   (`TAOSTATS_API_KEY`). Used for the things the chain doesn't index cheaply: per-wallet
   stake portfolios (wallet tracker) and per-wallet transfer labelling (whale wallets).
3. **There is no synthetic fallback.** If the chain is unreachable the scan fails loudly
   and the previous snapshot is left untouched — we never fabricate numbers.

### 2.2 What on-chain gives us (real, key-free)

Read directly from `SubtensorModule` storage:

- **Price** of each alpha token in TAO: `SubnetTAO / SubnetAlphaIn` (pool reserves).
- **Market cap & liquidity**: `price × (SubnetAlphaIn + SubnetAlphaOut)` and `SubnetTAO`.
- **Volume**: `SubnetVolume`.
- **Emission weight**: price-share across subnets (network's revealed preference).
- **Validator / miner counts**: `ValidatorPermit` (count of `true`) and `SubnetworkN`.
- **Net capital flow**: `SubnetProtocolFlow` — real net TAO flowing into each pool, our
  on-chain smart-money signal.
- **Lifecycle / economics**: registration cost (`Burn`), subnet age
  (`NetworkRegisteredAt` vs current block), and `Tempo`.
- **On-chain identity**: `SubnetIdentitiesV3` + `TokenSymbol` give the authoritative
  **name, symbol, GitHub repo, Discord, website, owner, and description** for each
  subnet. This *replaces* a hand-curated registry — the chain is the source of truth and
  is also where we get each team's GitHub repo for the development scan.
- **Decentralization (on-demand)**: the **Nakamoto coefficient** (smallest set of
  validators controlling >50% of stake) and top validators by stake share are computed
  live from `ValidatorPermit` + `Keys` + `TotalHotkeyAlpha` for a single subnet (~1-2s).
  It's run on demand (per subnet detail view), not in the bulk scan, because it needs the
  per-validator stake distribution. Real numbers or nothing — never guessed.

### 2.3 USD pricing

TAO/USD is read from CoinGecko's public endpoint (cached 5 min) to render USD values
alongside TAO. No key; returns `null` (TAO-only) on failure rather than a fake price.

---

## 3. Development signal (the leading indicator)

This is the highest-alpha feed because it moves *before* price. The hard part is the
**mapping**: which GitHub org / HF org / repo belongs to which netuid. We maintain a
curated registry (`backend/app/registry/subnets.py`) seeded from the public subnet
directory and refined over time.

The subnet → repo mapping is **read straight from the chain** (`SubnetIdentitiesV3.github_repo`);
~107 of 128 subnets publish a repo on-chain today.

### 3.1 GitHub

- **API**: REST `https://api.github.com`. Auth with `GITHUB_TOKEN` to lift the rate limit
  from 60/h to 5000/h. **Without a token we can only cover a rotating ~20 repos per cycle**
  (the anonymous limit can't cover 100+ repos), and uncovered subnets simply report zero
  development that cycle — never a fabricated number. A token unlocks full coverage.
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

## 8. What's live now vs gated

| Pillar / feature | Status with **zero keys** | Unlock |
|---|---|---|
| Prices, market cap, liquidity, volume | ✅ **Live** (chain) | — |
| Emission weight, validator/miner counts | ✅ **Live** (chain) | — |
| Net capital flow (smart-money) | ✅ **Live** (chain) | — |
| Subnet identity (name/symbol/GitHub/site) | ✅ **Live** (chain) | — |
| Development signal (commits/releases) | ⚠️ ~20 repos/cycle | `GITHUB_TOKEN` → all repos |
| 24h price/emission deltas | ⏳ Accrues after ~24h of scans | — |
| Awareness pillar (social) | 🔒 Excluded (marked `n/a`) | `X_BEARER_TOKEN` / `DISCORD_BOT_TOKEN` |
| Wallet tracker / per-wallet whales | 🔒 Returns 503 | `TAOSTATS_API_KEY` |
| AI feed summaries + Oracle answers | ⚠️ Deterministic templates | `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` |

### Credentials

| Env var | Unlocks |
|---|---|
| `SUBTENSOR_ENDPOINT` | Override the chain endpoint (default Finney) |
| `GITHUB_TOKEN` | Full-coverage, high-rate GitHub dev signal |
| `X_BEARER_TOKEN` / `DISCORD_BOT_TOKEN` | The awareness pillar |
| `TAOSTATS_API_KEY` | Wallet tracker + per-wallet whale labelling |
| `OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY` | LLM feed summaries + Oracle |

**With zero credentials the stack runs on real, live on-chain data** for the core economic
and identity signals; the remaining pillars are honestly marked `n/a`/gated rather than
filled with synthetic values, and light up as keys are added.
