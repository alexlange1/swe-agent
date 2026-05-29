# Alpha — Frontend

Next.js 14 (App Router) + TypeScript + Tailwind. A faithful replica of the AlphaGap
marketing site plus a **live intelligence dashboard** wired to the Alpha backend.

## Run

```bash
npm install
API_BASE_URL=http://localhost:8000 npm run dev   # http://localhost:3000
```

`API_BASE_URL` tells Next.js where to proxy `/api/*` requests (see `next.config.js`).
Defaults to `http://localhost:8000`.

## Pages

- `/` — landing page: hero with a **live** aGap leaderboard, what-we-track, the
  problem/solution narrative, how-it-works, the aGap score, features, an interactive
  **TAO Oracle** demo (calls the backend), Telegram alerts, pricing, testimonials.
- `/dashboard` — tabbed workspace:
  - **Leaderboard** — all subnets, sortable by aGap / dev / emission / price / heat,
    whale filter, search, click-through detail drawer with the full score breakdown.
  - **AI Feed** — signals expanded into the 4-section AI breakdown.
  - **Whales** — on-chain accumulation events with wallet labels.
  - **Wallet Tracker** — look up any address's cross-subnet portfolio.
  - **TAO Oracle** — grounded AI chat.
- `/subscribe` — pricing tiers; handles the Stripe-style `?canceled=true` redirect.

## Structure

```
app/            routes (landing, dashboard, subscribe) + layout + globals
components/
  landing/      hero, marketing sections, oracle demo
  dashboard/    leaderboard, feed, whales, wallet tracker, subnet detail, ui helpers
  Nav, Footer, Logo, Pricing
lib/api.ts      typed backend client
```
