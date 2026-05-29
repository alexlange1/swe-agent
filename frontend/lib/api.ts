// Typed client for the Alpha backend. Uses relative /api paths which Next.js
// rewrites to the FastAPI service (see next.config.js).

export interface ScoreBreakdown {
  development: number;
  market_gap: number;
  awareness: number | null; // null when no social data source
  smart_money: number;
}

export interface Subnet {
  netuid: number;
  name: string;
  symbol: string;
  price_tao: number;
  price_change_24h: number;
  market_cap_tao: number;
  emission_share: number;
  emission_change: number;
  volume_24h_tao: number;
  net_tao_flow: number;
  github: string | null;
  url: string | null;
  description: string | null;
  validators: number;
  miners: number;
  max_validators: number;
  nakamoto_coefficient: number;
  registration_cost_tao: number;
  age_days: number;
  tempo: number;
  commits_7d: number;
  contributors_7d: number;
  releases_30d: number;
  mentions_24h: number;
  heat_score: number;
  buy_sell_ratio: number;
  is_whale_accumulating: boolean;
  agap_score: number;
  scores: ScoreBreakdown;
  awareness_available: boolean;
  collected_at: string;
}

export interface CapitalFlows {
  signal: string;
  note: string;
  subnets: Subnet[];
}

export interface Signal {
  id: number;
  netuid: number;
  subnet_name: string;
  kind: string;
  title: string;
  summary: string;
  what_built: string;
  why_matters: string;
  simple_terms: string;
  alpha_take: string;
  signal_strength: number;
  source_url: string;
  created_at: string;
}

export interface Whale {
  netuid: number;
  subnet_name: string;
  wallet: string;
  wallet_label: string;
  direction: string;
  amount_tao: number;
  buy_sell_ratio: number;
  created_at: string;
}

export interface WalletPosition {
  netuid: number;
  subnet_name: string;
  symbol: string;
  stake_alpha: number;
  value_tao: number;
  change_24h: number;
}

export interface WalletPortfolio {
  address: string;
  label: string;
  total_value_tao: number;
  change_24h_tao: number;
  positions: WalletPosition[];
}

export interface Health {
  status: string;
  version: string;
  providers: Record<string, boolean>;
  subnets_tracked: number;
  last_scan: string | null;
  tao_usd: number | null;
}

export interface Mover {
  netuid: number;
  name: string;
  symbol: string;
  agap_score: number;
  agap_change: number;
  price_tao: number;
  price_change_24h: number;
}

export interface Overview {
  subnets_tracked: number;
  tao_usd: number | null;
  total_liquidity_tao: number;
  total_market_cap_tao: number;
  total_volume_tao: number;
  total_commits_7d: number;
  subnets_with_dev: number;
  avg_agap: number;
  net_inflow_subnets: number;
  top_gainers: Mover[];
  top_losers: Mover[];
  last_scan: string | null;
}

export interface HistoryPoint {
  ts: string;
  price_tao: number;
  emission_share: number;
  agap_score: number;
  net_tao_flow: number;
}

export interface SubnetHistory {
  netuid: number;
  points: HistoryPoint[];
}

export interface TopValidator {
  uid: number;
  hotkey: string;
  stake_alpha: number;
  stake_pct: number;
}

export interface Decentralization {
  netuid: number;
  validators: number;
  nakamoto_coefficient: number;
  total_validator_stake_alpha: number;
  top_validators: TopValidator[];
}

export interface Alert {
  id: number;
  metric: string;
  op: string;
  threshold: number;
  netuid: number | null;
  label: string;
  last_triggered_netuid: number | null;
  last_triggered_at: string | null;
  created_at: string;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

export const api = {
  health: () => get<Health>("/api/health"),
  overview: () => get<Overview>("/api/overview"),
  subnets: (params: Record<string, string> = {}) =>
    get<Subnet[]>("/api/subnets?" + new URLSearchParams(params).toString()),
  subnet: (netuid: number) => get<Subnet>(`/api/subnets/${netuid}`),
  history: (netuid: number) => get<SubnetHistory>(`/api/subnets/${netuid}/history`),
  decentralization: (netuid: number) =>
    get<Decentralization>(`/api/subnets/${netuid}/decentralization`),
  signals: (params: Record<string, string> = {}) =>
    get<Signal[]>("/api/signals?" + new URLSearchParams(params).toString()),
  whales: () => get<CapitalFlows>("/api/whales"),
  alerts: {
    list: () => get<Alert[]>("/api/alerts"),
    create: (body: {
      metric: string;
      op: string;
      threshold: number;
      netuid?: number | null;
      label: string;
    }) => post<Alert>("/api/alerts", body),
    remove: async (id: number) => {
      const res = await fetch(`/api/alerts/${id}`, { method: "DELETE" });
      if (!res.ok) throw new Error(`delete -> ${res.status}`);
      return res.json();
    },
    telegramStatus: () => get<{ telegram_configured: boolean }>("/api/alerts/status"),
    test: () => post<{ sent: boolean }>("/api/alerts/test", {}),
  },
  wallet: (address: string) => get<WalletPortfolio>(`/api/wallets/${address}`),
  oracle: async (question: string) => {
    const res = await fetch("/api/oracle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(`oracle -> ${res.status}`);
    return res.json() as Promise<{ answer: string; sources: string[]; grounded: boolean }>;
  },
};
