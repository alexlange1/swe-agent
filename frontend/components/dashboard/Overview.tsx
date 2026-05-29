"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type Overview as OverviewT, type Mover } from "@/lib/api";

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card p-5">
      <div className="text-xs uppercase tracking-wide text-slate-500">{label}</div>
      <div className="mt-1 text-2xl font-extrabold text-white">{value}</div>
      {sub && <div className="mt-0.5 text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

function fmt(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  return n.toFixed(0);
}

function MoverRow({ m }: { m: Mover }) {
  const up = m.agap_change >= 0;
  return (
    <Link
      href={`/subnet/${m.netuid}`}
      className="flex items-center gap-3 px-4 py-2.5 transition hover:bg-white/5"
    >
      <span className="grid h-7 w-7 place-items-center rounded-lg bg-alpha-500/10 text-xs font-bold text-alpha-400">
        {m.symbol}
      </span>
      <span className="min-w-0 flex-1 truncate text-sm font-medium text-white">
        SN{m.netuid} · {m.name}
      </span>
      <span className="font-mono text-xs text-slate-400">{m.agap_score.toFixed(0)}</span>
      <span className={`w-16 text-right font-mono text-sm ${up ? "text-alpha-400" : "text-rose-400"}`}>
        {up ? "+" : ""}
        {m.agap_change.toFixed(1)}
      </span>
    </Link>
  );
}

export function Overview() {
  const [d, setD] = useState<OverviewT | null>(null);

  useEffect(() => {
    api.overview().then(setD).catch(() => setD(null));
  }, []);

  if (!d) {
    return (
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="h-24 animate-pulse rounded-2xl bg-white/5" />
        ))}
      </div>
    );
  }

  const usd = d.tao_usd;
  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Subnets tracked" value={String(d.subnets_tracked)} sub="live on-chain" />
        <Stat
          label="TAO price"
          value={usd ? `$${usd.toFixed(2)}` : "—"}
          sub="CoinGecko"
        />
        <Stat
          label="Total pool liquidity"
          value={`${fmt(d.total_liquidity_tao)} τ`}
          sub={usd ? `$${fmt(d.total_liquidity_tao * usd)}` : undefined}
        />
        <Stat
          label="Total alpha mkt cap"
          value={`${fmt(d.total_market_cap_tao)} τ`}
          sub={usd ? `$${fmt(d.total_market_cap_tao * usd)}` : undefined}
        />
        <Stat label="Avg aGap" value={d.avg_agap.toFixed(1)} sub="across all subnets" />
        <Stat
          label="Commits (7d)"
          value={fmt(d.total_commits_7d)}
          sub={`${d.subnets_with_dev} subnets shipping`}
        />
        <Stat label="Net inflow subnets" value={String(d.net_inflow_subnets)} sub="positive TAO flow" />
        <Stat label="On-chain volume" value={`${fmt(d.total_volume_tao)} τ`} />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <div className="card overflow-hidden">
          <div className="border-b border-white/5 px-4 py-3 text-sm font-semibold text-white">
            📈 Top aGap gainers (24h)
          </div>
          {d.top_gainers.length ? (
            d.top_gainers.map((m) => <MoverRow key={m.netuid} m={m} />)
          ) : (
            <div className="p-6 text-center text-sm text-slate-600">
              Movers appear once ~24h of scan history accrues.
            </div>
          )}
        </div>
        <div className="card overflow-hidden">
          <div className="border-b border-white/5 px-4 py-3 text-sm font-semibold text-white">
            📉 Top aGap losers (24h)
          </div>
          {d.top_losers.length ? (
            d.top_losers.map((m) => <MoverRow key={m.netuid} m={m} />)
          ) : (
            <div className="p-6 text-center text-sm text-slate-600">
              Movers appear once ~24h of scan history accrues.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
