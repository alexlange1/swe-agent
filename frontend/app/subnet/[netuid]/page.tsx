"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  api,
  type Decentralization,
  type Signal,
  type Subnet,
  type SubnetHistory,
} from "@/lib/api";
import { Sparkline } from "@/components/dashboard/Sparkline";
import { Pct, ScoreBar, kindIcon, timeAgo } from "@/components/dashboard/ui";
import { useWatchlist } from "@/lib/watchlist";

export default function SubnetPage({ params }: { params: { netuid: string } }) {
  const netuid = Number(params.netuid);
  const [s, setS] = useState<Subnet | null>(null);
  const [hist, setHist] = useState<SubnetHistory | null>(null);
  const [dec, setDec] = useState<Decentralization | null>(null);
  const [decLoading, setDecLoading] = useState(false);
  const [signals, setSignals] = useState<Signal[]>([]);
  const { has, toggle } = useWatchlist();

  useEffect(() => {
    api.subnet(netuid).then(setS).catch(() => {});
    api.history(netuid).then(setHist).catch(() => {});
    api.signals({ netuid: String(netuid), limit: "12" }).then(setSignals).catch(() => {});
  }, [netuid]);

  function loadDecentralization() {
    setDecLoading(true);
    api.decentralization(netuid).then(setDec).catch(() => {}).finally(() => setDecLoading(false));
  }

  if (!s) {
    return (
      <div className="container-x py-16">
        <div className="h-40 animate-pulse rounded-2xl bg-white/5" />
      </div>
    );
  }

  const prices = (hist?.points || []).map((p) => p.price_tao);
  const agaps = (hist?.points || []).map((p) => p.agap_score);

  return (
    <div className="container-x py-10">
      <Link href="/dashboard" className="text-sm text-slate-400 hover:text-white">
        ← Dashboard
      </Link>

      <div className="mt-4 flex flex-wrap items-center gap-4">
        <span className="grid h-14 w-14 place-items-center rounded-2xl bg-alpha-500/10 text-2xl font-bold text-alpha-400">
          {s.symbol}
        </span>
        <div className="flex-1">
          <h1 className="text-3xl font-extrabold tracking-tight text-white">
            SN{s.netuid} · {s.name}
          </h1>
          <div className="mt-1 text-sm text-slate-400">
            {s.price_tao.toFixed(5)} τ <Pct value={s.price_change_24h} /> · aGap{" "}
            <span className="font-mono font-bold text-alpha-400">{s.agap_score.toFixed(0)}</span>
          </div>
        </div>
        <button
          onClick={() => toggle(s.netuid)}
          className={has(s.netuid) ? "btn-primary !py-2" : "btn-ghost !py-2"}
        >
          {has(s.netuid) ? "★ Watching" : "☆ Watch"}
        </button>
        {s.github && (
          <a href={s.github} target="_blank" rel="noreferrer" className="btn-ghost !py-2">
            GitHub ↗
          </a>
        )}
        {s.url && (
          <a href={s.url} target="_blank" rel="noreferrer" className="btn-ghost !py-2">
            Website ↗
          </a>
        )}
      </div>

      {s.description && <p className="mt-4 max-w-3xl text-sm text-slate-400">{s.description}</p>}

      <div className="mt-6 grid gap-5 lg:grid-cols-3">
        {/* charts */}
        <div className="space-y-5 lg:col-span-2">
          <div className="card p-5">
            <Sparkline values={prices} label="Price (TAO)" />
          </div>
          <div className="card p-5">
            <Sparkline values={agaps} label="aGap score" stroke="#6ee7a8" />
          </div>

          {/* decentralization */}
          <div className="card p-5">
            <div className="flex items-center justify-between">
              <h2 className="font-bold text-white">Decentralization</h2>
              {!dec && (
                <button onClick={loadDecentralization} className="btn-ghost !py-1.5 text-xs" disabled={decLoading}>
                  {decLoading ? "Querying chain…" : "Analyze on-chain"}
                </button>
              )}
            </div>
            {!dec && (
              <p className="mt-2 text-sm text-slate-500">
                Live validator stake concentration (Nakamoto coefficient) — queried directly from chain on demand.
              </p>
            )}
            {dec && (
              <div className="mt-4">
                <div className="flex flex-wrap gap-3">
                  <div className="rounded-xl border border-white/5 bg-ink-900/50 p-3">
                    <div className="text-[11px] uppercase tracking-wide text-slate-500">Nakamoto coeff.</div>
                    <div className="mt-0.5 font-mono text-xl font-bold text-white">
                      {dec.nakamoto_coefficient}
                    </div>
                  </div>
                  <div className="rounded-xl border border-white/5 bg-ink-900/50 p-3">
                    <div className="text-[11px] uppercase tracking-wide text-slate-500">Validators</div>
                    <div className="mt-0.5 font-mono text-xl font-bold text-white">{dec.validators}</div>
                  </div>
                </div>
                {dec.nakamoto_coefficient <= 1 && (
                  <div className="mt-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-300">
                    ⚠️ A single validator controls &gt;50% of stake — critical centralisation risk.
                  </div>
                )}
                <div className="mt-4 space-y-2">
                  <div className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Top validators by stake
                  </div>
                  {dec.top_validators.map((v) => (
                    <div key={v.uid} className="flex items-center gap-2 text-xs">
                      <span className="w-12 font-mono text-slate-500">uid {v.uid}</span>
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/5">
                        <div className="h-full rounded-full bg-alpha-500" style={{ width: `${Math.min(100, v.stake_pct)}%` }} />
                      </div>
                      <span className="w-12 text-right font-mono text-slate-300">{v.stake_pct.toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* stats + scores */}
        <div className="space-y-5">
          <div className="card p-5">
            <div className="text-sm font-semibold text-white">aGap breakdown</div>
            <div className="mt-3 space-y-3">
              <ScoreBar label="Development" value={s.scores.development} />
              <ScoreBar label="Market Gap" value={s.scores.market_gap} />
              <ScoreBar label="Awareness (hidden)" value={s.scores.awareness} hint="Requires a social data source" />
              <ScoreBar label="Smart Money" value={s.scores.smart_money} />
            </div>
          </div>

          <div className="card grid grid-cols-2 gap-px overflow-hidden bg-white/5">
            {[
              ["Market cap", `${Math.round(s.market_cap_tao).toLocaleString()} τ`],
              ["Liquidity", `${Math.round(s.net_tao_flow >= 0 ? s.market_cap_tao : s.market_cap_tao).toLocaleString()} τ`],
              ["Volume", `${Math.round(s.volume_24h_tao).toLocaleString()} τ`],
              ["Net flow", `${s.net_tao_flow >= 0 ? "+" : ""}${s.net_tao_flow.toFixed(2)} τ`],
              ["Emission share", `${(s.emission_share * 100).toFixed(3)}%`],
              ["Validators", `${s.validators}/${s.max_validators}`],
              ["Miners", String(s.miners)],
              ["Commits 7d", String(s.commits_7d)],
              ["Releases 30d", String(s.releases_30d)],
              ["Reg. cost", `${s.registration_cost_tao.toFixed(4)} τ`],
              ["Age", `${s.age_days.toFixed(0)}d`],
              ["Tempo", String(s.tempo)],
            ].map(([k, v]) => (
              <div key={k as string} className="bg-ink-850 p-3">
                <div className="text-[11px] uppercase tracking-wide text-slate-500">{k}</div>
                <div className="mt-0.5 font-mono text-sm font-bold text-white">{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* dev timeline */}
      <div className="mt-6">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Intelligence timeline
        </h2>
        <div className="mt-3 space-y-2">
          {signals.length === 0 && (
            <p className="text-sm text-slate-600">No signals recorded for this subnet yet.</p>
          )}
          {signals.map((sig) => (
            <div key={sig.id} className="card flex items-start gap-3 p-4">
              <span className="text-lg">{kindIcon(sig.kind)}</span>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <span className="capitalize">{sig.kind}</span>
                  <span>· {timeAgo(sig.created_at)}</span>
                  <span className="ml-auto font-mono">⚡{sig.signal_strength}</span>
                </div>
                <div className="mt-0.5 text-sm font-semibold text-white">{sig.title}</div>
                {sig.alpha_take && <div className="mt-1 text-xs text-slate-400">{sig.alpha_take}</div>}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
