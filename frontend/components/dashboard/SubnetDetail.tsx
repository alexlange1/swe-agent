"use client";

import { useEffect, useState } from "react";
import { api, type Signal, type Subnet } from "@/lib/api";
import { Pct, ScoreBar, timeAgo, kindIcon } from "./ui";

export function SubnetDetail({ netuid, onClose }: { netuid: number; onClose: () => void }) {
  const [subnet, setSubnet] = useState<Subnet | null>(null);
  const [signals, setSignals] = useState<Signal[]>([]);

  useEffect(() => {
    api.subnet(netuid).then(setSubnet).catch(() => {});
    api.signals({ netuid: String(netuid), limit: "10" }).then(setSignals).catch(() => {});
  }, [netuid]);

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative h-full w-full max-w-md overflow-y-auto border-l border-white/10 bg-ink-900 p-6 shadow-2xl">
        <button
          onClick={onClose}
          className="absolute right-4 top-4 grid h-8 w-8 place-items-center rounded-lg border border-white/10 text-slate-400 hover:bg-white/5"
          aria-label="Close"
        >
          ✕
        </button>

        {!subnet ? (
          <div className="mt-10 h-40 animate-pulse rounded-2xl bg-white/5" />
        ) : (
          <>
            <div className="flex items-center gap-3">
              <span className="grid h-12 w-12 place-items-center rounded-xl bg-alpha-500/10 text-xl font-bold text-alpha-400">
                {subnet.symbol}
              </span>
              <div>
                <h2 className="text-xl font-extrabold text-white">
                  SN{subnet.netuid} · {subnet.name}
                </h2>
                <div className="text-sm text-slate-400">
                  {subnet.price_tao.toFixed(4)} τ <Pct value={subnet.price_change_24h} />
                </div>
              </div>
            </div>

            <div className="mt-5 rounded-2xl border border-alpha-500/20 bg-alpha-500/5 p-4">
              <div className="flex items-baseline justify-between">
                <span className="text-sm font-semibold text-slate-300">aGap Score</span>
                <span className="font-mono text-3xl font-black text-alpha-400">
                  {subnet.agap_score.toFixed(0)}
                </span>
              </div>
              <div className="mt-4 space-y-3">
                <ScoreBar label="Development" value={subnet.scores.development} />
                <ScoreBar label="Market Gap" value={subnet.scores.market_gap} />
                <ScoreBar label="Awareness (hidden)" value={subnet.scores.awareness} />
                <ScoreBar label="Smart Money" value={subnet.scores.smart_money} />
              </div>
            </div>

            <div className="mt-5 grid grid-cols-2 gap-3">
              {[
                ["Commits 7d", subnet.commits_7d],
                ["Contributors", subnet.contributors_7d],
                ["Emission Δ", `${subnet.emission_change >= 0 ? "+" : ""}${subnet.emission_change.toFixed(0)}%`],
                ["Buy/Sell", `${subnet.buy_sell_ratio.toFixed(1)}x`],
                ["Validators", subnet.validators],
                ["Nakamoto", subnet.nakamoto_coefficient],
                ["Mentions 24h", subnet.mentions_24h],
                ["Heat", subnet.heat_score.toFixed(0)],
              ].map(([label, val]) => (
                <div key={label as string} className="rounded-xl border border-white/5 bg-ink-850 p-3">
                  <div className="text-[11px] uppercase tracking-wide text-slate-500">{label}</div>
                  <div className="mt-0.5 font-mono text-lg font-bold text-white">{val}</div>
                </div>
              ))}
            </div>

            {subnet.nakamoto_coefficient <= 1 && (
              <div className="mt-4 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-300">
                ⚠️ Nakamoto coefficient of {subnet.nakamoto_coefficient} — a single
                validator may control consensus. Centralisation risk.
              </div>
            )}

            <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-slate-500">
              Recent signals
            </h3>
            <div className="mt-3 space-y-2">
              {signals.length === 0 && (
                <p className="text-sm text-slate-600">No recent signals for this subnet.</p>
              )}
              {signals.map((s) => (
                <div key={s.id} className="rounded-xl border border-white/5 bg-ink-850 p-3">
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <span>{kindIcon(s.kind)}</span>
                    <span className="capitalize">{s.kind}</span>
                    <span>· {timeAgo(s.created_at)}</span>
                    <span className="ml-auto font-mono">⚡{s.signal_strength}</span>
                  </div>
                  <div className="mt-1 text-sm font-semibold text-white">{s.title}</div>
                  {s.alpha_take && (
                    <div className="mt-1 text-xs text-slate-400">{s.alpha_take}</div>
                  )}
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
