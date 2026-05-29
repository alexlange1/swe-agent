"use client";

import { useEffect, useMemo, useState } from "react";
import { api, type Subnet } from "@/lib/api";
import { AgapBadge, Pct, ScoreBar } from "./ui";

const SORTS = [
  ["agap", "aGap"],
  ["dev", "Dev"],
  ["emission_change", "Emission Δ"],
  ["price_change", "24h %"],
  ["heat", "Heat"],
  ["market_cap", "Mkt Cap"],
];

export function Leaderboard({ onSelect }: { onSelect: (n: number) => void }) {
  const [rows, setRows] = useState<Subnet[]>([]);
  const [sort, setSort] = useState("agap");
  const [whalesOnly, setWhalesOnly] = useState(false);
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .subnets({ sort, order: "desc", limit: "200", whales_only: String(whalesOnly) })
      .then(setRows)
      .catch(() => setRows([]))
      .finally(() => setLoading(false));
  }, [sort, whalesOnly]);

  const filtered = useMemo(() => {
    if (!q.trim()) return rows;
    const t = q.toLowerCase();
    return rows.filter(
      (r) => r.name.toLowerCase().includes(t) || String(r.netuid).includes(t)
    );
  }, [rows, q]);

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search subnet or netuid…"
          className="w-56 rounded-xl border border-white/10 bg-ink-900 px-4 py-2 text-sm text-white outline-none placeholder:text-slate-600 focus:border-alpha-500/50"
        />
        <div className="flex flex-wrap gap-1.5">
          {SORTS.map(([key, label]) => (
            <button
              key={key}
              onClick={() => setSort(key)}
              className={
                "rounded-lg px-3 py-1.5 text-xs font-semibold transition " +
                (sort === key
                  ? "bg-alpha-500 text-ink-950"
                  : "border border-white/10 bg-white/5 text-slate-300 hover:bg-white/10")
              }
            >
              {label}
            </button>
          ))}
        </div>
        <button
          onClick={() => setWhalesOnly((v) => !v)}
          className={
            "ml-auto rounded-lg px-3 py-1.5 text-xs font-semibold transition " +
            (whalesOnly
              ? "bg-alpha-500 text-ink-950"
              : "border border-white/10 bg-white/5 text-slate-300 hover:bg-white/10")
          }
        >
          🐋 Whales only
        </button>
      </div>

      <div className="card overflow-hidden">
        <div className="hidden grid-cols-12 gap-2 border-b border-white/5 px-4 py-3 text-[11px] font-semibold uppercase tracking-wide text-slate-500 md:grid">
          <div className="col-span-1">#</div>
          <div className="col-span-4">Subnet</div>
          <div className="col-span-1 text-right">aGap</div>
          <div className="col-span-2 text-right">Price (TAO)</div>
          <div className="col-span-1 text-right">24h</div>
          <div className="col-span-1 text-right">Dev 7d</div>
          <div className="col-span-2 text-right">Emission Δ</div>
        </div>

        {loading && (
          <div className="space-y-2 p-4">
            {Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="h-10 animate-pulse rounded-lg bg-white/5" />
            ))}
          </div>
        )}

        {!loading &&
          filtered.map((s, i) => (
            <button
              key={s.netuid}
              onClick={() => onSelect(s.netuid)}
              className="grid w-full grid-cols-2 items-center gap-2 border-b border-white/5 px-4 py-3 text-left transition last:border-0 hover:bg-white/5 md:grid-cols-12"
            >
              <div className="hidden font-mono text-xs text-slate-500 md:col-span-1 md:block">
                {i + 1}
              </div>
              <div className="col-span-1 flex items-center gap-3 md:col-span-4">
                <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-alpha-500/10 text-sm font-bold text-alpha-400">
                  {s.symbol}
                </span>
                <div className="min-w-0">
                  <div className="truncate text-sm font-semibold text-white">
                    SN{s.netuid} · {s.name}
                    {s.is_whale_accumulating && <span className="ml-1.5">🐋</span>}
                    {s.nakamoto_coefficient <= 1 && (
                      <span className="ml-1.5" title="Centralisation risk">⚠️</span>
                    )}
                  </div>
                  <div className="text-xs text-slate-500">
                    heat {s.heat_score.toFixed(0)} · {s.mentions_24h} mentions
                  </div>
                </div>
              </div>
              <div className="col-span-1 text-right md:col-span-1">
                <AgapBadge score={s.agap_score} />
              </div>
              <div className="hidden text-right font-mono text-sm text-slate-300 md:col-span-2 md:block">
                {s.price_tao.toFixed(4)}
              </div>
              <div className="hidden text-right text-sm md:col-span-1 md:block">
                <Pct value={s.price_change_24h} />
              </div>
              <div className="hidden text-right font-mono text-sm text-slate-300 md:col-span-1 md:block">
                {s.commits_7d}
              </div>
              <div className="hidden text-right text-sm md:col-span-2 md:block">
                <Pct value={s.emission_change} />
              </div>
            </button>
          ))}

        {!loading && filtered.length === 0 && (
          <div className="p-10 text-center text-sm text-slate-500">No subnets match.</div>
        )}
      </div>
    </div>
  );
}
