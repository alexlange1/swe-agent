"use client";

import { useEffect, useState } from "react";
import { api, type Whale } from "@/lib/api";
import { timeAgo } from "./ui";

const LABEL_TONE: Record<string, string> = {
  validator: "text-sky-300 bg-sky-500/10",
  founder: "text-purple-300 bg-purple-500/10",
  fund: "text-amber-300 bg-amber-500/10",
  exchange: "text-rose-300 bg-rose-500/10",
  unknown: "text-slate-400 bg-white/5",
};

export function Whales() {
  const [rows, setRows] = useState<Whale[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.whales().then(setRows).catch(() => setRows([])).finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <div className="mb-4 card p-5">
        <div className="flex items-center gap-3">
          <span className="text-3xl">🐋</span>
          <div>
            <h2 className="font-bold text-white">Whale Detection</h2>
            <p className="text-sm text-slate-400">
              Large-wallet accumulation detected from on-chain buy/sell flow. A high
              buy/sell ratio means big wallets are buying while retail is flat.
            </p>
          </div>
        </div>
      </div>

      {loading && (
        <div className="space-y-2">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-14 animate-pulse rounded-xl bg-white/5" />
          ))}
        </div>
      )}

      <div className="card divide-y divide-white/5">
        {!loading &&
          rows.map((w, i) => (
            <div key={i} className="flex items-center gap-3 px-4 py-3">
              <span className="text-lg">🐋</span>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-white">
                  SN{w.netuid} · {w.subnet_name}
                </div>
                <div className="truncate font-mono text-xs text-slate-500">{w.wallet}</div>
              </div>
              <span
                className={`hidden rounded-full px-2 py-0.5 text-[11px] font-medium capitalize sm:inline ${
                  LABEL_TONE[w.wallet_label] || LABEL_TONE.unknown
                }`}
              >
                {w.wallet_label}
              </span>
              <div className="text-right">
                <div className="font-mono text-sm font-bold text-alpha-400">
                  {w.amount_tao.toFixed(0)} τ
                </div>
                <div className="text-[11px] text-slate-500">{w.buy_sell_ratio.toFixed(1)}x ratio</div>
              </div>
            </div>
          ))}
      </div>

      {!loading && rows.length === 0 && (
        <div className="card p-10 text-center text-sm text-slate-500">
          No whale accumulation detected in the latest scan.
        </div>
      )}
    </div>
  );
}
