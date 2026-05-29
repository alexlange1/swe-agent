"use client";

import { useEffect, useState } from "react";
import { api, type CapitalFlows } from "@/lib/api";
import { Flow, Pct } from "./ui";

export function Whales() {
  const [data, setData] = useState<CapitalFlows | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.whales().then(setData).catch(() => setData(null)).finally(() => setLoading(false));
  }, []);

  return (
    <div>
      <div className="mb-4 card p-5">
        <div className="flex items-center gap-3">
          <span className="text-3xl">🐋</span>
          <div>
            <h2 className="font-bold text-white">Capital Flow Detection</h2>
            <p className="text-sm text-slate-400">
              {data?.note ||
                "Real on-chain net capital flow into each subnet pool. Positive flow means TAO is accumulating in the subnet."}
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
          data?.subnets.map((s) => (
            <div key={s.netuid} className="flex items-center gap-3 px-4 py-3">
              <span className="grid h-8 w-8 place-items-center rounded-lg bg-alpha-500/10 text-sm font-bold text-alpha-400">
                {s.symbol}
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-semibold text-white">
                  SN{s.netuid} · {s.name}
                </div>
                <div className="text-xs text-slate-500">
                  liquidity context · price {s.price_tao.toFixed(4)} τ{" "}
                  <Pct value={s.price_change_24h} />
                </div>
              </div>
              <div className="text-right">
                <div className="font-mono text-sm font-bold">
                  <Flow value={s.net_tao_flow} />
                </div>
                <div className="text-[11px] text-slate-500">net pool flow</div>
              </div>
            </div>
          ))}
      </div>

      {!loading && (!data || data.subnets.length === 0) && (
        <div className="card p-10 text-center text-sm text-slate-500">
          No positive net inflow detected in the latest scan.
        </div>
      )}
    </div>
  );
}
