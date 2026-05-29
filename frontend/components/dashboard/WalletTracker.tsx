"use client";

import { useState } from "react";
import { api, type WalletPortfolio } from "@/lib/api";
import { Pct } from "./ui";

const SAMPLE = "5HEo565WAy4Dbq3Sv271SAi7syBSofyfhhwRNjFNSM2gP9M2";

export function WalletTracker() {
  const [address, setAddress] = useState("");
  const [data, setData] = useState<WalletPortfolio | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function lookup(addr: string) {
    if (!addr.trim()) return;
    setLoading(true);
    setError("");
    setData(null);
    try {
      const res = await api.wallet(addr.trim());
      setData(res);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "";
      setError(
        msg.includes("503")
          ? "Wallet tracking requires an on-chain stake indexer. Set TAOSTATS_API_KEY on the backend to enable it — no data is fabricated."
          : "Could not resolve that address."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <div className="card p-5">
        <div className="flex items-center gap-3">
          <span className="text-3xl">🔍</span>
          <div>
            <h2 className="font-bold text-white">Wallet Tracker</h2>
            <p className="text-sm text-slate-400">
              Look up any TAO wallet to reveal its full cross-subnet portfolio.
            </p>
          </div>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            lookup(address);
          }}
          className="mt-4 flex flex-col gap-2 sm:flex-row"
        >
          <input
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="Enter a coldkey address (5…)"
            className="flex-1 rounded-xl border border-white/10 bg-ink-900 px-4 py-2.5 font-mono text-sm text-white outline-none placeholder:text-slate-600 focus:border-alpha-500/50"
          />
          <button type="submit" className="btn-primary !py-2.5" disabled={loading}>
            {loading ? "Resolving…" : "Track"}
          </button>
        </form>
        <button
          onClick={() => {
            setAddress(SAMPLE);
            lookup(SAMPLE);
          }}
          className="mt-2 text-xs text-slate-500 hover:text-alpha-400"
        >
          Try a sample wallet →
        </button>
      </div>

      {error && (
        <div className="mt-4 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-300">
          {error}
        </div>
      )}

      {data && (
        <div className="mt-4 space-y-4">
          <div className="card flex flex-wrap items-center justify-between gap-4 p-5">
            <div>
              <div className="truncate font-mono text-xs text-slate-500">{data.address}</div>
              <div className="mt-1 flex items-center gap-2">
                <span className="text-2xl font-extrabold text-white">
                  {data.total_value_tao.toFixed(2)} τ
                </span>
                <span className="rounded-full bg-white/5 px-2 py-0.5 text-[11px] capitalize text-slate-300">
                  {data.label}
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-xs text-slate-500">24h change</div>
              <div className="text-lg font-bold">
                <Pct value={(data.change_24h_tao / Math.max(1, data.total_value_tao)) * 100} />
              </div>
            </div>
          </div>

          <div className="card divide-y divide-white/5">
            {data.positions.map((p) => (
              <div key={p.netuid} className="flex items-center gap-3 px-4 py-3">
                <span className="grid h-8 w-8 place-items-center rounded-lg bg-alpha-500/10 text-sm font-bold text-alpha-400">
                  {p.symbol}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-semibold text-white">
                    SN{p.netuid} · {p.subnet_name}
                  </div>
                  <div className="text-xs text-slate-500">
                    {p.stake_alpha.toLocaleString()} α staked
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-mono text-sm text-slate-200">{p.value_tao.toFixed(2)} τ</div>
                  <div className="text-xs">
                    <Pct value={p.change_24h} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
