"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, type Health } from "@/lib/api";
import { Leaderboard } from "@/components/dashboard/Leaderboard";
import { Feed } from "@/components/dashboard/Feed";
import { Whales } from "@/components/dashboard/Whales";
import { WalletTracker } from "@/components/dashboard/WalletTracker";
import { SubnetDetail } from "@/components/dashboard/SubnetDetail";
import { OracleDemo } from "@/components/landing/OracleDemo";

const TABS = [
  ["leaderboard", "📊 Leaderboard"],
  ["feed", "🧠 AI Feed"],
  ["whales", "🐋 Whales"],
  ["wallet", "🔍 Wallet Tracker"],
  ["oracle", "🔮 TAO Oracle"],
];

function DashboardInner() {
  const params = useSearchParams();
  const initialTab = params.get("tab") || "leaderboard";
  const initialNetuid = params.get("netuid");

  const [tab, setTab] = useState(initialTab);
  const [selected, setSelected] = useState<number | null>(
    initialNetuid ? Number(initialNetuid) : null
  );
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => {});
  }, []);

  const chainLive = health?.providers?.chain;

  return (
    <div className="container-x py-10">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-white">
            Intelligence Dashboard
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Live aGap intelligence across the Bittensor ecosystem.
          </p>
        </div>
        <div className="flex items-center gap-3 text-xs text-slate-500">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-2 w-2 animate-pulseglow rounded-full bg-alpha-400" />
            {health ? `${health.subnets_tracked} subnets tracked` : "connecting…"}
          </span>
          <span className="rounded-full border border-white/10 px-2 py-0.5">
            {chainLive ? "live on-chain" : "connecting…"}
          </span>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap gap-1.5">
        {TABS.map(([key, label]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={
              "rounded-xl px-4 py-2 text-sm font-semibold transition " +
              (tab === key
                ? "bg-alpha-500 text-ink-950"
                : "border border-white/10 bg-white/5 text-slate-300 hover:bg-white/10")
            }
          >
            {label}
          </button>
        ))}
      </div>

      <div className="mt-6">
        {tab === "leaderboard" && <Leaderboard onSelect={setSelected} />}
        {tab === "feed" && <Feed />}
        {tab === "whales" && <Whales />}
        {tab === "wallet" && <WalletTracker />}
        {tab === "oracle" && (
          <div className="mx-auto max-w-2xl">
            <OracleDemo />
          </div>
        )}
      </div>

      {selected !== null && (
        <SubnetDetail netuid={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

export default function DashboardPage() {
  return (
    <Suspense fallback={<div className="container-x py-20 text-slate-500">Loading…</div>}>
      <DashboardInner />
    </Suspense>
  );
}
