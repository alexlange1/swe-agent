"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type Health, type Subnet } from "@/lib/api";

export function Hero() {
  const [health, setHealth] = useState<Health | null>(null);
  const [top, setTop] = useState<Subnet[]>([]);

  useEffect(() => {
    api.health().then(setHealth).catch(() => {});
    api.subnets({ sort: "agap", limit: "5" }).then(setTop).catch(() => {});
  }, []);

  return (
    <section className="relative overflow-hidden bg-grid-fade">
      <div
        className="pointer-events-none absolute inset-0 opacity-[0.04]"
        style={{
          backgroundImage:
            "linear-gradient(rgba(255,255,255,.6) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.6) 1px, transparent 1px)",
          backgroundSize: "44px 44px",
          maskImage: "radial-gradient(circle at 50% 0%, black, transparent 70%)",
        }}
      />
      <div className="container-x relative grid gap-12 py-20 md:py-28 lg:grid-cols-2 lg:items-center">
        <div>
          <span className="chip">
            <span className="h-1.5 w-1.5 animate-pulseglow rounded-full bg-alpha-400" />
            Bittensor Subnet Intelligence
          </span>
          <h1 className="mt-5 text-4xl font-extrabold leading-[1.05] tracking-tight text-white sm:text-5xl lg:text-6xl">
            Find the <span className="gradient-text">alpha gap</span> before everyone else.
          </h1>
          <p className="mt-5 max-w-xl text-lg leading-relaxed text-slate-300">
            Our AI scans thousands of data points across the entire Bittensor ecosystem
            to surface undervalued subnets — before the market catches on.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/subscribe" className="btn-primary">
              Start for Free →
            </Link>
            <Link href="/dashboard" className="btn-ghost">
              Explore the dashboard ↓
            </Link>
          </div>
          <p className="mt-5 text-sm text-slate-500">
            Free preview · Pro from $29/mo · Premium from $49/mo
          </p>
        </div>

        {/* Live leaderboard preview */}
        <div className="card overflow-hidden p-1.5">
          <div className="flex items-center justify-between px-4 py-3">
            <div className="flex items-center gap-2 text-sm font-semibold text-white">
              <span className="h-2 w-2 animate-pulseglow rounded-full bg-alpha-400" />
              Live aGap Leaderboard
            </div>
            <span className="font-mono text-xs text-slate-500">
              {health ? `${health.subnets_tracked} subnets` : "connecting…"}
            </span>
          </div>
          <div className="divide-y divide-white/5 rounded-xl bg-ink-900/60">
            {(top.length ? top : Array.from({ length: 5 })).map((s, i) =>
              s ? (
                <Link
                  href={`/dashboard?netuid=${(s as Subnet).netuid}`}
                  key={(s as Subnet).netuid}
                  className="flex items-center gap-3 px-4 py-3 transition hover:bg-white/5"
                >
                  <span className="w-6 font-mono text-xs text-slate-500">#{i + 1}</span>
                  <span className="grid h-8 w-8 place-items-center rounded-lg bg-alpha-500/10 text-sm font-bold text-alpha-400">
                    {(s as Subnet).symbol}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-semibold text-white">
                      SN{(s as Subnet).netuid} · {(s as Subnet).name}
                    </div>
                    <div className="text-xs text-slate-500">
                      {(s as Subnet).commits_7d} commits/7d ·{" "}
                      {(s as Subnet).is_whale_accumulating ? "🐋 accumulating" : "watching"}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm font-bold text-alpha-400">
                      {(s as Subnet).agap_score.toFixed(0)}
                    </div>
                    <div className="text-[10px] uppercase tracking-wide text-slate-600">
                      aGap
                    </div>
                  </div>
                </Link>
              ) : (
                <div key={i} className="flex items-center gap-3 px-4 py-3">
                  <div className="h-8 w-full animate-pulse rounded-lg bg-white/5" />
                </div>
              )
            )}
          </div>
          <p className="px-4 py-2 text-center text-[11px] text-slate-600">
            Live data from the Alpha scan engine
          </p>
        </div>
      </div>
    </section>
  );
}
