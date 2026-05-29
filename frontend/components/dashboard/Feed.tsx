"use client";

import { useEffect, useState } from "react";
import { api, type Signal } from "@/lib/api";
import { kindIcon, timeAgo } from "./ui";

const FILTERS = [
  ["", "All"],
  ["development", "Dev"],
  ["whale", "Whales"],
  ["emission", "Emission"],
  ["social", "Social"],
];

export function Feed() {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [kind, setKind] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    const params: Record<string, string> = { limit: "60" };
    if (kind) params.kind = kind;
    api.signals(params).then(setSignals).catch(() => setSignals([])).finally(() => setLoading(false));
  }, [kind]);

  return (
    <div>
      <div className="mb-4 flex flex-wrap gap-1.5">
        {FILTERS.map(([key, label]) => (
          <button
            key={label}
            onClick={() => setKind(key)}
            className={
              "rounded-lg px-3 py-1.5 text-xs font-semibold transition " +
              (kind === key
                ? "bg-alpha-500 text-ink-950"
                : "border border-white/10 bg-white/5 text-slate-300 hover:bg-white/10")
            }
          >
            {label}
          </button>
        ))}
      </div>

      {loading && (
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-32 animate-pulse rounded-2xl bg-white/5" />
          ))}
        </div>
      )}

      <div className="space-y-3">
        {!loading &&
          signals.map((s) => (
            <article key={s.id} className="card p-5">
              <div className="flex items-start gap-3">
                <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-alpha-500/10 text-lg">
                  {kindIcon(s.kind)}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold uppercase tracking-wide text-alpha-400">
                      SN{s.netuid} · {s.subnet_name}
                    </span>
                    <span className="text-xs text-slate-600">{timeAgo(s.created_at)}</span>
                    <span className="ml-auto inline-flex items-center gap-1 rounded-full bg-white/5 px-2 py-0.5 text-[11px] font-mono text-slate-300">
                      ⚡ {s.signal_strength}
                    </span>
                  </div>
                  <h3 className="mt-1 font-bold text-white">{s.title}</h3>

                  <div className="mt-3 grid gap-3 sm:grid-cols-2">
                    {[
                      ["What they built", s.what_built],
                      ["Why it matters", s.why_matters],
                      ["In simple terms", s.simple_terms],
                      ["The Alpha take", s.alpha_take],
                    ]
                      .filter(([, v]) => v)
                      .map(([label, v]) => (
                        <div key={label} className="rounded-xl border border-white/5 bg-ink-900/50 p-3">
                          <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                            {label}
                          </div>
                          <p className="mt-1 text-sm leading-relaxed text-slate-300">{v}</p>
                        </div>
                      ))}
                  </div>

                  {s.source_url && (
                    <a
                      href={s.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-3 inline-block text-xs font-semibold text-alpha-400 hover:underline"
                    >
                      View source →
                    </a>
                  )}
                </div>
              </div>
            </article>
          ))}
      </div>

      {!loading && signals.length === 0 && (
        <div className="card p-10 text-center text-sm text-slate-500">No signals yet.</div>
      )}
    </div>
  );
}
