"use client";

import { useEffect, useState } from "react";
import { api, type Alert } from "@/lib/api";

const METRICS = [
  ["agap_score", "aGap score"],
  ["price_change_24h", "Price 24h %"],
  ["net_tao_flow", "Net TAO flow"],
  ["commits_7d", "Commits 7d"],
  ["emission_change", "Emission Δ %"],
];

export function Alerts() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [metric, setMetric] = useState("agap_score");
  const [op, setOp] = useState(">");
  const [threshold, setThreshold] = useState(60);
  const [label, setLabel] = useState("");
  const [netuid, setNetuid] = useState("");
  const [tg, setTg] = useState<boolean | null>(null);
  const [toast, setToast] = useState("");

  const load = () => api.alerts.list().then(setAlerts).catch(() => setAlerts([]));
  useEffect(() => {
    load();
    api.alerts.telegramStatus().then((s) => setTg(s.telegram_configured)).catch(() => setTg(false));
  }, []);

  async function create() {
    await api.alerts.create({
      metric, op, threshold,
      netuid: netuid ? Number(netuid) : null,
      label: label || `${metric} ${op} ${threshold}`,
    });
    setLabel("");
    load();
  }

  async function testTelegram() {
    try {
      const r = await api.alerts.test();
      setToast(r.sent ? "Test alert sent to Telegram ✓" : "Send failed");
    } catch {
      setToast("Telegram not configured — set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID");
    }
    setTimeout(() => setToast(""), 4000);
  }

  return (
    <div className="grid gap-5 lg:grid-cols-2">
      <div className="card p-5">
        <h2 className="font-bold text-white">Create an alert</h2>
        <p className="mt-1 text-sm text-slate-400">
          Rules are evaluated against every scan. Connect Telegram to get pings.
        </p>
        <div className="mt-4 space-y-3">
          <div>
            <label className="text-xs text-slate-500">Metric</label>
            <select
              value={metric}
              onChange={(e) => setMetric(e.target.value)}
              className="mt-1 w-full rounded-xl border border-white/10 bg-ink-900 px-3 py-2 text-sm text-white outline-none focus:border-alpha-500/50"
            >
              {METRICS.map(([v, l]) => (
                <option key={v} value={v}>{l}</option>
              ))}
            </select>
          </div>
          <div className="flex gap-2">
            <div className="w-24">
              <label className="text-xs text-slate-500">Condition</label>
              <select
                value={op}
                onChange={(e) => setOp(e.target.value)}
                className="mt-1 w-full rounded-xl border border-white/10 bg-ink-900 px-3 py-2 text-sm text-white outline-none focus:border-alpha-500/50"
              >
                <option value=">">above</option>
                <option value="<">below</option>
              </select>
            </div>
            <div className="flex-1">
              <label className="text-xs text-slate-500">Threshold</label>
              <input
                type="number"
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
                className="mt-1 w-full rounded-xl border border-white/10 bg-ink-900 px-3 py-2 text-sm text-white outline-none focus:border-alpha-500/50"
              />
            </div>
            <div className="w-28">
              <label className="text-xs text-slate-500">Netuid (opt)</label>
              <input
                value={netuid}
                onChange={(e) => setNetuid(e.target.value)}
                placeholder="any"
                className="mt-1 w-full rounded-xl border border-white/10 bg-ink-900 px-3 py-2 text-sm text-white outline-none focus:border-alpha-500/50"
              />
            </div>
          </div>
          <div>
            <label className="text-xs text-slate-500">Label (optional)</label>
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="e.g. Breakout watch"
              className="mt-1 w-full rounded-xl border border-white/10 bg-ink-900 px-3 py-2 text-sm text-white outline-none focus:border-alpha-500/50"
            />
          </div>
          <button onClick={create} className="btn-primary w-full">Create alert</button>
        </div>

        <div className="mt-5 rounded-xl border border-white/5 bg-ink-900/50 p-4">
          <div className="flex items-center justify-between">
            <div className="text-sm font-semibold text-white">Telegram delivery</div>
            <span
              className={
                "rounded-full px-2 py-0.5 text-[11px] " +
                (tg ? "bg-alpha-500/15 text-alpha-400" : "bg-white/5 text-slate-400")
              }
            >
              {tg === null ? "…" : tg ? "connected" : "not configured"}
            </span>
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Set <code className="text-slate-300">TELEGRAM_BOT_TOKEN</code> and{" "}
            <code className="text-slate-300">TELEGRAM_CHAT_ID</code> on the backend, then test.
          </p>
          <button onClick={testTelegram} className="btn-ghost mt-3 !py-2 text-xs">
            Send test alert
          </button>
          {toast && <div className="mt-2 text-xs text-alpha-400">{toast}</div>}
        </div>
      </div>

      <div className="card overflow-hidden">
        <div className="border-b border-white/5 px-4 py-3 text-sm font-semibold text-white">
          Active alerts ({alerts.length})
        </div>
        {alerts.length === 0 && (
          <div className="p-8 text-center text-sm text-slate-600">No alerts yet.</div>
        )}
        <div className="divide-y divide-white/5">
          {alerts.map((a) => (
            <div key={a.id} className="flex items-center gap-3 px-4 py-3">
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-semibold text-white">{a.label}</div>
                <div className="font-mono text-xs text-slate-500">
                  {a.metric} {a.op} {a.threshold}
                  {a.netuid ? ` · SN${a.netuid}` : " · any"}
                  {a.last_triggered_netuid ? ` · last: SN${a.last_triggered_netuid}` : ""}
                </div>
              </div>
              <button
                onClick={() => api.alerts.remove(a.id).then(load)}
                className="rounded-lg border border-white/10 px-2 py-1 text-xs text-slate-400 hover:border-rose-500/40 hover:text-rose-300"
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
