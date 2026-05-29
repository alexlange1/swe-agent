"use client";

import { useState } from "react";
import { api } from "@/lib/api";

const SUGGESTIONS = [
  "Which subnets are whales accumulating right now?",
  "Which subnets shipped the most code this week?",
  "What are the biggest centralisation risks?",
  "Best alpha gaps right now — top 3",
];

interface Msg {
  role: "user" | "oracle";
  text: string;
}

export function OracleDemo() {
  const [messages, setMessages] = useState<Msg[]>([
    {
      role: "oracle",
      text: "Ask me anything about the Bittensor ecosystem — scores, signals, whale activity, dev momentum. I answer from live scan data.",
    },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  async function ask(q: string) {
    if (!q.trim() || busy) return;
    setMessages((m) => [...m, { role: "user", text: q }]);
    setInput("");
    setBusy(true);
    try {
      const res = await api.oracle(q);
      setMessages((m) => [...m, { role: "oracle", text: res.answer }]);
    } catch {
      setMessages((m) => [
        ...m,
        { role: "oracle", text: "The Oracle is offline. Start the backend to enable live answers." },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card p-1.5">
      <div className="flex items-center gap-2 px-4 py-3">
        <span className="grid h-8 w-8 place-items-center rounded-lg bg-alpha-500/15 text-alpha-400">
          🔮
        </span>
        <div className="flex-1">
          <div className="text-sm font-semibold text-white">TAO Oracle</div>
          <div className="text-[11px] text-alpha-400">Live · grounded on scan data</div>
        </div>
      </div>
      <div className="max-h-72 space-y-3 overflow-y-auto rounded-xl bg-ink-900/60 p-4">
        {messages.map((m, i) => (
          <div
            key={i}
            className={m.role === "user" ? "flex justify-end" : "flex justify-start"}
          >
            <div
              className={
                "max-w-[85%] whitespace-pre-wrap rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed " +
                (m.role === "user"
                  ? "bg-alpha-500 text-ink-950"
                  : "border border-white/10 bg-ink-800 text-slate-200")
              }
            >
              {m.text}
            </div>
          </div>
        ))}
        {busy && (
          <div className="flex justify-start">
            <div className="rounded-2xl border border-white/10 bg-ink-800 px-3.5 py-2.5 text-sm text-slate-400">
              <span className="animate-pulse">Consulting the Oracle…</span>
            </div>
          </div>
        )}
      </div>
      <div className="flex flex-wrap gap-1.5 px-2 py-3">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => ask(s)}
            className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300 transition hover:border-alpha-500/50 hover:text-white"
          >
            {s}
          </button>
        ))}
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
        className="flex gap-2 p-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask the Oracle…"
          className="flex-1 rounded-xl border border-white/10 bg-ink-900 px-4 py-2.5 text-sm text-white outline-none placeholder:text-slate-600 focus:border-alpha-500/50"
        />
        <button type="submit" disabled={busy} className="btn-primary !py-2.5 disabled:opacity-50">
          Ask
        </button>
      </form>
    </div>
  );
}
