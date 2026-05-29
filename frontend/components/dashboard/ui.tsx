// Small shared UI helpers for the dashboard.

export function ScoreBar({
  value,
  label,
  hint,
}: {
  value: number | null;
  label: string;
  hint?: string;
}) {
  if (value === null || value === undefined) {
    return (
      <div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400">{label}</span>
          <span className="font-mono text-slate-600" title={hint}>n/a</span>
        </div>
        <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-white/5">
          <div className="h-full w-full bg-[repeating-linear-gradient(45deg,rgba(255,255,255,.06),rgba(255,255,255,.06)_4px,transparent_4px,transparent_8px)]" />
        </div>
      </div>
    );
  }
  const hue = value >= 70 ? "bg-alpha-500" : value >= 45 ? "bg-amber-400" : "bg-slate-500";
  return (
    <div>
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-slate-300">{value.toFixed(0)}</span>
      </div>
      <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-white/5">
        <div className={`h-full rounded-full ${hue}`} style={{ width: `${Math.min(100, value)}%` }} />
      </div>
    </div>
  );
}

export function Flow({ value }: { value: number }) {
  const pos = value >= 0;
  return (
    <span className={pos ? "text-alpha-400" : "text-rose-400"}>
      {pos ? "+" : ""}
      {value.toFixed(2)} τ
    </span>
  );
}

export function AgapBadge({ score }: { score: number }) {
  const tone =
    score >= 70
      ? "text-alpha-400 bg-alpha-500/10 ring-alpha-500/30"
      : score >= 45
      ? "text-amber-300 bg-amber-500/10 ring-amber-500/30"
      : "text-slate-400 bg-white/5 ring-white/10";
  return (
    <span className={`inline-flex items-center rounded-lg px-2.5 py-1 font-mono text-sm font-bold ring-1 ${tone}`}>
      {score.toFixed(0)}
    </span>
  );
}

export function Pct({ value }: { value: number }) {
  const up = value >= 0;
  return (
    <span className={up ? "text-alpha-400" : "text-rose-400"}>
      {up ? "▲" : "▼"} {Math.abs(value).toFixed(1)}%
    </span>
  );
}

export function kindIcon(kind: string): string {
  return (
    {
      development: "🔮",
      whale: "🐋",
      emission: "⚡",
      social: "𝕏",
      discord: "💬",
      price: "💰",
    } as Record<string, string>
  )[kind] || "📡";
}

export function timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const m = Math.floor(diff / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}
