"use client";

// Dependency-free SVG line chart. Renders a smooth-ish polyline with an area fill.

export function Sparkline({
  values,
  width = 560,
  height = 160,
  stroke = "#34e89e",
  fill = true,
  label,
}: {
  values: number[];
  width?: number;
  height?: number;
  stroke?: string;
  fill?: boolean;
  label?: string;
}) {
  if (!values || values.length < 2) {
    return (
      <div
        className="grid place-items-center rounded-xl border border-white/5 bg-ink-900/50 text-xs text-slate-600"
        style={{ height }}
      >
        Not enough history yet — charts fill in as scans accrue.
      </div>
    );
  }

  const pad = 8;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const stepX = (width - pad * 2) / (values.length - 1);

  const pts = values.map((v, i) => {
    const x = pad + i * stepX;
    const y = pad + (height - pad * 2) * (1 - (v - min) / span);
    return [x, y] as const;
  });

  const line = pts.map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area =
    `${line} L${pts[pts.length - 1][0].toFixed(1)},${height - pad} ` +
    `L${pts[0][0].toFixed(1)},${height - pad} Z`;
  const gid = `g-${Math.round(min)}-${Math.round(max)}-${values.length}`;

  return (
    <div>
      {label && <div className="mb-1 text-xs text-slate-500">{label}</div>}
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" preserveAspectRatio="none">
        <defs>
          <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={stroke} stopOpacity="0.28" />
            <stop offset="100%" stopColor={stroke} stopOpacity="0" />
          </linearGradient>
        </defs>
        {fill && <path d={area} fill={`url(#${gid})`} />}
        <path d={line} fill="none" stroke={stroke} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r="3" fill={stroke} />
      </svg>
    </div>
  );
}
