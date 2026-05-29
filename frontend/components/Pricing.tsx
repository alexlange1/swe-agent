import Link from "next/link";

interface Tier {
  name: string;
  price: string;
  cadence: string;
  tagline: string;
  features: string[];
  cta: string;
  highlight?: boolean;
}

const TIERS: Tier[] = [
  {
    name: "Free",
    price: "$0",
    cadence: "forever",
    tagline: "A taste of the alpha.",
    features: [
      "Top 10 aGap leaderboard",
      "Daily delayed signals",
      "Public subnet pages",
      "Limited AI feed preview",
    ],
    cta: "Start for Free",
  },
  {
    name: "Pro",
    price: "$29",
    cadence: "/mo",
    tagline: "For the active subnet trader.",
    features: [
      "Full 128-subnet leaderboard",
      "Real-time AI intelligence feed",
      "Whale detection & emission analysis",
      "Wallet tracker",
      "Sortable signals by every metric",
    ],
    cta: "Go Pro",
    highlight: true,
  },
  {
    name: "Premium",
    price: "$49",
    cadence: "/mo",
    tagline: "The full edge.",
    features: [
      "Everything in Pro",
      "TAO Oracle — 15 AI queries/day",
      "Telegram alerts (7 alert types)",
      "Daily deep-dive reports",
      "Early trend detection",
    ],
    cta: "Go Premium",
  },
  {
    name: "Ultra",
    price: "$99",
    cadence: "/mo",
    tagline: "Maximum throughput.",
    features: [
      "Everything in Premium",
      "TAO Oracle — 50 AI queries/day",
      "Priority scan refresh",
      "API access",
      "Custom alert thresholds",
    ],
    cta: "Go Ultra",
  },
];

export function Pricing() {
  return (
    <section id="pricing" className="border-t border-white/5 bg-ink-900/40 py-20">
      <div className="container-x">
        <div className="mx-auto max-w-2xl text-center">
          <div className="text-sm font-semibold uppercase tracking-widest text-alpha-400">
            Pricing
          </div>
          <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            Pick your edge
          </h2>
          <p className="mt-4 text-lg text-slate-400">
            Free to start. Upgrade when the alpha pays for itself. Cancel anytime.
          </p>
        </div>
        <div className="mt-12 grid gap-5 lg:grid-cols-4">
          {TIERS.map((t) => (
            <div
              key={t.name}
              className={
                "card flex flex-col p-6 " +
                (t.highlight ? "border-alpha-500/50 shadow-glow" : "")
              }
            >
              {t.highlight && (
                <span className="mb-3 inline-flex w-fit rounded-full bg-alpha-500 px-3 py-1 text-xs font-bold text-ink-950">
                  Most popular
                </span>
              )}
              <div className="text-lg font-bold text-white">{t.name}</div>
              <div className="mt-2 flex items-baseline gap-1">
                <span className="text-4xl font-extrabold text-white">{t.price}</span>
                <span className="text-sm text-slate-500">{t.cadence}</span>
              </div>
              <p className="mt-2 text-sm text-slate-400">{t.tagline}</p>
              <ul className="mt-5 flex-1 space-y-2 text-sm text-slate-300">
                {t.features.map((f) => (
                  <li key={f} className="flex items-start gap-2">
                    <span className="mt-0.5 text-alpha-400">✓</span> {f}
                  </li>
                ))}
              </ul>
              <Link
                href={`/subscribe?plan=${t.name.toLowerCase()}`}
                className={"mt-6 " + (t.highlight ? "btn-primary" : "btn-ghost")}
              >
                {t.cta}
              </Link>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
