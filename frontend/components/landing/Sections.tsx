import { OracleDemo } from "./OracleDemo";

function SectionHeading({
  eyebrow,
  title,
  subtitle,
}: {
  eyebrow?: string;
  title: React.ReactNode;
  subtitle?: string;
}) {
  return (
    <div className="mx-auto max-w-2xl text-center">
      {eyebrow && (
        <div className="text-sm font-semibold uppercase tracking-widest text-alpha-400">
          {eyebrow}
        </div>
      )}
      <h2 className="mt-3 text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
        {title}
      </h2>
      {subtitle && <p className="mt-4 text-lg text-slate-400">{subtitle}</p>}
    </div>
  );
}

const TRACKED = [
  ["📡", "Development Updates", "Every commit, PR & release"],
  ["⚡", "Emission Shifts", "Network value signals"],
  ["⛏️", "Miner Activity", "Registration & growth"],
  ["𝕏", "Social Velocity", "Tweets, threads & hype"],
  ["📈", "Unusual Volume", "The start of big moves"],
  ["💬", "Discord Buzz", "Server activity & alerts"],
  ["🐋", "Whale Watching", "Large wallet accumulation"],
  ["🔍", "Wallet Tracker", "Track any TAO wallet"],
];

export function WhatWeTrack() {
  return (
    <section className="border-t border-white/5 py-20">
      <div className="container-x">
        <SectionHeading
          eyebrow="What we track"
          title="Thousands of data points. Every subnet. Every day."
        />
        <div className="mt-12 grid grid-cols-2 gap-4 md:grid-cols-4">
          {TRACKED.map(([icon, title, sub]) => (
            <div key={title} className="card p-5 transition hover:border-alpha-500/30">
              <div className="text-2xl">{icon}</div>
              <div className="mt-3 font-semibold text-white">{title}</div>
              <div className="mt-1 text-sm text-slate-400">{sub}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

const PROBLEMS = [
  ["🔬", "Scattered across platforms", "Teams push updates to technical platforms most investors never check. Critical developments go unnoticed for days."],
  ["📉", "Markets react too late", "Token prices stay flat while teams ship major upgrades. By the time Twitter finds out, smart money has already moved."],
  ["💡", "The gap is your alpha", "Between a team shipping a breakthrough and the market pricing it in — there's a window. We find it first."],
];

export function Problem() {
  return (
    <section className="border-t border-white/5 bg-ink-900/40 py-20">
      <div className="container-x">
        <SectionHeading
          title={
            <>
              128 teams are building.
              <br />
              <span className="text-slate-500">You have no idea what they&apos;re doing.</span>
            </>
          }
          subtitle="Subnet teams ship updates constantly — but it's nearly impossible to track where and when. By the time social media catches on, the opportunity has moved."
        />
        <div className="mt-12 grid gap-5 md:grid-cols-3">
          {PROBLEMS.map(([icon, title, body]) => (
            <div key={title} className="card p-6">
              <div className="text-3xl">{icon}</div>
              <h3 className="mt-4 text-lg font-bold text-white">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">{body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

const STEPS = [
  ["01", "Scan everything", "We continuously monitor thousands of data points across the entire ecosystem — dev activity, on-chain metrics, social sentiment, and market data for all subnets.", ["Development", "On-chain", "Social", "Market"]],
  ["02", "Analyze with AI", "Our AI engine digests complex technical updates and translates them into plain English. We tell you exactly what a subnet is building and why it matters.", ["AI Analysis", "Plain English", "Actionable"]],
  ["03", "Find the gap", "We cross-reference development quality against market awareness using proprietary scoring. When a subnet builds hard but the market hasn't noticed — that's the alpha gap.", ["Proprietary Scoring", "Gap Detection", "Multi-Signal"]],
  ["04", "Deliver actionable alpha", "Every signal comes with a full intelligence breakdown and our take. Plus daily deep-dive reports on the top alpha gap subnets.", ["Signal Feed", "Daily Reports", "Leaderboard"]],
];

export function HowItWorks() {
  return (
    <section id="how" className="border-t border-white/5 py-20">
      <div className="container-x">
        <SectionHeading eyebrow="How it works" title="An AI brain that never sleeps" />
        <div className="mt-12 grid gap-5 md:grid-cols-2">
          {STEPS.map(([num, title, body, tags]) => (
            <div key={num as string} className="card p-6">
              <div className="flex items-baseline gap-3">
                <span className="font-mono text-3xl font-black text-alpha-500/40">{num}</span>
                <h3 className="text-lg font-bold text-white">{title}</h3>
              </div>
              <p className="mt-3 text-sm leading-relaxed text-slate-400">{body}</p>
              <div className="mt-4 flex flex-wrap gap-2">
                {(tags as string[]).map((t) => (
                  <span key={t} className="chip">{t}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

const FACTORS = [
  ["⚡", "Development", "How actively is the team shipping?"],
  ["📉", "Market Gap", "Has the price caught up yet?"],
  ["👁", "Awareness", "Does the market know about this?"],
  ["🐋", "Smart Money", "Are insiders accumulating?"],
];

export function AgapScore() {
  return (
    <section id="score" className="border-t border-white/5 bg-ink-900/40 py-20">
      <div className="container-x">
        <SectionHeading
          eyebrow="The aGap Score"
          title="One score. One question."
          subtitle="Our proprietary composite score that answers: is this subnet undervalued by the market?"
        />
        <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {FACTORS.map(([icon, title, sub]) => (
            <div key={title} className="card p-6 text-center">
              <div className="text-3xl">{icon}</div>
              <div className="mt-3 font-bold text-white">{title}</div>
              <div className="mt-1 text-sm text-slate-400">{sub}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

const FEATURES = [
  ["📊", "Alpha Leaderboard", "All subnets ranked by our proprietary aGap score. Dev activity, price momentum, emission value, social buzz, and whale movements at a glance."],
  ["🧠", "AI Intelligence Feed", "Every GitHub push & HuggingFace deployment — analyzed by AI into 4 sections: What they built, Why it matters, In simple terms, and The Alpha take."],
  ["🐋", "Whale Detection", "We analyze buy/sell transaction sizes to detect when large wallets accumulate before the crowd. A 🐋 flags subnets where whale buys dwarf retail sells."],
  ["📡", "Emission Analysis", "Our eVal metric detects when the network allocates more value to a subnet than the market realizes — insiders confident before retail catches on."],
  ["🔥", "Early Trend Detection", "We monitor social campaigns, influencer activity, and launches. Get flagged when buzz is about to spike — before the crowd piles in."],
  ["🔍", "Wallet Tracker", "Track any TAO wallet across the network. See who is staking where, monitor top wallets by 24h movement, and reveal any address's full portfolio."],
];

export function Features() {
  return (
    <section id="features" className="border-t border-white/5 py-20">
      <div className="container-x">
        <SectionHeading eyebrow="Features" title="Everything you need to find alpha" />
        <div className="mt-12 grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map(([icon, title, body]) => (
            <div key={title} className="card group p-6 transition hover:border-alpha-500/30 hover:shadow-glow">
              <div className="text-3xl transition group-hover:scale-110">{icon}</div>
              <h3 className="mt-4 text-lg font-bold text-white">{title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">{body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export function Oracle() {
  return (
    <section className="border-t border-white/5 bg-ink-900/40 py-20">
      <div className="container-x grid gap-10 lg:grid-cols-2 lg:items-center">
        <div>
          <span className="chip">New — Premium Feature</span>
          <h2 className="mt-5 text-3xl font-extrabold tracking-tight text-white sm:text-4xl">
            Ask the <span className="gradient-text">TAO Oracle</span> anything
          </h2>
          <p className="mt-4 text-lg text-slate-400">
            Live AI chat using data from every Bittensor subnet — scores, signals, whale
            activity, dev momentum, and more. Ask anything, get instant answers in plain English.
          </p>
          <ul className="mt-6 space-y-2 text-sm text-slate-300">
            {[
              "Pulls live data from every subnet scan",
              "Scores, signals, whales, dev activity",
              "Grounded answers — no hallucinated picks",
            ].map((t) => (
              <li key={t} className="flex items-center gap-2">
                <span className="text-alpha-400">✓</span> {t}
              </li>
            ))}
          </ul>
        </div>
        <OracleDemo />
      </div>
    </section>
  );
}

const ALERTS = [
  ["📊", "aGap Score Change", "Catches momentum shifts the moment they happen"],
  ["⚡", "Emissions Change", "Be first when validators rotate weight to a subnet"],
  ["🔮", "Development Updates", "GitHub spikes & HF releases — filtered by signal strength"],
  ["🐋", "Whale Activity", "Large wallet moves & unusual volume from on-chain flow"],
  ["💬", "Discord Alpha", "High-signal posts across all subnet servers"],
  ["𝕏", "Going Viral on X", "KOL posts catching fire — before the crowd piles in"],
  ["💰", "Price Movement", "Your threshold, your subnets — throttled to avoid spam"],
];

export function Alerts() {
  return (
    <section className="border-t border-white/5 py-20">
      <div className="container-x">
        <SectionHeading
          eyebrow="Premium"
          title={
            <>
              Don&apos;t watch the screen.
              <br />
              <span className="text-slate-500">Let the screen watch for you.</span>
            </>
          }
          subtitle="Alpha Premium connects directly to your Telegram. The moment something worth acting on happens, you get a ping. No dashboards. No FOMO."
        />
        <div className="mt-12 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {ALERTS.map(([icon, title, sub]) => (
            <div key={title} className="card p-5">
              <div className="text-2xl">{icon}</div>
              <div className="mt-3 font-semibold text-white">{title}</div>
              <div className="mt-1 text-xs text-slate-400">{sub}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

const TESTIMONIALS = [
  ["It paid for itself in 3–4 days with modest TAO trading subnets off signals. The AI is INSANELY fast. Highly recommend.", "Verified subscriber"],
  ["I made my $49 back day one. It's perfectly named because the alpha in this is crazy and could never be tracked by a single person.", "Verified subscriber"],
  ["We got the signal days ago. That one play could pay your monthly subscription — it's damn near doubled since I got in.", "Verified subscriber"],
  ["Followed your call. Up 16 $TAO and a believer.", "Verified subscriber"],
  ["+20% since I bought the Pro version.", "Verified subscriber"],
  ["10% total portfolio up in 2 hours. Alpha is a printer.", "Verified subscriber"],
];

export function Testimonials() {
  return (
    <section className="border-t border-white/5 bg-ink-900/40 py-20">
      <div className="container-x">
        <SectionHeading
          eyebrow="Testimonials"
          title="What people are saying"
          subtitle="Real words from real subscribers."
        />
        <div className="mt-12 grid gap-5 md:grid-cols-2 lg:grid-cols-3">
          {TESTIMONIALS.map(([quote, who], i) => (
            <figure key={i} className="card p-6">
              <div className="text-3xl leading-none text-alpha-500/40">&ldquo;</div>
              <blockquote className="mt-2 text-sm leading-relaxed text-slate-300">{quote}</blockquote>
              <figcaption className="mt-4 text-xs font-medium text-alpha-400">{who}</figcaption>
            </figure>
          ))}
        </div>
      </div>
    </section>
  );
}

export function FinalCta() {
  return (
    <section className="border-t border-white/5 py-24">
      <div className="container-x text-center">
        <h2 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl">
          Stop guessing. <span className="gradient-text">Start finding alpha.</span>
        </h2>
        <p className="mx-auto mt-5 max-w-xl text-lg text-slate-400">
          Join the traders who see what the market doesn&apos;t. Free to start. Pro from
          $29/mo. Cancel anytime.
        </p>
        <div className="mt-8 flex justify-center gap-3">
          <a href="/subscribe" className="btn-primary">Start for Free →</a>
          <a href="/#pricing" className="btn-ghost">Explore Premium</a>
        </div>
      </div>
    </section>
  );
}
