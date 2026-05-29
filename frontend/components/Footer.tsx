import Link from "next/link";
import { Logo } from "./Logo";

export function Footer() {
  return (
    <footer className="border-t border-white/5 bg-ink-950">
      <div className="container-x grid gap-10 py-14 md:grid-cols-4">
        <div className="md:col-span-2">
          <Logo />
          <p className="mt-4 max-w-sm text-sm leading-relaxed text-slate-400">
            Bittensor subnet intelligence. We scan thousands of data points across every
            subnet, every day — and surface the alpha gap before the market catches on.
          </p>
          <p className="mt-4 text-xs text-slate-600">
            Not financial advice. Crypto assets are volatile; do your own research.
          </p>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-white">Product</h4>
          <ul className="mt-4 space-y-2 text-sm text-slate-400">
            <li><Link href="/dashboard" className="hover:text-white">Dashboard</Link></li>
            <li><Link href="/#features" className="hover:text-white">Features</Link></li>
            <li><Link href="/#score" className="hover:text-white">aGap Score</Link></li>
            <li><Link href="/subscribe" className="hover:text-white">Pricing</Link></li>
          </ul>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-white">Intelligence</h4>
          <ul className="mt-4 space-y-2 text-sm text-slate-400">
            <li><Link href="/dashboard?tab=feed" className="hover:text-white">AI Feed</Link></li>
            <li><Link href="/dashboard?tab=whales" className="hover:text-white">Whale Detection</Link></li>
            <li><Link href="/dashboard?tab=wallet" className="hover:text-white">Wallet Tracker</Link></li>
            <li><Link href="/dashboard?tab=oracle" className="hover:text-white">TAO Oracle</Link></li>
          </ul>
        </div>
      </div>
      <div className="border-t border-white/5">
        <div className="container-x flex flex-col items-center justify-between gap-2 py-6 text-xs text-slate-500 sm:flex-row">
          <span>© {new Date().getFullYear()} Alpha. Built for the Bittensor ecosystem.</span>
          <span className="font-mono">α — find the gap.</span>
        </div>
      </div>
    </footer>
  );
}
