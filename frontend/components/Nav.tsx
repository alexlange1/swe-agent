"use client";

import Link from "next/link";
import { useState } from "react";
import { Logo } from "./Logo";

const links = [
  { href: "/#features", label: "Features" },
  { href: "/#how", label: "How it works" },
  { href: "/#score", label: "aGap Score" },
  { href: "/#pricing", label: "Pricing" },
  { href: "/dashboard", label: "Dashboard" },
];

export function Nav() {
  const [open, setOpen] = useState(false);
  return (
    <header className="sticky top-0 z-50 border-b border-white/5 bg-ink-950/80 backdrop-blur-xl">
      <nav className="container-x flex h-16 items-center justify-between">
        <Link href="/" className="shrink-0">
          <Logo />
        </Link>
        <div className="hidden items-center gap-7 md:flex">
          {links.map((l) => (
            <Link
              key={l.href}
              href={l.href}
              className="text-sm font-medium text-slate-300 transition hover:text-white"
            >
              {l.label}
            </Link>
          ))}
        </div>
        <div className="hidden items-center gap-3 md:flex">
          <Link href="/dashboard" className="text-sm font-semibold text-slate-200 hover:text-white">
            Sign in
          </Link>
          <Link href="/subscribe" className="btn-primary !py-2">
            Start for Free
          </Link>
        </div>
        <button
          aria-label="Menu"
          className="md:hidden text-slate-200"
          onClick={() => setOpen((v) => !v)}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 6h18M3 12h18M3 18h18" />
          </svg>
        </button>
      </nav>
      {open && (
        <div className="border-t border-white/5 bg-ink-900 md:hidden">
          <div className="container-x flex flex-col gap-1 py-3">
            {links.map((l) => (
              <Link
                key={l.href}
                href={l.href}
                onClick={() => setOpen(false)}
                className="rounded-lg px-3 py-2 text-sm text-slate-300 hover:bg-white/5"
              >
                {l.label}
              </Link>
            ))}
            <Link href="/subscribe" className="btn-primary mt-2" onClick={() => setOpen(false)}>
              Start for Free
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
