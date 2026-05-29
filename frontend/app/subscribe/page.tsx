import Link from "next/link";
import { Pricing } from "@/components/Pricing";

export const metadata = { title: "Subscribe | Alpha" };

export default function SubscribePage({
  searchParams,
}: {
  searchParams: { canceled?: string; plan?: string };
}) {
  const canceled = searchParams.canceled === "true";
  return (
    <div className="py-16">
      <div className="container-x">
        {canceled && (
          <div className="mb-8 flex items-start gap-3 rounded-2xl border border-amber-500/30 bg-amber-500/10 p-5">
            <span className="text-xl">⚠️</span>
            <div>
              <div className="font-semibold text-amber-300">Checkout canceled</div>
              <p className="mt-1 text-sm text-amber-200/80">
                No worries — your card was not charged. Pick a plan below whenever you&apos;re
                ready to start finding the alpha gap.
              </p>
            </div>
          </div>
        )}
        <div className="mx-auto max-w-2xl text-center">
          <span className="chip">
            <span className="h-1.5 w-1.5 animate-pulseglow rounded-full bg-alpha-400" />
            Join the edge
          </span>
          <h1 className="mt-5 text-4xl font-extrabold tracking-tight text-white sm:text-5xl">
            Start finding <span className="gradient-text">alpha</span>
          </h1>
          <p className="mt-4 text-lg text-slate-400">
            Free to start. Upgrade when it pays for itself. Cancel anytime.
          </p>
        </div>
      </div>

      <div className="mt-4">
        <Pricing />
      </div>

      <div className="container-x mt-12 text-center">
        <p className="text-sm text-slate-500">
          Want to look around first?{" "}
          <Link href="/dashboard" className="font-semibold text-alpha-400 hover:underline">
            Explore the live dashboard →
          </Link>
        </p>
      </div>
    </div>
  );
}
