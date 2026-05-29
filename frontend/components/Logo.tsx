export function Logo({ className = "" }: { className?: string }) {
  return (
    <span className={`inline-flex items-center gap-2 ${className}`}>
      <span className="relative flex h-8 w-8 items-center justify-center rounded-lg bg-alpha-500/15 ring-1 ring-alpha-500/40">
        <span className="text-lg font-black text-alpha-400">α</span>
        <span className="absolute -right-0.5 -top-0.5 h-2 w-2 animate-pulseglow rounded-full bg-alpha-400" />
      </span>
      <span className="text-lg font-extrabold tracking-tight text-white">
        Alpha
      </span>
    </span>
  );
}
