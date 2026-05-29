"use client";

import { useCallback, useEffect, useState } from "react";

const KEY = "alpha.watchlist";

function read(): number[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(KEY) || "[]");
  } catch {
    return [];
  }
}

export function useWatchlist() {
  const [list, setList] = useState<number[]>([]);

  useEffect(() => {
    setList(read());
    const onStorage = () => setList(read());
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const toggle = useCallback((netuid: number) => {
    setList((cur) => {
      const next = cur.includes(netuid)
        ? cur.filter((n) => n !== netuid)
        : [...cur, netuid];
      localStorage.setItem(KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const has = useCallback((netuid: number) => list.includes(netuid), [list]);

  return { list, toggle, has };
}
