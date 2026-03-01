// ─── useDashboard hook ────────────────────────────────────────────────────────
// Centralises ALL data fetching for the dashboard in one place.
// Components subscribe to slices of this data — no component calls the API
// itself, which eliminates duplicate requests and full-page re-renders.
//
// Pattern:
//   1. On mount, load from cache instantly (no spinner).
//   2. Fire all API calls in parallel.
//   3. Update each slice independently as it resolves.
//   4. Auto-refresh every 5 minutes via setInterval.

import { useState, useEffect, useCallback, useRef } from "react";
import {
    fetchCoins,
    fetchGlobalStats,
    fetchTrending,
} from "../services/coingecko";

const REFRESH_INTERVAL = 5 * 60 * 1000; // 5 min

export function useDashboard() {
    const [coins, setCoins] = useState([]);
    const [globalStats, setGlobalStats] = useState(null);
    const [trending, setTrending] = useState([]);

    // Per-slice loading & error — no single spinner blocks everything
    const [status, setStatus] = useState({
        coins: "idle",        // "idle" | "loading" | "ready" | "error"
        global: "idle",
        trending: "idle",
    });
    const [isStale, setIsStale] = useState(false);
    const [lastUpdated, setLastUpdated] = useState(null);
    const [isRefreshing, setIsRefreshing] = useState(false);

    const intervalRef = useRef(null);

    const updateStatus = (key, val) =>
        setStatus((prev) => ({ ...prev, [key]: val }));

    const loadAll = useCallback(async (isManualRefresh = false) => {
        if (isManualRefresh) setIsRefreshing(true);

        let anyStale = false;

        // Fire all 4 requests in parallel — each updates its own slice
        const tasks = [
            fetchCoins(20)
                .then(({ data, stale }) => {
                    setCoins(data);
                    updateStatus("coins", "ready");
                    if (stale) anyStale = true;
                })
                .catch(() => updateStatus("coins", "error")),

            fetchGlobalStats()
                .then(({ data, stale }) => {
                    setGlobalStats(data);
                    updateStatus("global", "ready");
                    if (stale) anyStale = true;
                })
                .catch(() => updateStatus("global", "error")),

            fetchTrending()
                .then(({ data, stale }) => {
                    setTrending(data);
                    updateStatus("trending", "ready");
                    if (stale) anyStale = true;
                })
                .catch(() => updateStatus("trending", "error")),
        ];

        // Set loading state only for slices not yet populated
        setStatus((prev) => ({
            coins:    prev.coins    === "ready" ? "ready" : "loading",
            global:   prev.global   === "ready" ? "ready" : "loading",
            trending: prev.trending === "ready" ? "ready" : "loading",
        }));

        await Promise.allSettled(tasks);

        setIsStale(anyStale);
        setLastUpdated(new Date());
        if (isManualRefresh) setIsRefreshing(false);
    }, []);

    // Initial load
    useEffect(() => {
        loadAll();
        intervalRef.current = setInterval(() => loadAll(), REFRESH_INTERVAL);
        return () => clearInterval(intervalRef.current);
    }, [loadAll]);

    const refresh = () => loadAll(true);

    // Convenience booleans for components
    const isInitialLoad =
        status.coins === "loading" && coins.length === 0;

    return {
        coins,
        globalStats,
        trending,
        status,
        isStale,
        isRefreshing,
        lastUpdated,
        isInitialLoad,
        refresh,
    };
}
