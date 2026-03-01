// ─── useCoinDetail ────────────────────────────────────────────────────────────
// Fetches all data for the CoinDetail page.
// - Coin detail + price chart fire in parallel on mount.
// - Changing `days` only refetches the chart (coin detail stays cached).
// - Each slice has independent loading/error state so the page renders
//   progressively rather than blocking on the slowest request.

import { useState, useEffect, useCallback } from "react";
import { fetchCoinDetail, fetchCoinChart } from "../services/coingecko";

export function useCoinDetail(coinId) {
    const [coin, setCoin]           = useState(null);
    const [chartData, setChartData] = useState(null);
    const [days, setDays]           = useState(7);

    const [coinStatus,  setCoinStatus]  = useState("loading"); // loading | ready | error
    const [chartStatus, setChartStatus] = useState("loading");
    const [isStale,     setIsStale]     = useState(false);

    // ── Load coin detail once on mount (or when coinId changes) ──────────────
    useEffect(() => {
        if (!coinId) return;
        setCoin(null);
        setCoinStatus("loading");

        fetchCoinDetail(coinId)
            .then(({ data, stale }) => {
                setCoin(data);
                setCoinStatus("ready");
                if (stale) setIsStale(true);
            })
            .catch(() => setCoinStatus("error"));
    }, [coinId]);

    // ── Load chart whenever coinId or days changes ────────────────────────────
    const loadChart = useCallback(() => {
        if (!coinId) return;
        setChartStatus("loading");

        fetchCoinChart(coinId, days)
            .then(({ data, stale }) => {
                setChartData(data);
                setChartStatus("ready");
                if (stale) setIsStale(true);
            })
            .catch(() => setChartStatus("error"));
    }, [coinId, days]);

    useEffect(() => { loadChart(); }, [loadChart]);

    return {
        coin,
        chartData,
        days,
        setDays,
        coinStatus,
        chartStatus,
        isStale,
        refetchChart: loadChart,
    };
}
