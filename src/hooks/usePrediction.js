/**
 * usePrediction.js
 */
import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import api from "../services/api";

const POLL_INTERVAL_MS  = 3000;
const MAX_POLL_ATTEMPTS = 40;

export const ALL_HORIZONS   = ["1w", "1m", "3m", "6m", "1y"];
export const HORIZON_LABELS = { "1w": "1 Week", "1m": "1 Month", "3m": "3 Months", "6m": "6 Months", "1y": "1 Year" };

export const COINS = [
    { symbol: "BTC", name: "Bitcoin",      icon: "₿",  color: "#f7931a" },
    { symbol: "ETH", name: "Ethereum",     icon: "Ξ",  color: "#627eea" },
    { symbol: "BNB", name: "Binance Coin", icon: "⬡",  color: "#f3ba2f" },
    { symbol: "SOL", name: "Solana",       icon: "◎",  color: "#9945ff" },
    { symbol: "XRP", name: "XRP",          icon: "✕",  color: "#346aa9" },
];

export const MODELS = [
    { key: "ensemble",    label: "Ensemble",    color: "#4f7ef7", primary: true  },
    { key: "prophet",     label: "Prophet",     color: "#10b981", primary: false },
    { key: "sarima",      label: "SARIMA",      color: "#f59e0b", primary: false },
    { key: "lstm",        label: "LSTM",        color: "#a78bfa", primary: false },
    { key: "transformer", label: "Transformer", color: "#f472b6", primary: false },
];

export function usePrediction() {
    const [selectedCoin, setSelectedCoin]     = useState("BTC");
    const [activeHorizons, setActiveHorizons] = useState(new Set(ALL_HORIZONS));
    const [visibleModels, setVisibleModels]   = useState(new Set(["ensemble", "prophet", "sarima", "lstm", "transformer"]));
    const [showCI, setShowCI]                 = useState(true);
    const [loading, setLoading]               = useState(false);
    const [error, setError]                   = useState(null);
    const [data, setData]                     = useState(null);
    const [backendReady, setBackendReady]     = useState(null);
    const [trainedCoins, setTrainedCoins]     = useState([]);
    const [lastUpdated, setLastUpdated]       = useState(null);

    const pollRef   = useRef(null);
    const pollCount = useRef(0);

    const checkHealth = useCallback(async () => {
        try {
            const res = await api.get("/api/health");
            const { status, trained_coins } = res.data;
            setTrainedCoins(trained_coins || []);
            if (status === "ready") { setBackendReady(true); return true; }
        } catch { }
        return false;
    }, []);

    useEffect(() => {
        let stopped = false;
        const startPolling = async () => {
            const ready = await checkHealth();
            if (ready || stopped) return;
            setBackendReady(false);
            pollRef.current = setInterval(async () => {
                pollCount.current += 1;
                if (pollCount.current >= MAX_POLL_ATTEMPTS) {
                    clearInterval(pollRef.current);
                    setError("Backend took too long to start. Please refresh the page.");
                    return;
                }
                const r = await checkHealth();
                if (r) clearInterval(pollRef.current);
            }, POLL_INTERVAL_MS);
        };
        startPolling();
        return () => { stopped = true; clearInterval(pollRef.current); };
    }, [checkHealth]);

    const fetchPrediction = useCallback(async (coin) => {
        setLoading(true);
        setError(null);
        try {
            const res = await api.post("/api/predict", { coin });
            setData(res.data);
            setLastUpdated(new Date());
        } catch (err) {
            setError(err.message || "Failed to fetch prediction");
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (backendReady) fetchPrediction(selectedCoin);
    }, [selectedCoin, backendReady, fetchPrediction]);

    const filteredForecast = useMemo(() => {
        if (!data?.forecast) return null;
        const f = data.forecast;
        const indices = f.horizons
            .map((h, i) => ({ h, i }))
            .filter(({ h }) => activeHorizons.has(h));
        const pick = (arr) => arr ? indices.map(({ i }) => arr[i] ?? null) : [];
        return {
            ...f,
            horizons:       indices.map(({ h }) => h),
            ensemble:       pick(f.ensemble),
            ensemble_lower: pick(f.ensemble_lower),
            ensemble_upper: pick(f.ensemble_upper),
            prophet:        pick(f.prophet),
            sarima:         pick(f.sarima),
            lstm:           pick(f.lstm),
            transformer:    pick(f.transformer),
        };
    }, [data, activeHorizons]);

    const toggleHorizon = useCallback((h) => {
        setActiveHorizons(prev => {
            const next = new Set(prev);
            if (next.has(h)) { if (next.size > 1) next.delete(h); }
            else next.add(h);
            return next;
        });
    }, []);

    const toggleModel = useCallback((key) => {
        if (key === "ensemble") return;
        setVisibleModels(prev => {
            const next = new Set(prev);
            if (next.has(key)) { if (next.size > 1) next.delete(key); }
            else next.add(key);
            return next;
        });
    }, []);

    const selectCoin   = useCallback((coin) => { if (!loading) setSelectedCoin(coin); }, [loading]);
    const refresh      = useCallback(() => { if (backendReady && !loading) fetchPrediction(selectedCoin); }, [backendReady, loading, selectedCoin, fetchPrediction]);

    return {
        selectedCoin, selectCoin, trainedCoins,
        activeHorizons, toggleHorizon,
        visibleModels, toggleModel,
        showCI, setShowCI,
        loading, error, data, filteredForecast, backendReady, lastUpdated,
        refresh,
    };
}
