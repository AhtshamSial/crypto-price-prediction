// ─── CoinGecko API Service ────────────────────────────────────────────────────
// All API calls live here. Dashboard components never call axios directly.
// Implements a localStorage cache layer: stale data is shown immediately while
// fresh data loads in the background, so the spinner never blocks the UI.

import axios from "axios";

const BASE = "https://api.coingecko.com/api/v3";
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes in ms

// ─── Cache helpers ────────────────────────────────────────────────────────────

function cacheSet(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify({ ts: Date.now(), data }));
    } catch {
        // localStorage may be full — fail silently
    }
}

function cacheGet(key) {
    try {
        const raw = localStorage.getItem(key);
        if (!raw) return null;
        const { ts, data } = JSON.parse(raw);
        return { data, stale: Date.now() - ts > CACHE_TTL };
    } catch {
        return null;
    }
}

// ─── Generic fetcher with cache-then-network pattern ─────────────────────────
// Returns { data, fromCache, stale } so callers can show staleness indicators.

async function fetchWithCache(cacheKey, fetcher) {
    const cached = cacheGet(cacheKey);

    // Return cached data immediately so UI never blocks
    if (cached && !cached.stale) {
        return { data: cached.data, fromCache: true, stale: false };
    }

    try {
        const fresh = await fetcher();
        cacheSet(cacheKey, fresh);
        return { data: fresh, fromCache: false, stale: false };
    } catch (err) {
        // Network failed — serve stale cache rather than crash
        if (cached) {
            return { data: cached.data, fromCache: true, stale: true };
        }
        throw err;
    }
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Fetches top-N coins with sparkline data.
 */
export async function fetchCoins(perPage = 20) {
    return fetchWithCache(`cg_coins_${perPage}`, async () => {
        const res = await axios.get(`${BASE}/coins/markets`, {
            params: {
                vs_currency: "usd",
                order: "market_cap_desc",
                per_page: perPage,
                page: 1,
                sparkline: true,
                price_change_percentage: "1h,24h,7d",
            },
        });
        return res.data;
    });
}

/**
 * Fetches global market stats (market cap, dominance, volume, etc.)
 */
export async function fetchGlobalStats() {
    return fetchWithCache("cg_global", async () => {
        const res = await axios.get(`${BASE}/global`);
        return res.data.data;
    });
}

/**
 * Fetches trending coins (hot searches on CoinGecko).
 */
export async function fetchTrending() {
    return fetchWithCache("cg_trending", async () => {
        const res = await axios.get(`${BASE}/search/trending`);
        return res.data.coins.slice(0, 5).map((c) => c.item);
    });
}

/**
 * Fetches full coin detail — price, market data, description, links, community.
 * Cached for 5 min. Used exclusively by the CoinDetail page.
 */
export async function fetchCoinDetail(coinId) {
    return fetchWithCache(`cg_coin_${coinId}`, async () => {
        const res = await axios.get(`${BASE}/coins/${coinId}`, {
            params: {
                localization: false,
                tickers: false,
                market_data: true,
                community_data: true,
                developer_data: false,
                sparkline: true,
            },
        });
        return res.data;
    });
}

/**
 * Fetches OHLC price chart data for a coin.
 * days: 1 | 7 | 14 | 30 | 90 | 180 | 365
 */
export async function fetchCoinOHLC(coinId, days = 7) {
    return fetchWithCache(`cg_ohlc_${coinId}_${days}`, async () => {
        const res = await axios.get(`${BASE}/coins/${coinId}/ohlc`, {
            params: { vs_currency: "usd", days },
        });
        return res.data; // [[timestamp, open, high, low, close], ...]
    });
}

/**
 * Fetches historical market chart (prices + volumes) for a coin.
 * Used to draw the main price line chart.
 */
export async function fetchCoinChart(coinId, days = 7) {
    return fetchWithCache(`cg_chart_${coinId}_${days}`, async () => {
        const res = await axios.get(`${BASE}/coins/${coinId}/market_chart`, {
            params: { vs_currency: "usd", days, interval: days <= 1 ? "minutely" : "daily" },
        });
        return res.data; // { prices: [[ts, price]], market_caps: [...], total_volumes: [...] }
    });
}
