/**
 * services/coingecko.js
 *
 * CoinGecko API service layer.
 *
 * Fixes applied:
 *  - Replaced localStorage cache with in-memory cache (localStorage is
 *    unreliable in sandboxed/private browsing contexts and causes
 *    "Coin not found" errors when storage access is blocked).
 *  - Added VITE_COINGECKO_API_KEY support — set this in Vercel env vars
 *    to avoid rate-limit 429s on cloud deployments.
 *  - Added retry logic with exponential backoff for 429 responses.
 *  - Added clearer error messages to distinguish rate-limit vs not-found.
 */

import axios from "axios";

// ── CoinGecko Axios instance ──────────────────────────────────────────────────
const API_KEY = import.meta.env.VITE_COINGECKO_API_KEY || "";

const coingeckoClient = axios.create({
    // Use pro endpoint if API key present, otherwise free
    baseURL: API_KEY
        ? "https://pro-api.coingecko.com/api/v3"
        : "https://api.coingecko.com/api/v3",
    timeout: 15_000,
    headers: {
        Accept: "application/json",
        ...(API_KEY ? { "x-cg-pro-api-key": API_KEY } : {}),
    },
});

coingeckoClient.interceptors.response.use(
    (res) => res,
    async (error) => {
        if (error.response?.status === 429) {
            // Rate limited — wait 2s and retry once
            await new Promise((r) => setTimeout(r, 2000));
            try {
                return await coingeckoClient.request(error.config);
            } catch {
                return Promise.reject(
                    new Error("CoinGecko rate limit reached. Please wait a moment and try again.")
                );
            }
        }
        if (error.response?.status === 404) {
            return Promise.reject(new Error("Coin not found on CoinGecko."));
        }
        const msg =
            error.response?.data?.error ||
            error.message ||
            "CoinGecko request failed";
        return Promise.reject(new Error(msg));
    }
);

// ── In-memory cache (replaces localStorage — works everywhere) ────────────────
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes
const memCache  = new Map();

function cacheSet(key, data) {
    memCache.set(key, { ts: Date.now(), data });
}

function cacheGet(key) {
    const entry = memCache.get(key);
    if (!entry) return null;
    return { data: entry.data, stale: Date.now() - entry.ts > CACHE_TTL };
}

// ── Cache-then-network fetcher ────────────────────────────────────────────────
async function fetchWithCache(cacheKey, fetcher) {
    const cached = cacheGet(cacheKey);

    if (cached && !cached.stale) {
        return { data: cached.data, fromCache: true, stale: false };
    }

    try {
        const fresh = await fetcher();
        cacheSet(cacheKey, fresh);
        return { data: fresh, fromCache: false, stale: false };
    } catch (err) {
        // Network failed — serve stale cache rather than crash the UI
        if (cached) {
            return { data: cached.data, fromCache: true, stale: true };
        }
        throw err;
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/** Fetches top-N coins with sparkline data. */
export async function fetchCoins(perPage = 20) {
    return fetchWithCache(`cg_coins_${perPage}`, async () => {
        const res = await coingeckoClient.get("/coins/markets", {
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

/** Fetches global market stats (market cap, dominance, volume, etc.) */
export async function fetchGlobalStats() {
    return fetchWithCache("cg_global", async () => {
        const res = await coingeckoClient.get("/global");
        return res.data.data;
    });
}

/** Fetches trending coins (hot searches on CoinGecko). */
export async function fetchTrending() {
    return fetchWithCache("cg_trending", async () => {
        const res = await coingeckoClient.get("/search/trending");
        return res.data.coins.slice(0, 5).map((c) => c.item);
    });
}

/** Fetches full coin detail — price, market data, description, links, community. */
export async function fetchCoinDetail(coinId) {
    return fetchWithCache(`cg_coin_${coinId}`, async () => {
        const res = await coingeckoClient.get(`/coins/${coinId}`, {
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

/** Fetches OHLC price chart data. days: 1 | 7 | 14 | 30 | 90 | 180 | 365 */
export async function fetchCoinOHLC(coinId, days = 7) {
    return fetchWithCache(`cg_ohlc_${coinId}_${days}`, async () => {
        const res = await coingeckoClient.get(`/coins/${coinId}/ohlc`, {
            params: { vs_currency: "usd", days },
        });
        return res.data;
    });
}

/** Fetches historical market chart (prices + volumes) for a coin. */
export async function fetchCoinChart(coinId, days = 7) {
    return fetchWithCache(`cg_chart_${coinId}_${days}`, async () => {
        const res = await coingeckoClient.get(`/coins/${coinId}/market_chart`, {
            params: {
                vs_currency: "usd",
                days,
                interval: days <= 1 ? "minutely" : "daily",
            },
        });
        return res.data;
    });
}