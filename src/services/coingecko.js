// src/services/coingecko.js
import axios from "axios";

const COINGECKO_BASE = "https://api.coingecko.com/api/v3";

export async function fetchMarketData() {
    const response = await axios.get(`${COINGECKO_BASE}/coins/markets`, {
        params: { vs_currency: "usd", order: "market_cap_desc", per_page: 10, page: 1 }
    });
    return response.data;
}

export async function fetchCoinChart(coinId = "bitcoin") {
    const response = await axios.get(`${COINGECKO_BASE}/coins/${coinId}/market_chart`, {
        params: { vs_currency: "usd", days: 7 }
    });
    return response.data;
}
