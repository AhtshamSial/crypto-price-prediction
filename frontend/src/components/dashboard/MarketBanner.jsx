import React, { memo } from "react";

function fmt(n) {
    if (!n) return "$0";
    if (n >= 1e12) return "$" + (n / 1e12).toFixed(2) + "T";
    if (n >= 1e9)  return "$" + (n / 1e9).toFixed(1) + "B";
    if (n >= 1e6)  return "$" + (n / 1e6).toFixed(1) + "M";
    return "$" + n.toLocaleString();
}

const MarketBanner = memo(({ globalStats, isStale }) => {
    if (!globalStats) {
        return (
            <div className="market-banner market-banner--skeleton">
                {[1,2,3,4,5].map(i => (
                    <div key={i} className="banner-item">
                        <div className="skeleton-line" style={{ width: "80px" }} />
                    </div>
                ))}
            </div>
        );
    }

    const mcap = globalStats.total_market_cap?.usd || 0;
    const vol  = globalStats.total_volume?.usd || 0;
    const pct  = globalStats.market_cap_change_percentage_24h_usd || 0;
    const btc  = globalStats.market_cap_percentage?.btc || 0;
    const eth  = globalStats.market_cap_percentage?.eth || 0;
    const coins = globalStats.active_cryptocurrencies || 0;
    const up   = pct >= 0;

    return (
        <div className={`market-banner ${isStale ? "market-banner--stale" : ""}`}>
            <div className="banner-item">
                <span className="banner-label">Coins:</span>
                <span className="banner-value">{coins.toLocaleString()}</span>
            </div>
            <div className="banner-separator" />
            <div className="banner-item">
                <span className="banner-label">Market Cap:</span>
                <span className="banner-value">{fmt(mcap)}</span>
                <span className={`banner-change ${up ? "up" : "down"}`}>
                    {up ? "▲" : "▼"}{Math.abs(pct).toFixed(1)}%
                </span>
            </div>
            <div className="banner-separator" />
            <div className="banner-item">
                <span className="banner-label">24h Vol:</span>
                <span className="banner-value">{fmt(vol)}</span>
            </div>
            <div className="banner-separator" />
            <div className="banner-item">
                <span className="banner-label">Dominance:</span>
                <span className="banner-value">BTC {btc.toFixed(1)}%</span>
                <span className="banner-value" style={{ marginLeft: "8px" }}>ETH {eth.toFixed(1)}%</span>
            </div>
            {isStale && (
                <div className="banner-stale-badge" title="Showing cached data">
                    ⚡ Cached
                </div>
            )}
        </div>
    );
});

MarketBanner.displayName = "MarketBanner";
export default MarketBanner;
