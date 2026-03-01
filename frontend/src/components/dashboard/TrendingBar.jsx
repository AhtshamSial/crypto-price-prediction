// ─── TrendingBar ──────────────────────────────────────────────────────────────
// Horizontal scrolling strip of trending coins (mirrors CoinGecko's 🔥 section).
// Completely isolated — only re-renders when `trending` prop changes.

import React, { memo } from "react";

const TrendingBar = memo(({ trending, loading }) => {
    if (loading || !trending || trending.length === 0) {
        return (
            <div className="trending-bar">
                <span className="trending-bar__label">🔥 Trending</span>
                {[1,2,3,4,5].map(i => (
                    <div key={i} className="trending-pill trending-pill--skeleton">
                        <div className="skeleton-line" style={{ width: "60px" }} />
                    </div>
                ))}
            </div>
        );
    }

    return (
        <div className="trending-bar">
            <span className="trending-bar__label">🔥 Trending</span>
            <div className="trending-bar__items">
                {trending.map((coin) => {
                    const pct = coin.data?.price_change_percentage_24h?.usd;
                    const up  = pct >= 0;
                    return (
                        <div key={coin.id} className="trending-pill">
                            <img
                                src={coin.small}
                                alt={coin.symbol}
                                width={16}
                                height={16}
                                className="trending-pill__img"
                            />
                            <span className="trending-pill__name">{coin.symbol.toUpperCase()}</span>
                            {pct !== undefined && (
                                <span className={`trending-pill__change ${up ? "up" : "down"}`}>
                                    {up ? "+" : ""}{pct.toFixed(1)}%
                                </span>
                            )}
                        </div>
                    );
                })}
            </div>
        </div>
    );
});

TrendingBar.displayName = "TrendingBar";
export default TrendingBar;
