import React, { memo } from "react";
import { fmtPrice, fmtPct, fmtCompact } from "./utils";

function StatTile({ label, value, up, mono }) {
    return (
        <div className="pred-market-tile">
            <span className="pred-tile-label">{label}</span>
            <span className={`pred-tile-value${mono ? " mono" : ""}${up === true ? " up" : up === false ? " down" : ""}`}>
                {value}
            </span>
        </div>
    );
}

const MarketSummary = memo(({ market, coin }) => {
    if (!market) return null;
    const chg = market.price_change_pct_24h;
    return (
        <div className="pred-market-bar">
            <div className="pred-market-coin-badge">
                <span className="pred-market-coin-label">{coin}</span>
                <span className="pred-market-coin-sub">/ USDT</span>
            </div>
            <div className="pred-market-tiles">
                <StatTile label="Price"      value={fmtPrice(market.price)} mono />
                <StatTile label="24h Change" value={fmtPct(chg)} up={chg != null ? chg >= 0 : undefined} mono />
                <StatTile label="24h Volume" value={fmtCompact(market.volume_24h)} mono />
                <StatTile label="Bid"        value={fmtPrice(market.bid)} mono />
                <StatTile label="Ask"        value={fmtPrice(market.ask)} mono />
                <StatTile label="Spread"     value={fmtPrice(market.spread)} mono />
            </div>
        </div>
    );
});

MarketSummary.displayName = "MarketSummary";
export default MarketSummary;
