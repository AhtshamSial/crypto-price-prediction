import React from "react";
import "../App.css";

const GlobalStats = ({ stats }) => {
    return (
        <div className="global-stats-container">
            <div className="global-stats-row">
                <div className="stat-card border-primary border border-2">
                    <h6>Total Market Cap</h6>
                    <h4>${stats.total_market_cap?.usd.toLocaleString()}</h4>
                    <small>USD</small>
                </div>
                <div className="stat-card border-success border border-2">
                    <h6>BTC Dominance</h6>
                    <h4>{stats.market_cap_percentage?.btc.toFixed(2)}%</h4>
                    <small>Of total market</small>
                </div>
                <div className="stat-card border-warning border border-2">
                    <h6>Active Coins</h6>
                    <h4>{stats.active_cryptocurrencies}</h4>
                    <small>Coins tracked</small>
                </div>
                <div className="stat-card border-danger border  border-2">
                    <h6>24h Volume</h6>
                    <h4>${stats.total_volume?.usd.toLocaleString()}</h4>
                    <small>USD</small>
                </div>
            </div>
        </div>
    );
};

export default GlobalStats;
