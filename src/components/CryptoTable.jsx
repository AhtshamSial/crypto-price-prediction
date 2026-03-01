// ─── CryptoTable ──────────────────────────────────────────────────────────────
// Full coin table. memo() + row-level memo prevents re-rendering all 20 rows
// when only one coin's price updates.

import React, { memo, useState } from "react";
import { Line } from "react-chartjs-2";
import { useNavigate } from "react-router-dom";
import "../Styles/Table.css";

const sparklineOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { display: false }, tooltip: { enabled: false } },
    scales: { x: { display: false }, y: { display: false } },
    elements: { point: { radius: 0 } },
};

function fmtPrice(n) {
    if (n === null || n === undefined) return "—";
    if (n >= 1000) return "$" + n.toLocaleString("en-US", { maximumFractionDigits: 0 });
    if (n >= 1)    return "$" + n.toFixed(2);
    return "$" + n.toPrecision(4);
}

function fmtLarge(n) {
    if (!n) return "—";
    if (n >= 1e12) return "$" + (n / 1e12).toFixed(2) + "T";
    if (n >= 1e9)  return "$" + (n / 1e9).toFixed(1) + "B";
    if (n >= 1e6)  return "$" + (n / 1e6).toFixed(1) + "M";
    return "$" + n.toLocaleString();
}

// Single row — memo so only changed rows re-render
const CoinRow = memo(({ coin, rank }) => {
    const navigate = useNavigate();
    const pct24h = coin.price_change_percentage_24h;
    const pct7d  = coin.price_change_percentage_7d_in_currency;
    const up24   = pct24h >= 0;
    const up7d   = pct7d  >= 0;

    const sparkData = {
        labels: coin.sparkline_in_7d?.price?.map((_, i) => i) || [],
        datasets: [{
            data: coin.sparkline_in_7d?.price || [],
            borderColor: up24 ? "#10b981" : "#ef4444",
            borderWidth: 1.5,
            fill: false,
            tension: 0.3,
        }],
    };

    return (
        <tr className="coin-row" onClick={() => navigate(`/coin/${coin.id}`)}>
            <td className="col-rank">{rank}</td>
            <td className="col-coin">
                <img src={coin.image} alt={coin.name} width={22} height={22} className="coin-row__img" />
                <div className="coin-row__names">
                    <span className="coin-row__name">{coin.name}</span>
                    <span className="coin-row__symbol">{coin.symbol.toUpperCase()}</span>
                </div>
            </td>
            <td className="col-price">{fmtPrice(coin.current_price)}</td>
            <td className={`col-change ${up24 ? "col-change--up" : "col-change--down"}`}>
                {pct24h != null ? `${up24 ? "+" : ""}${pct24h.toFixed(2)}%` : "—"}
            </td>
            <td className={`col-change ${up7d ? "col-change--up" : "col-change--down"}`}>
                {pct7d != null ? `${up7d ? "+" : ""}${pct7d.toFixed(2)}%` : "—"}
            </td>
            <td className="col-mcap">{fmtLarge(coin.market_cap)}</td>
            <td className="col-vol">{fmtLarge(coin.total_volume)}</td>
            <td className="col-spark">
                <div className="sparkline-cell">
                    <Line data={sparkData} options={sparklineOptions} />
                </div>
            </td>
        </tr>
    );
});
CoinRow.displayName = "CoinRow";

// Table skeleton while loading
function TableSkeleton({ rows = 10 }) {
    return (
        <tbody>
            {Array.from({ length: rows }).map((_, i) => (
                <tr key={i} className="skeleton-row">
                    {[40, 160, 90, 60, 60, 110, 100, 100].map((w, j) => (
                        <td key={j}><div className="skeleton-line" style={{ width: w }} /></td>
                    ))}
                </tr>
            ))}
        </tbody>
    );
}

const CryptoTable = memo(({ coins, loading }) => {
    const [sortKey, setSortKey] = useState("market_cap");
    const [sortDir, setSortDir] = useState("desc");

    const handleSort = (key) => {
        if (sortKey === key) {
            setSortDir(d => d === "asc" ? "desc" : "asc");
        } else {
            setSortKey(key);
            setSortDir("desc");
        }
    };

    const sorted = [...coins].sort((a, b) => {
        const av = a[sortKey] ?? 0;
        const bv = b[sortKey] ?? 0;
        return sortDir === "asc" ? av - bv : bv - av;
    });

    const SortIcon = ({ col }) => {
        if (sortKey !== col) return <span className="sort-icon sort-icon--none">⇅</span>;
        return <span className="sort-icon">{sortDir === "asc" ? "↑" : "↓"}</span>;
    };

    const th = (label, key) => (
        <th onClick={() => handleSort(key)} className={`sortable ${sortKey === key ? "active" : ""}`}>
            {label} <SortIcon col={key} />
        </th>
    );

    return (
        <div className="coin-table-wrapper">
            <div className="coin-table-scroll">
                <table className="coin-table">
                    <thead>
                        <tr>
                            <th className="col-rank">#</th>
                            <th className="col-coin">Coin</th>
                            {th("Price", "current_price")}
                            {th("24h %", "price_change_percentage_24h")}
                            {th("7d %", "price_change_percentage_7d_in_currency")}
                            {th("Market Cap", "market_cap")}
                            {th("24h Volume", "total_volume")}
                            <th className="col-spark">7d Chart</th>
                        </tr>
                    </thead>
                    {loading && coins.length === 0
                        ? <TableSkeleton />
                        : (
                            <tbody>
                                {sorted.map((coin, i) => (
                                    <CoinRow
                                        key={coin.id}
                                        coin={coin}
                                        rank={i + 1}
                                    />
                                ))}
                            </tbody>
                        )
                    }
                </table>
            </div>
        </div>
    );
});

CryptoTable.displayName = "CryptoTable";
export default CryptoTable;
