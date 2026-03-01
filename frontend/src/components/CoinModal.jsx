// ─── CoinModal ─────────────────────────────────────────────────────────────────
// Improved detail modal: more stats, better chart, closes on backdrop click.

import React, { memo, useMemo } from "react";
import { Line } from "react-chartjs-2";

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
    return "$" + n.toLocaleString();
}

const CoinModal = memo(({ coin, onClose }) => {
    const pct = coin.price_change_percentage_24h;
    const up  = pct >= 0;

    const chartData = useMemo(() => ({
        labels: coin.sparkline_in_7d?.price?.map((_, i) => {
            const day = Math.floor(i / 24);
            return i % 24 === 0 ? `Day ${day + 1}` : "";
        }) || [],
        datasets: [{
            label: coin.symbol.toUpperCase() + " 7d",
            data: coin.sparkline_in_7d?.price || [],
            borderColor: up ? "#10b981" : "#ef4444",
            backgroundColor: up ? "rgba(16,185,129,0.08)" : "rgba(239,68,68,0.08)",
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 2,
        }],
    }), [coin.id]);

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: "rgba(15,15,30,0.92)",
                callbacks: {
                    label: ctx => ` $${ctx.parsed.y.toLocaleString("en-US", {
                        minimumFractionDigits: 2, maximumFractionDigits: 2,
                    })}`,
                },
            },
        },
        scales: {
            x: { grid: { display: false }, ticks: { font: { size: 10 }, maxTicksLimit: 8 } },
            y: {
                grid: { color: "rgba(0,0,0,0.04)" },
                ticks: { font: { size: 10 }, callback: v => "$" + v.toLocaleString() },
            },
        },
    };

    return (
        <div className="modal-backdrop" onClick={onClose}>
            <div className="coin-modal" onClick={e => e.stopPropagation()}>
                <button className="coin-modal__close" onClick={onClose} aria-label="Close">✕</button>

                {/* Header */}
                <div className="coin-modal__header">
                    <img src={coin.image} alt={coin.name} width={40} height={40} />
                    <div>
                        <h3 className="coin-modal__name">{coin.name}</h3>
                        <span className="coin-modal__symbol">{coin.symbol.toUpperCase()}</span>
                    </div>
                    <div className="coin-modal__price-block">
                        <span className="coin-modal__price">{fmtPrice(coin.current_price)}</span>
                        <span className={`coin-modal__pct ${up ? "up" : "down"}`}>
                            {up ? "▲" : "▼"} {Math.abs(pct || 0).toFixed(2)}%
                        </span>
                    </div>
                </div>

                {/* Stats grid */}
                <div className="coin-modal__stats">
                    {[
                        ["Market Cap",    fmtLarge(coin.market_cap)],
                        ["24h Volume",    fmtLarge(coin.total_volume)],
                        ["24h High",      fmtPrice(coin.high_24h)],
                        ["24h Low",       fmtPrice(coin.low_24h)],
                        ["ATH",           fmtPrice(coin.ath)],
                        ["Circulating",   coin.circulating_supply
                            ? coin.circulating_supply.toLocaleString("en-US", { maximumFractionDigits: 0 })
                            : "—"],
                    ].map(([label, val]) => (
                        <div key={label} className="coin-modal__stat">
                            <span className="coin-modal__stat-label">{label}</span>
                            <span className="coin-modal__stat-value">{val}</span>
                        </div>
                    ))}
                </div>

                {/* Chart */}
                <div className="coin-modal__chart-title">7-Day Price Chart</div>
                <div className="coin-modal__chart">
                    <Line data={chartData} options={chartOptions} />
                </div>
            </div>
        </div>
    );
});

CoinModal.displayName = "CoinModal";
export default CoinModal;
