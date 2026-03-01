// ─── StatCard ─────────────────────────────────────────────────────────────────
// Metric card with a prominent embedded area sparkline chart.
// Supports gradient fill, interactive tooltip on hover, and animated entry.

import React, { memo, useMemo } from "react";
import { Line } from "react-chartjs-2";

function buildChartData(prices, color, up) {
    if (!prices || prices.length === 0) return null;

    // Thin the dataset to ~40 points for performance while keeping shape
    const step = Math.max(1, Math.floor(prices.length / 40));
    const thinned = prices.filter((_, i) => i % step === 0);

    return {
        labels: thinned.map((_, i) => i),
        datasets: [{
            data: thinned,
            borderColor: color,
            borderWidth: 2,
            fill: true,
            backgroundColor: (ctx) => {
                if (!ctx.chart.chartArea) return "transparent";
                const { ctx: c, chartArea: { top, bottom } } = ctx.chart;
                const grad = c.createLinearGradient(0, top, 0, bottom);
                grad.addColorStop(0,   color + "40"); // 25% opacity at top
                grad.addColorStop(0.6, color + "12"); // 7%  opacity mid
                grad.addColorStop(1,   color + "00"); // 0%  opacity at bottom
                return grad;
            },
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: color,
            pointHoverBorderColor: "#fff",
            pointHoverBorderWidth: 2,
        }],
    };
}

function buildChartOptions(color, formatTooltip) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600, easing: "easeOutQuart" },
        interaction: { mode: "index", intersect: false },
        plugins: {
            legend: { display: false },
            tooltip: {
                enabled: true,
                backgroundColor: "rgba(10,12,20,0.88)",
                padding: 8,
                cornerRadius: 6,
                titleFont: { size: 0 },     // hide title (just show value)
                bodyFont: { size: 12, weight: "600" },
                bodyColor: color,
                callbacks: {
                    title: () => "",          // suppress date labels
                    label: (ctx) => formatTooltip(ctx.parsed.y),
                },
            },
        },
        scales: {
            x: { display: false },
            y: { display: false },
        },
    };
}

// ─── Skeleton ─────────────────────────────────────────────────────────────────
function StatCardSkeleton() {
    return (
        <div className="stat-card stat-card--loading">
            <div className="skeleton-block" style={{ height: "13px", width: "55%", marginBottom: "10px" }} />
            <div className="skeleton-block" style={{ height: "30px", width: "75%", marginBottom: "6px" }} />
            <div className="skeleton-block" style={{ height: "14px", width: "40%", marginBottom: "14px" }} />
            <div className="skeleton-block" style={{ height: "64px", borderRadius: "8px" }} />
        </div>
    );
}

// ─── StatCard ─────────────────────────────────────────────────────────────────
const StatCard = memo(({
    icon,
    label,
    value,
    subValue,
    change,
    chartPrices,
    chartColor = "#667eea",
    formatTooltip,
    loading,
}) => {
    if (loading) return <StatCardSkeleton />;

    const up = change === undefined ? true : change >= 0;
    const lineColor = chartPrices?.length
        ? chartColor
        : (up ? "#10b981" : "#ef4444");

    const tooltipFn = formatTooltip || ((v) => v.toLocaleString());

    const chartData    = useMemo(() => buildChartData(chartPrices, lineColor, up),
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [chartPrices?.length, lineColor]);

    const chartOptions = useMemo(() => buildChartOptions(lineColor, tooltipFn),
        [lineColor]);

    return (
        <div className="stat-card">
            {/* Header row: icon + label */}
            <div className="stat-card__header">
                <span className="stat-card__icon">{icon}</span>
                <span className="stat-card__label">{label}</span>
            </div>

            {/* Primary value */}
            <div className="stat-card__value">{value}</div>

            {/* Sub-text or change badge */}
            <div className="stat-card__meta">
                {change !== undefined && (
                    <span className={`stat-card__change ${up ? "stat-card__change--up" : "stat-card__change--down"}`}>
                        {up ? "▲" : "▼"} {Math.abs(change).toFixed(2)}%
                        <span className="stat-card__change-label"> 24h</span>
                    </span>
                )}
                {subValue && (
                    <span className="stat-card__sub">{subValue}</span>
                )}
            </div>

            {/* Chart area — always present, renders skeleton bars if no data */}
            <div className="stat-card__chart">
                {chartData
                    ? <Line data={chartData} options={chartOptions} />
                    : <div className="stat-card__chart-empty" />
                }
            </div>
        </div>
    );
});

StatCard.displayName = "StatCard";
export default StatCard;
