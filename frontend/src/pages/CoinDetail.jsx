import React, { useMemo, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { Line } from "react-chartjs-2";
import {
    Chart as ChartJS, CategoryScale, LinearScale,
    PointElement, LineElement, Tooltip, Legend, Filler,
} from "chart.js";
import { useCoinDetail } from "../hooks/useCoinDetail";
import "../Styles/CoinDetail.css";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

// ── Formatters ────────────────────────────────────────────────────────────────
function fmtUSD(n) {
    if (n === null || n === undefined) return "—";
    if (n >= 1e12) return "$" + (n / 1e12).toFixed(3) + "T";
    if (n >= 1e9)  return "$" + (n / 1e9).toFixed(2) + "B";
    if (n >= 1e6)  return "$" + (n / 1e6).toFixed(2) + "M";
    if (n >= 1000) return "$" + n.toLocaleString("en-US", { maximumFractionDigits: 2 });
    if (n >= 1)    return "$" + n.toFixed(4);
    return "$" + n.toPrecision(4);
}
function fmtPct(n) {
    if (n === null || n === undefined) return "—";
    const sign = n >= 0 ? "+" : "";
    return sign + n.toFixed(2) + "%";
}
function fmtNum(n) {
    if (!n) return "—";
    return n.toLocaleString("en-US", { maximumFractionDigits: 0 });
}

// ── Skeleton blocks ───────────────────────────────────────────────────────────
function Skeleton({ w = "100%", h = 16, r = 6 }) {
    return <div className="cd-skeleton" style={{ width: w, height: h, borderRadius: r }} />;
}

// ── Stat row item ─────────────────────────────────────────────────────────────
function StatRow({ label, value, up, isChange }) {
    return (
        <div className="cd-stat-row">
            <span className="cd-stat-row__label">{label}</span>
            <span className={`cd-stat-row__value ${isChange ? (up ? "cd-up" : "cd-down") : ""}`}>
                {value}
            </span>
        </div>
    );
}

// ── Range selector ────────────────────────────────────────────────────────────
const RANGES = [
    { label: "24H", days: 1 },
    { label: "7D",  days: 7 },
    { label: "30D", days: 30 },
    { label: "90D", days: 90 },
    { label: "1Y",  days: 365 },
];

// ── Main page ─────────────────────────────────────────────────────────────────
export default function CoinDetail() {
    const { coinId } = useParams();
    const navigate   = useNavigate();
    const { coin, chartData, days, setDays, coinStatus, chartStatus, isStale } = useCoinDetail(coinId);
    const [aboutExpanded, setAboutExpanded] = useState(false);

    // Derived values
    const md        = coin?.market_data;
    const price     = md?.current_price?.usd;
    const pct24h    = md?.price_change_percentage_24h;
    const pct7d     = md?.price_change_percentage_7d;
    const pct30d    = md?.price_change_percentage_30d;
    const pct1y     = md?.price_change_percentage_1y;
    const up24      = pct24h >= 0;
    const accentColor = up24 ? "#10b981" : "#ef4444";

    // Chart
    const lineData = useMemo(() => {
        if (!chartData?.prices?.length) return null;
        const pts = chartData.prices;
        return {
            labels: pts.map(([ts]) => {
                const d = new Date(ts);
                return days <= 1
                    ? d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
                    : d.toLocaleDateString([], { month: "short", day: "numeric" });
            }),
            datasets: [{
                label: coin?.symbol?.toUpperCase() + " Price",
                data: pts.map(([, p]) => p),
                borderColor: accentColor,
                backgroundColor: (ctx) => {
                    if (!ctx.chart.chartArea) return "transparent";
                    const { ctx: c, chartArea: { top, bottom } } = ctx.chart;
                    const g = c.createLinearGradient(0, top, 0, bottom);
                    g.addColorStop(0,   accentColor + "33");
                    g.addColorStop(0.7, accentColor + "08");
                    g.addColorStop(1,   accentColor + "00");
                    return g;
                },
                fill: true,
                tension: 0.35,
                pointRadius: 0,
                pointHoverRadius: 5,
                pointHoverBackgroundColor: accentColor,
                pointHoverBorderColor: "#fff",
                pointHoverBorderWidth: 2,
                borderWidth: 2,
            }],
        };
    }, [chartData, accentColor, days, coin?.symbol]);

    const lineOptions = useMemo(() => ({
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        animation: { duration: 400 },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: "rgba(13,17,23,0.92)",
                padding: 12,
                cornerRadius: 8,
                borderColor: "rgba(255,255,255,0.08)",
                borderWidth: 1,
                titleFont: { size: 11, weight: "500" },
                bodyFont: { size: 14, weight: "700", family: "'IBM Plex Mono'" },
                bodyColor: accentColor,
                callbacks: { label: (ctx) => " " + fmtUSD(ctx.parsed.y) },
            },
        },
        scales: {
            x: {
                grid: { display: false },
                border: { display: false },
                ticks: {
                    color: "#6b7a99",
                    font: { size: 11 },
                    maxTicksLimit: days <= 1 ? 8 : 7,
                    maxRotation: 0,
                },
            },
            y: {
                position: "right",
                grid: { color: "rgba(0,0,0,0.04)", drawBorder: false },
                border: { display: false },
                ticks: {
                    color: "#6b7a99",
                    font: { size: 11, family: "'IBM Plex Mono'" },
                    callback: (v) => fmtUSD(v),
                    maxTicksLimit: 6,
                },
            },
        },
    }), [accentColor, days]);

    // Error state
    if (coinStatus === "error") {
        return (
            <div className="cd-error">
                <div className="cd-error__icon">⚠️</div>
                <h2>Coin not found</h2>
                <p>"{coinId}" doesn't exist or the API is unavailable.</p>
                <button onClick={() => navigate("/")}>← Back to Dashboard</button>
            </div>
        );
    }

    return (
        <div className="cd-page">

            {/* ── Breadcrumb ── */}
            <div className="cd-breadcrumb">
                <Link to="/" className="cd-breadcrumb__link">Dashboard</Link>
                <span className="cd-breadcrumb__sep">›</span>
                {coin
                    ? <span className="cd-breadcrumb__current">{coin.name}</span>
                    : <Skeleton w={80} h={14} />
                }
            </div>

            {/* ── Hero header ── */}
            <div className="cd-hero">
                <div className="cd-hero__left">
                    {coin
                        ? <img src={coin.image?.large} alt={coin.name} className="cd-hero__img" />
                        : <Skeleton w={56} h={56} r={50} />
                    }
                    <div className="cd-hero__names">
                        {coin ? (
                            <>
                                <h1 className="cd-hero__name">{coin.name}</h1>
                                <div className="cd-hero__meta">
                                    <span className="cd-hero__symbol">{coin.symbol?.toUpperCase()}</span>
                                    {coin.market_cap_rank && (
                                        <span className="cd-hero__rank">#{coin.market_cap_rank}</span>
                                    )}
                                    {coin.categories?.[0] && (
                                        <span className="cd-hero__category">{coin.categories[0]}</span>
                                    )}
                                </div>
                            </>
                        ) : (
                            <>
                                <Skeleton w={160} h={28} />
                                <Skeleton w={100} h={16} />
                            </>
                        )}
                    </div>
                </div>

                <div className="cd-hero__right">
                    {coin ? (
                        <>
                            <div className="cd-hero__price">{fmtUSD(price)}</div>
                            <div className={`cd-hero__change ${up24 ? "cd-up" : "cd-down"}`}>
                                {up24 ? "▲" : "▼"} {Math.abs(pct24h || 0).toFixed(2)}%
                                <span className="cd-hero__change-label"> 24h</span>
                            </div>
                            {isStale && <div className="cd-stale-badge">⚡ Cached</div>}
                        </>
                    ) : (
                        <>
                            <Skeleton w={180} h={36} />
                            <Skeleton w={80} h={20} />
                        </>
                    )}
                </div>
            </div>

            {/* ── Main layout: chart + stats ── */}
            <div className="cd-layout">

                {/* Left: chart */}
                <div className="cd-chart-col">
                    <div className="cd-card">
                        {/* Range selector */}
                        <div className="cd-range-bar">
                            <span className="cd-range-bar__label">Price Chart</span>
                            <div className="cd-range-bar__btns">
                                {RANGES.map(r => (
                                    <button
                                        key={r.days}
                                        className={`cd-range-btn ${days === r.days ? "cd-range-btn--active" : ""}`}
                                        onClick={() => setDays(r.days)}
                                    >
                                        {r.label}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Chart area */}
                        <div className="cd-chart-area">
                            {chartStatus === "loading" ? (
                                <div className="cd-chart-loading">
                                    <div className="cd-chart-loading__skeleton" />
                                </div>
                            ) : lineData ? (
                                <Line data={lineData} options={lineOptions} />
                            ) : (
                                <div className="cd-chart-empty">No chart data available</div>
                            )}
                        </div>

                        {/* Volume bar row */}
                        {chartData?.total_volumes && (
                            <div className="cd-vol-row">
                                <span className="cd-vol-row__label">24h Volume</span>
                                <span className="cd-vol-row__value">
                                    {fmtUSD(chartData.total_volumes.at(-1)?.[1])}
                                </span>
                            </div>
                        )}
                    </div>

                    {/* Performance grid */}
                    <div className="cd-card cd-perf-card">
                        <div className="cd-card__title">Price Performance</div>
                        <div className="cd-perf-grid">
                            {[
                                { label: "24h Low",  value: fmtUSD(md?.low_24h?.usd) },
                                { label: "24h High", value: fmtUSD(md?.high_24h?.usd) },
                                { label: "ATH",      value: fmtUSD(md?.ath?.usd) },
                                { label: "ATL",      value: fmtUSD(md?.atl?.usd) },
                            ].map(({ label, value }) => (
                                <div key={label} className="cd-perf-item">
                                    <span className="cd-perf-item__label">{label}</span>
                                    <span className="cd-perf-item__value">{coinStatus === "loading" ? <Skeleton w={80} h={14} /> : value}</span>
                                </div>
                            ))}
                        </div>

                        {/* Price change timeline */}
                        <div className="cd-card__title" style={{ marginTop: "20px" }}>Price Change</div>
                        <div className="cd-changes">
                            {[
                                { label: "24h",  pct: pct24h },
                                { label: "7d",   pct: pct7d },
                                { label: "30d",  pct: pct30d },
                                { label: "1y",   pct: pct1y },
                            ].map(({ label, pct }) => (
                                <div key={label} className="cd-change-item">
                                    <span className="cd-change-item__label">{label}</span>
                                    {coinStatus === "loading"
                                        ? <Skeleton w={60} h={22} r={20} />
                                        : (
                                            <span className={`cd-change-item__badge ${pct >= 0 ? "cd-up-badge" : "cd-down-badge"}`}>
                                                {pct >= 0 ? "▲" : "▼"} {Math.abs(pct || 0).toFixed(2)}%
                                            </span>
                                        )
                                    }
                                </div>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Right: stats sidebar */}
                <div className="cd-stats-col">

                    {/* Market stats */}
                    <div className="cd-card">
                        <div className="cd-card__title">Market Stats</div>
                        {coinStatus === "loading" ? (
                            Array.from({ length: 6 }).map((_, i) => (
                                <div key={i} className="cd-stat-row">
                                    <Skeleton w="45%" h={13} />
                                    <Skeleton w="40%" h={13} />
                                </div>
                            ))
                        ) : (
                            <>
                                <StatRow label="Market Cap"        value={fmtUSD(md?.market_cap?.usd)} />
                                <StatRow label="Fully Diluted Val" value={fmtUSD(md?.fully_diluted_valuation?.usd)} />
                                <StatRow label="24h Volume"        value={fmtUSD(md?.total_volume?.usd)} />
                                <StatRow label="Vol / Mkt Cap"
                                    value={md?.total_volume?.usd && md?.market_cap?.usd
                                        ? ((md.total_volume.usd / md.market_cap.usd) * 100).toFixed(2) + "%"
                                        : "—"}
                                />
                                <StatRow label="Circulating Supply"
                                    value={md?.circulating_supply
                                        ? fmtNum(md.circulating_supply) + " " + coin?.symbol?.toUpperCase()
                                        : "—"}
                                />
                                <StatRow label="Total Supply"
                                    value={md?.total_supply
                                        ? fmtNum(md.total_supply) + " " + coin?.symbol?.toUpperCase()
                                        : "∞"}
                                />
                                <StatRow label="Max Supply"
                                    value={md?.max_supply
                                        ? fmtNum(md.max_supply) + " " + coin?.symbol?.toUpperCase()
                                        : "∞"}
                                />
                            </>
                        )}
                    </div>

                    {/* Info card */}
                    <div className="cd-card">
                        <div className="cd-card__title">Info</div>
                        {coinStatus === "loading" ? (
                            Array.from({ length: 4 }).map((_, i) => (
                                <div key={i} className="cd-stat-row">
                                    <Skeleton w="35%" h={13} />
                                    <Skeleton w="50%" h={13} />
                                </div>
                            ))
                        ) : (
                            <>
                                {coin?.genesis_date && (
                                    <StatRow label="Genesis Date" value={coin.genesis_date} />
                                )}
                                <StatRow label="Hashing Algo"    value={coin?.hashing_algorithm || "—"} />
                                <StatRow label="Block Time"      value={coin?.block_time_in_minutes ? coin.block_time_in_minutes + " min" : "—"} />
                                <StatRow label="Community Score" value={coin?.community_score?.toFixed(1) ?? "—"} />
                                {coin?.links?.homepage?.[0] && (
                                    <div className="cd-stat-row">
                                        <span className="cd-stat-row__label">Website</span>
                                        <a
                                            href={coin.links.homepage[0]}
                                            target="_blank"
                                            rel="noreferrer"
                                            className="cd-stat-row__link"
                                        >
                                            {new URL(coin.links.homepage[0]).hostname} ↗
                                        </a>
                                    </div>
                                )}
                                {coin?.links?.blockchain_site?.filter(Boolean)?.[0] && (
                                    <div className="cd-stat-row">
                                        <span className="cd-stat-row__label">Explorer</span>
                                        <a
                                            href={coin.links.blockchain_site[0]}
                                            target="_blank"
                                            rel="noreferrer"
                                            className="cd-stat-row__link"
                                        >
                                            View ↗
                                        </a>
                                    </div>
                                )}
                            </>
                        )}
                    </div>

                </div>
            </div>

            {/* ── About — full-width below the two-column layout ── */}
            {(coinStatus === "ready" && coin?.description?.en) && (
                <div className="cd-card cd-about-card">
                    <div className="cd-about-header">
                        <div className="cd-card__title" style={{ margin: 0 }}>
                            About {coin.name}
                        </div>
                        <button
                            className="cd-about-toggle"
                            onClick={() => setAboutExpanded(e => !e)}
                        >
                            {aboutExpanded ? "Show less ▲" : "Read more ▼"}
                        </button>
                    </div>
                    <div className={`cd-about-body ${aboutExpanded ? "cd-about-body--open" : ""}`}>
                        <p
                            className="cd-about-text"
                            dangerouslySetInnerHTML={{ __html: coin.description.en }}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}
