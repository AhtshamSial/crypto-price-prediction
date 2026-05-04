/**
 * components/prediction/IndicatorsGrid.jsx
 *
 * Redesigned Technical Indicators component:
 *  - Each group tab shows a count badge of available indicators
 *  - Group icon added per category (Trend / Momentum / Volatility / Levels)
 *  - RSI section upgraded: larger gauge, zone labels, dynamic signal pill
 *  - MACD section shows histogram-style bar for MACD vs Signal divergence
 *  - Bollinger Band section shows a visual price-position-in-band widget
 *  - All tiles now show a colored left-border accent matching indicator theme
 *  - Signal pills (Bullish / Bearish / Neutral) inferred for key indicators
 *  - Responsive: 2-col on mobile, 3-col on tablet, 4-col on desktop
 */

import React, { memo, useState, useMemo } from "react";
import { fmtPrice } from "./utils";

// ── Config ────────────────────────────────────────────────────────────────────

const GROUPS = [
    {
        key: "trend",
        label: "Trend",
        icon: "📈",
        items: [
            { key: "sma_20", label: "SMA 20", fmt: "price", desc: "20-day simple moving average" },
            { key: "sma_50", label: "SMA 50", fmt: "price", desc: "50-day simple moving average" },
            { key: "sma_200", label: "SMA 200", fmt: "price", desc: "200-day simple moving average" },
            { key: "ema_14", label: "EMA 14", fmt: "price", desc: "14-day exponential moving average" },
            { key: "ema_50", label: "EMA 50", fmt: "price", desc: "50-day exponential moving average" },
            { key: "adx_14", label: "ADX 14", fmt: "dec2", desc: "Average directional index — trend strength" },
        ],
    },
    {
        key: "momentum",
        label: "Momentum",
        icon: "⚡",
        items: [
            { key: "rsi_14", label: "RSI 14", fmt: "rsi", desc: "Relative strength index (overbought >70, oversold <30)" },
            { key: "macd", label: "MACD", fmt: "dec4", desc: "MACD line" },
            { key: "macd_signal", label: "MACD Signal", fmt: "dec4", desc: "MACD signal line" },
            { key: "stoch_k", label: "Stoch %K", fmt: "dec2", desc: "Stochastic oscillator %K" },
            { key: "stoch_d", label: "Stoch %D", fmt: "dec2", desc: "Stochastic oscillator %D" },
            { key: "cci", label: "CCI", fmt: "dec2", desc: "Commodity channel index" },
            { key: "roc_10", label: "ROC 10", fmt: "dec4", desc: "Rate of change (10-period)" },
            { key: "williams_r", label: "Williams %R", fmt: "dec2", desc: "Williams %R oscillator" },
        ],
    },
    {
        key: "volatility",
        label: "Volatility",
        icon: "🌊",
        items: [
            { key: "bb_upper", label: "BB Upper", fmt: "price", desc: "Bollinger Band upper band" },
            { key: "bb_lower", label: "BB Lower", fmt: "price", desc: "Bollinger Band lower band" },
            { key: "bb_width", label: "BB Width", fmt: "dec4", desc: "Bollinger Band width (volatility)" },
            { key: "atr_14", label: "ATR 14", fmt: "dec4", desc: "Average true range (14-period)" },
            { key: "vol_7d", label: "Vol 7d", fmt: "dec4", desc: "7-day historical volatility" },
            { key: "vol_30d", label: "Vol 30d", fmt: "dec4", desc: "30-day historical volatility" },
        ],
    },
    {
        key: "levels",
        label: "Levels",
        icon: "🎯",
        items: [
            { key: "pivot", label: "Pivot", fmt: "price", desc: "Pivot point" },
            { key: "support_1", label: "Support 1", fmt: "price", desc: "First support level" },
            { key: "resist_1", label: "Resist 1", fmt: "price", desc: "First resistance level" },
            { key: "fib_382", label: "Fib 38.2%", fmt: "price", desc: "Fibonacci 38.2% retracement" },
            { key: "fib_618", label: "Fib 61.8%", fmt: "price", desc: "Fibonacci 61.8% retracement" },
            { key: "obv", label: "OBV", fmt: "large", desc: "On-balance volume" },
        ],
    },
];

// ── Formatters ────────────────────────────────────────────────────────────────

function fmtValue(value, type) {
    if (value == null || isNaN(value)) return "—";
    switch (type) {
        case "price": return fmtPrice(value);
        case "rsi": return value.toFixed(2);
        case "dec2": return value.toFixed(2);
        case "dec4": return value.toFixed(4);
        case "large": {
            const abs = Math.abs(value);
            if (abs >= 1e9) return (value / 1e9).toFixed(2) + "B";
            if (abs >= 1e6) return (value / 1e6).toFixed(2) + "M";
            if (abs >= 1e3) return (value / 1e3).toFixed(1) + "K";
            return value.toFixed(0);
        }
        default: return value.toFixed(4);
    }
}

// ── Signal helpers ────────────────────────────────────────────────────────────

function getSignal(key, value, indicators) {
    if (value == null) return null;
    switch (key) {
        case "rsi_14":
            return value >= 70 ? { label: "Overbought", type: "bearish" }
                : value <= 30 ? { label: "Oversold", type: "bullish" }
                    : { label: "Neutral", type: "neutral" };
        case "macd": {
            const sig = indicators?.macd_signal;
            if (sig == null) return null;
            return value > sig ? { label: "Bullish", type: "bullish" }
                : { label: "Bearish", type: "bearish" };
        }
        case "stoch_k":
            return value >= 80 ? { label: "Overbought", type: "bearish" }
                : value <= 20 ? { label: "Oversold", type: "bullish" }
                    : { label: "Neutral", type: "neutral" };
        case "cci":
            return value > 100 ? { label: "Overbought", type: "bearish" }
                : value < -100 ? { label: "Oversold", type: "bullish" }
                    : { label: "Neutral", type: "neutral" };
        case "williams_r":
            return value >= -20 ? { label: "Overbought", type: "bearish" }
                : value <= -80 ? { label: "Oversold", type: "bullish" }
                    : { label: "Neutral", type: "neutral" };
        case "adx_14":
            return value >= 25 ? { label: "Trending", type: "bullish" }
                : { label: "Ranging", type: "neutral" };
        default: return null;
    }
}

// ── Special widgets ───────────────────────────────────────────────────────────

function RSIWidget({ value }) {
    if (value == null) return null;
    const pct = Math.min(100, Math.max(0, value));
    const color = value >= 70 ? "#ef4444" : value <= 30 ? "#10b981" : "#667eea";
    const label = value >= 70 ? "Overbought" : value <= 30 ? "Oversold" : "Neutral";

    return (
        <div className="ind-rsi-widget">
            <div className="ind-rsi-header">
                <span className="ind-rsi-title">RSI (14)</span>
                <span className="ind-rsi-val" style={{ color }}>{value.toFixed(2)}</span>
                <span className={`ind-signal-pill ind-signal-pill--${value >= 70 ? "bearish" : value <= 30 ? "bullish" : "neutral"}`}>
                    {label}
                </span>
            </div>

            <div className="ind-rsi-track-wrap">
                {/* Zone labels */}
                <div className="ind-rsi-zones">
                    <span className="ind-rsi-zone-label ind-rsi-zone-label--left">Oversold (30)</span>
                    <span className="ind-rsi-zone-label ind-rsi-zone-label--right">Overbought (70)</span>
                </div>
                {/* Track */}
                <div className="ind-rsi-track">
                    <div className="ind-rsi-zone ind-rsi-zone--os" />
                    <div className="ind-rsi-zone ind-rsi-zone--ob" />
                    <div className="ind-rsi-fill" style={{ width: `${pct}%`, background: color }} />
                    <div className="ind-rsi-needle" style={{ left: `${pct}%`, borderColor: color }}>
                        <div className="ind-rsi-needle-tip" style={{ background: color }} />
                    </div>
                </div>
                {/* Scale */}
                <div className="ind-rsi-scale">
                    {[0, 25, 50, 75, 100].map(v => (
                        <span key={v} className="ind-rsi-scale-tick">{v}</span>
                    ))}
                </div>
            </div>
        </div>
    );
}

function MACDWidget({ macd, signal }) {
    if (macd == null || signal == null) return null;
    const diff = macd - signal;
    const isBullish = diff > 0;
    const maxAbs = Math.max(Math.abs(macd), Math.abs(signal), 0.0001);
    const macdPct = (macd / maxAbs) * 50 + 50;
    const sigPct = (signal / maxAbs) * 50 + 50;
    const diffColor = isBullish ? "#10b981" : "#ef4444";

    return (
        <div className="ind-macd-widget">
            <div className="ind-macd-header">
                <span className="ind-rsi-title">MACD</span>
                <span className={`ind-signal-pill ind-signal-pill--${isBullish ? "bullish" : "bearish"}`}>
                    {isBullish ? "Bullish crossover" : "Bearish crossover"}
                </span>
            </div>
            <div className="ind-macd-bars">
                <div className="ind-macd-row">
                    <span className="ind-macd-label">MACD</span>
                    <div className="ind-macd-track">
                        <div className="ind-macd-center" />
                        <div className="ind-macd-bar" style={{
                            left: macd >= 0 ? "50%" : `${macdPct}%`,
                            width: `${Math.abs(macdPct - 50)}%`,
                            background: macd >= 0 ? "#10b981" : "#ef4444",
                        }} />
                    </div>
                    <span className="ind-macd-val" style={{ color: macd >= 0 ? "#10b981" : "#ef4444" }}>
                        {macd.toFixed(4)}
                    </span>
                </div>
                <div className="ind-macd-row">
                    <span className="ind-macd-label">Signal</span>
                    <div className="ind-macd-track">
                        <div className="ind-macd-center" />
                        <div className="ind-macd-bar" style={{
                            left: signal >= 0 ? "50%" : `${sigPct}%`,
                            width: `${Math.abs(sigPct - 50)}%`,
                            background: signal >= 0 ? "#10b981" : "#ef4444",
                            opacity: 0.6,
                        }} />
                    </div>
                    <span className="ind-macd-val">{signal.toFixed(4)}</span>
                </div>
                <div className="ind-macd-row">
                    <span className="ind-macd-label">Histogram</span>
                    <div className="ind-macd-track">
                        <div className="ind-macd-center" />
                        <div className="ind-macd-bar" style={{
                            left: isBullish ? "50%" : `${((diff / maxAbs) * 50 + 50)}%`,
                            width: `${Math.abs((diff / maxAbs) * 50)}%`,
                            background: diffColor,
                        }} />
                    </div>
                    <span className="ind-macd-val" style={{ color: diffColor }}>
                        {diff >= 0 ? "+" : ""}{diff.toFixed(4)}
                    </span>
                </div>
            </div>
        </div>
    );
}

function BBWidget({ upper, lower, price }) {
    if (upper == null || lower == null) return null;
    const range = upper - lower;
    const mid = lower + range / 2;
    const posPct = price != null && range > 0
        ? Math.min(100, Math.max(0, ((price - lower) / range) * 100))
        : null;
    const zone = posPct != null
        ? posPct >= 80 ? "Near upper band (overbought)"
            : posPct <= 20 ? "Near lower band (oversold)"
                : "Within bands (neutral)"
        : null;

    return (
        <div className="ind-bb-widget">
            <div className="ind-macd-header">
                <span className="ind-rsi-title">Bollinger Bands</span>
                {zone && (
                    <span className={`ind-signal-pill ind-signal-pill--${posPct >= 80 ? "bearish" : posPct <= 20 ? "bullish" : "neutral"}`}>
                        {zone}
                    </span>
                )}
            </div>
            <div className="ind-bb-track-wrap">
                <div className="ind-bb-labels">
                    <span>Lower: {fmtPrice(lower)}</span>
                    <span>Mid: {fmtPrice(mid)}</span>
                    <span>Upper: {fmtPrice(upper)}</span>
                </div>
                <div className="ind-bb-track">
                    <div className="ind-bb-fill" />
                    {posPct != null && (
                        <div
                            className="ind-bb-price-marker"
                            style={{ left: `${posPct}%` }}
                            title={`Price position: ${posPct.toFixed(0)}% within band`}
                        >
                            <div className="ind-bb-price-line" />
                            <span className="ind-bb-price-label">Price</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

// ── Tile ──────────────────────────────────────────────────────────────────────

function IndicatorTile({ item, value, indicators }) {
    const signal = getSignal(item.key, value, indicators);
    const display = fmtValue(value, item.fmt);
    const accent = signal?.type === "bullish" ? "#10b981"
        : signal?.type === "bearish" ? "#ef4444"
            : "#667eea";

    return (
        <div className="ind-tile" style={{ "--tile-accent": accent }} title={item.desc}>
            <div className="ind-tile-accent-bar" />
            <span className="ind-tile-label">{item.label}</span>
            <span className="ind-tile-value">{display}</span>
            {signal && (
                <span className={`ind-signal-pill ind-signal-pill--${signal.type}`}>
                    {signal.label}
                </span>
            )}
        </div>
    );
}

// ── Main component ────────────────────────────────────────────────────────────

const IndicatorsGrid = memo(({ indicators }) => {
    const [activeKey, setActiveKey] = useState("trend");

    const counts = useMemo(() => {
        const map = {};
        for (const g of GROUPS) {
            map[g.key] = g.items.filter(i => indicators?.[i.key] != null).length;
        }
        return map;
    }, [indicators]);

    if (!indicators || !Object.keys(indicators).length) return null;

    const group = GROUPS.find(g => g.key === activeKey);
    const isMomentum = activeKey === "momentum";
    const isVolatility = activeKey === "volatility";

    return (
        <div className="pred-card ind-root">
            {/* ── Header ── */}
            <div className="pred-card-header ind-header">
                <div>
                    <h3 className="pred-card-title">Technical Indicators</h3>
                    <p className="pred-card-hint">
                        {Object.values(counts).reduce((a, b) => a + b, 0)} indicators across {GROUPS.length} categories
                    </p>
                </div>
            </div>

            {/* ── Group tabs ── */}
            <div className="ind-tabs">
                {GROUPS.map(g => (
                    <button
                        key={g.key}
                        className={`ind-tab ${activeKey === g.key ? "ind-tab--active" : ""}`}
                        onClick={() => setActiveKey(g.key)}
                    >
                        <span className="ind-tab-icon">{g.icon}</span>
                        <span className="ind-tab-label">{g.label}</span>
                        {counts[g.key] > 0 && (
                            <span className="ind-tab-count">{counts[g.key]}</span>
                        )}
                    </button>
                ))}
            </div>

            {/* ── Special widgets for Momentum ── */}
            {isMomentum && (
                <div className="ind-widgets">
                    <RSIWidget value={indicators.rsi_14} />
                    <MACDWidget macd={indicators.macd} signal={indicators.macd_signal} />
                </div>
            )}

            {/* ── Bollinger Band widget for Volatility ── */}
            {isVolatility && (
                <div className="ind-widgets">
                    <BBWidget
                        upper={indicators.bb_upper}
                        lower={indicators.bb_lower}
                        price={indicators.current_price}
                    />
                </div>
            )}

            {/* ── Tiles grid ── */}
            <div className="ind-grid">
                {group?.items.map(item => {
                    const v = indicators[item.key];
                    if (v == null) return null;
                    return (
                        <IndicatorTile
                            key={item.key}
                            item={item}
                            value={v}
                            indicators={indicators}
                        />
                    );
                })}
            </div>
        </div>
    );
});

IndicatorsGrid.displayName = "IndicatorsGrid";
export default IndicatorsGrid;
