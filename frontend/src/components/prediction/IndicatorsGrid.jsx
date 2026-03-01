import React, { memo, useState } from "react";
import { fmtPrice } from "./utils";

const INDICATOR_GROUPS = [
    {
        label: "Trend",
        items: [
            { key: "sma_20",   label: "SMA 20",      fmt: "price" },
            { key: "sma_50",   label: "SMA 50",      fmt: "price" },
            { key: "sma_200",  label: "SMA 200",     fmt: "price" },
            { key: "ema_14",   label: "EMA 14",      fmt: "price" },
            { key: "ema_50",   label: "EMA 50",      fmt: "price" },
            { key: "adx_14",   label: "ADX 14",      fmt: "dec2"  },
        ],
    },
    {
        label: "Momentum",
        items: [
            { key: "rsi_14",   label: "RSI 14",      fmt: "rsi"   },
            { key: "macd",     label: "MACD",        fmt: "dec4"  },
            { key: "macd_signal", label: "MACD Sig", fmt: "dec4"  },
            { key: "stoch_k",  label: "Stoch %K",    fmt: "dec2"  },
            { key: "stoch_d",  label: "Stoch %D",    fmt: "dec2"  },
            { key: "cci",      label: "CCI",         fmt: "dec2"  },
            { key: "roc_10",   label: "ROC 10",      fmt: "dec4"  },
            { key: "williams_r", label: "Williams %R", fmt: "dec2" },
        ],
    },
    {
        label: "Volatility",
        items: [
            { key: "bb_upper", label: "BB Upper",    fmt: "price" },
            { key: "bb_lower", label: "BB Lower",    fmt: "price" },
            { key: "bb_width", label: "BB Width",    fmt: "dec4"  },
            { key: "atr_14",   label: "ATR 14",      fmt: "dec4"  },
            { key: "vol_7d",   label: "Vol 7d",      fmt: "dec4"  },
            { key: "vol_30d",  label: "Vol 30d",     fmt: "dec4"  },
        ],
    },
    {
        label: "Levels",
        items: [
            { key: "pivot",    label: "Pivot",       fmt: "price" },
            { key: "support_1",label: "Support 1",   fmt: "price" },
            { key: "resist_1", label: "Resist 1",    fmt: "price" },
            { key: "fib_382",  label: "Fib 38.2%",   fmt: "price" },
            { key: "fib_618",  label: "Fib 61.8%",   fmt: "price" },
            { key: "obv",      label: "OBV",         fmt: "large" },
        ],
    },
];

function fmt(value, type) {
    if (value == null || isNaN(value)) return "—";
    switch (type) {
        case "price": return fmtPrice(value);
        case "rsi":   return value.toFixed(2);
        case "dec2":  return value.toFixed(2);
        case "dec4":  return value.toFixed(4);
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

function rsiColor(v) {
    if (v >= 70) return "var(--red)";
    if (v <= 30) return "var(--green)";
    return "var(--accent)";
}

function RSIGauge({ value }) {
    const pct = Math.min(100, Math.max(0, value));
    const color = rsiColor(value);
    return (
        <div className="pred-rsi-gauge" title={`RSI: ${value?.toFixed(2)}`}>
            <div className="pred-rsi-track">
                <div className="pred-rsi-zone-ob" />
                <div className="pred-rsi-zone-os" />
                <div className="pred-rsi-needle" style={{ left: `${pct}%`, background: color }} />
            </div>
            <span className="pred-rsi-value" style={{ color }}>{value?.toFixed(2)}</span>
        </div>
    );
}

const IndicatorsGrid = memo(({ indicators }) => {
    const [activeGroup, setActiveGroup] = useState("Trend");
    if (!indicators || !Object.keys(indicators).length) return null;

    const group = INDICATOR_GROUPS.find(g => g.label === activeGroup);

    return (
        <div className="pred-card">
            <div className="pred-card-header">
                <h3 className="pred-card-title"> Technical Indicators</h3>
                <div className="pred-ind-tabs">
                    {INDICATOR_GROUPS.map(g => (
                        <button
                            key={g.label}
                            className={`pred-ind-tab${activeGroup === g.label ? " active" : ""}`}
                            onClick={() => setActiveGroup(g.label)}
                        >
                            {g.label}
                        </button>
                    ))}
                </div>
            </div>

            {activeGroup === "Momentum" && indicators.rsi_14 != null && (
                <div className="pred-rsi-row">
                    <span className="pred-rsi-label">RSI (14)</span>
                    <RSIGauge value={indicators.rsi_14} />
                    <span className="pred-rsi-signal">
                        {indicators.rsi_14 >= 70 ? "⚠️ Overbought" : indicators.rsi_14 <= 30 ? "🟢 Oversold" : "Neutral"}
                    </span>
                </div>
            )}

            <div className="pred-ind-grid">
                {group?.items.map(({ key, label, fmt: fmtType }) => {
                    const v = indicators[key];
                    if (v == null) return null;
                    const color = key === "rsi_14" ? rsiColor(v) : undefined;
                    return (
                        <div key={key} className="pred-ind-tile">
                            <span className="pred-ind-label">{label}</span>
                            <span className="pred-ind-value" style={color ? { color } : {}}>
                                {fmt(v, fmtType)}
                            </span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
});

IndicatorsGrid.displayName = "IndicatorsGrid";
export default IndicatorsGrid;
