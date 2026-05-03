/**
 * components/prediction/ModelAccuracyCards.jsx
 *
 * Redesigned Model Accuracy component:
 *  - Filters to top performers only (avgDirAcc >= HIGH_THRESHOLD)
 *  - Ranks all models by composite score (60% dir-acc + 40% MAPE quality)
 *  - Best model gets a gold crown + glowing card border
 *  - Per-horizon sparkline bars show short vs long-term accuracy at a glance
 *  - Low performers collapsed by default — visible but not cluttering the view
 *  - Composite score drives sort so a slightly lower dir-acc but much better
 *    MAPE model can still outrank a pure dir-acc winner
 */

import React, { memo, useState, useMemo } from "react";
import { MODELS } from "../../constants/prediction";

const HORIZON_ORDER = ["1w", "1m", "3m", "6m", "1y"];
const HIGH_THRESHOLD = 50; // avgDirAcc < this → low performer

// ── Stat helpers ──────────────────────────────────────────────────────────────

function computeModelStats(key, metrics) {
    if (!metrics) return null;
    const horizons = HORIZON_ORDER.filter(h => metrics[h]);
    if (!horizons.length) return null;

    const dirAccValues = horizons.map(h => (metrics[h]?.dir_acc ?? 0) * 100);
    const mapeValues = horizons.map(h => metrics[h]?.mape ?? null).filter(v => v != null);

    const avgDirAcc = dirAccValues.reduce((s, v) => s + v, 0) / dirAccValues.length;
    const avgMape = mapeValues.length
        ? mapeValues.reduce((s, v) => s + v, 0) / mapeValues.length
        : null;

    // 0–100 quality score: clamp MAPE at 20% → 0 score
    const mapeScore = avgMape != null ? Math.max(0, 100 - (avgMape / 20) * 100) : 50;
    const composite = avgDirAcc * 0.6 + mapeScore * 0.4;

    return { avgDirAcc, avgMape, composite, horizons, dirAccValues };
}

function getBarColor(pct, fallback) {
    if (pct >= 65) return "#10b981";
    if (pct >= 50) return fallback;
    return "#ef4444";
}

// ── Sub-components ────────────────────────────────────────────────────────────

function AccuracyBar({ pct, color }) {
    const clamped = Math.min(100, Math.max(0, pct));
    const barColor = getBarColor(pct, color);
    return (
        <div className="mac-bar-wrap">
            <div className="mac-bar-track">
                <div className="mac-bar-midline" title="50% baseline" />
                <div className="mac-bar-fill" style={{ width: `${clamped}%`, background: barColor }} />
            </div>
            <span className="mac-bar-label" style={{ color: barColor }}>
                {pct.toFixed(1)}%
            </span>
        </div>
    );
}

function HorizonSparkline({ horizons, values, color }) {
    const maxVal = Math.max(...values.filter(Boolean), 60);
    return (
        <div className="mac-sparkline">
            {HORIZON_ORDER.map(h => {
                const idx = horizons.indexOf(h);
                const val = idx >= 0 ? values[idx] : null;
                const barH = val != null ? `${Math.round((val / maxVal) * 100)}%` : "12%";
                const bg = val == null ? "var(--border)"
                    : val >= 65 ? "#10b981"
                        : val >= 50 ? color
                            : "#ef4444";
                return (
                    <div key={h} className="mac-spark-col">
                        <div className="mac-spark-bar-wrap">
                            <div
                                className="mac-spark-bar"
                                style={{ height: barH, background: bg, opacity: val == null ? 0.25 : 1 }}
                                title={val != null ? `${h}: ${val.toFixed(1)}%` : `${h}: no data`}
                            />
                        </div>
                        <span className="mac-spark-label">{h}</span>
                    </div>
                );
            })}
        </div>
    );
}

const RANK_META = [
    { emoji: "🥇", label: "Best", glow: "rgba(245,158,11,0.35)" },
    { emoji: "🥈", label: "2nd", glow: "rgba(148,163,184,0.25)" },
    { emoji: "🥉", label: "3rd", glow: "rgba(205,124,58,0.25)" },
];

function ModelCard({ model, stats, rank, isTopPerformer }) {
    const meta = rank <= 3 ? RANK_META[rank - 1] : null;

    return (
        <div
            className={[
                "mac-card",
                rank === 1 ? "mac-card--best" : "",
                !isTopPerformer ? "mac-card--low" : "",
            ].join(" ").trim()}
            style={{
                "--model-color": model.color,
                ...(meta ? { "--card-glow": meta.glow } : {}),
            }}
        >
            {/* ── Header row ── */}
            <div className="mac-card-header">
                <div className="mac-rank-badge">
                    {meta ? (
                        <span className="mac-rank-emoji">{meta.emoji}</span>
                    ) : (
                        <span className="mac-rank-num">#{rank}</span>
                    )}
                </div>

                <span className="mac-model-dot" style={{ background: model.color }} />
                <span className="mac-model-name">{model.label}</span>

                <div className="mac-badges">
                    {rank === 1 && <span className="mac-badge mac-badge--best">★ Best Model</span>}
                    {!isTopPerformer && <span className="mac-badge mac-badge--low">Low</span>}
                </div>

                <div className="mac-score-chip" title="Composite score: 60% dir-accuracy + 40% MAPE quality">
                    <span className="mac-score-val">{stats.composite.toFixed(0)}</span>
                    <span className="mac-score-unit">/ 100</span>
                </div>
            </div>

            {/* ── Metrics + sparkline ── */}
            <div className="mac-card-body">
                <div className="mac-metrics">
                    <div className="mac-metric-row">
                        <span className="mac-metric-label">Directional Accuracy</span>
                        <AccuracyBar pct={stats.avgDirAcc} color={model.color} />
                    </div>
                    {stats.avgMape != null && (
                        <div className="mac-metric-row">
                            <span className="mac-metric-label">Avg MAPE</span>
                            <div className="mac-mape-row">
                                <span
                                    className="mac-mape-val"
                                    style={{ color: stats.avgMape < 5 ? "#10b981" : stats.avgMape < 15 ? "var(--text-primary)" : "#ef4444" }}
                                >
                                    {stats.avgMape.toFixed(2)}%
                                </span>
                                <span className={`mac-mape-pill ${stats.avgMape < 5 ? "mac-mape-pill--excellent" :
                                        stats.avgMape < 10 ? "mac-mape-pill--good" :
                                            stats.avgMape < 20 ? "mac-mape-pill--fair" :
                                                "mac-mape-pill--poor"
                                    }`}>
                                    {stats.avgMape < 5 ? "Excellent" : stats.avgMape < 10 ? "Good" : stats.avgMape < 20 ? "Fair" : "Poor"}
                                </span>
                            </div>
                        </div>
                    )}
                </div>

                <div className="mac-sparkline-col">
                    <span className="mac-metric-label">Dir Acc by Horizon</span>
                    <HorizonSparkline
                        horizons={stats.horizons}
                        values={stats.dirAccValues}
                        color={model.color}
                    />
                </div>
            </div>
        </div>
    );
}

// ── Main component ────────────────────────────────────────────────────────────

const ModelAccuracyCards = memo(({ validationMetrics }) => {
    const [showLow, setShowLow] = useState(false);

    const { topModels, lowModels, bestAccuracy } = useMemo(() => {
        const ranked = MODELS
            .map(model => {
                const stats = computeModelStats(model.key, validationMetrics?.[model.key]);
                return stats ? { model, stats } : null;
            })
            .filter(Boolean)
            .sort((a, b) => b.stats.composite - a.stats.composite)
            .map((item, i) => ({ ...item, rank: i + 1 }));

        return {
            topModels: ranked.filter(m => m.stats.avgDirAcc >= HIGH_THRESHOLD),
            lowModels: ranked.filter(m => m.stats.avgDirAcc < HIGH_THRESHOLD),
            bestAccuracy: ranked[0]?.stats.avgDirAcc ?? null,
        };
    }, [validationMetrics]);

    if (!validationMetrics || !Object.keys(validationMetrics).length) return null;
    if (!topModels.length && !lowModels.length) return null;

    return (
        <div className="pred-card mac-root">
            {/* Header */}
            <div className="pred-card-header mac-header">
                <div>
                    <h3 className="pred-card-title">📐 Model Accuracy</h3>
                    <p className="pred-card-hint">
                        {topModels.length} top performer{topModels.length !== 1 ? "s" : ""}
                        {bestAccuracy != null && ` · Best: ${bestAccuracy.toFixed(1)}% dir acc`}
                    </p>
                </div>
                <div className="mac-legend">
                    <span className="mac-legend-dot" style={{ background: "#10b981" }} />
                    <span className="mac-legend-text">≥65% Excellent</span>
                    <span className="mac-legend-dot" style={{ background: "#667eea" }} />
                    <span className="mac-legend-text">≥50% Good</span>
                    <span className="mac-legend-dot" style={{ background: "#ef4444" }} />
                    <span className="mac-legend-text">&lt;50% Low</span>
                </div>
            </div>

            {/* Top performers */}
            {topModels.length > 0 ? (
                <div className="mac-grid">
                    {topModels.map(({ model, stats, rank }) => (
                        <ModelCard key={model.key} model={model} stats={stats} rank={rank} isTopPerformer />
                    ))}
                </div>
            ) : (
                <div className="mac-empty">
                    <span className="mac-empty-icon">⚠️</span>
                    <p>No models met the {HIGH_THRESHOLD}% accuracy threshold on this validation set.</p>
                </div>
            )}

            {/* Collapsible low performers */}
            {lowModels.length > 0 && (
                <div className="mac-low-section">
                    <button className="mac-low-toggle" onClick={() => setShowLow(p => !p)}>
                        <span className={`mac-low-chevron ${showLow ? "mac-low-chevron--open" : ""}`}>›</span>
                        {showLow ? "Hide" : "Show"} {lowModels.length} low-performing model{lowModels.length !== 1 ? "s" : ""}
                    </button>
                    {showLow && (
                        <div className="mac-grid mac-grid--low">
                            {lowModels.map(({ model, stats, rank }) => (
                                <ModelCard key={model.key} model={model} stats={stats} rank={rank} isTopPerformer={false} />
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
});

ModelAccuracyCards.displayName = "ModelAccuracyCards";
export default ModelAccuracyCards;
