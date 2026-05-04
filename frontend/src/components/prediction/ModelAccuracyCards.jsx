/**
 * components/prediction/ModelAccuracyCards.jsx
 *
 * Simplified: shows only the single best model by directional accuracy.
 * Displays: model name + directional accuracy percentage bar only.
 * Hidden: composite score, MAPE, MAPE rating, rank numbers, low performers.
 */

import React, { memo, useMemo } from "react";
import { MODELS } from "../../constants/prediction";

const HORIZON_ORDER = ["1w", "1m", "3m", "6m", "1y"];

function computeModelStats(key, metrics) {
    if (!metrics) return null;
    const horizons = HORIZON_ORDER.filter(h => metrics[h]);
    if (!horizons.length) return null;

    const dirAccValues = horizons.map(h => (metrics[h]?.dir_acc ?? 0) * 100);
    const avgDirAcc = dirAccValues.reduce((s, v) => s + v, 0) / dirAccValues.length;

    return { avgDirAcc };
}

// ── Main component ────────────────────────────────────────────────────────────

const ModelAccuracyCards = memo(({ validationMetrics }) => {
    const best = useMemo(() => {
        if (!validationMetrics || !Object.keys(validationMetrics).length) return null;

        return MODELS
            .map(model => {
                const stats = computeModelStats(model.key, validationMetrics?.[model.key]);
                return stats ? { model, stats } : null;
            })
            .filter(Boolean)
            .sort((a, b) => b.stats.avgDirAcc - a.stats.avgDirAcc)[0] ?? null;
    }, [validationMetrics]);

    if (!best) return null;

    const { model, stats } = best;
    const pct = Math.min(100, Math.max(0, stats.avgDirAcc));
    const barColor = pct >= 65 ? "#10b981" : pct >= 50 ? model.color : "#ef4444";

    return (
        <div className="pred-card mac-root">
            {/* Header */}
            <div className="pred-card-header">
                <h3 className="pred-card-title">📐 Model Accuracy</h3>
            </div>

            {/* Single best model */}
            <div className="mac-best-card">
                {/* Model name row */}
                <div className="mac-best-header">
                    <span className="mac-best-emoji">🥇</span>
                    <span className="mac-model-dot" style={{ background: model.color }} />
                    <span className="mac-model-name">{model.label}</span>
                    <span className="mac-badge mac-badge--best">★ Best Model</span>
                </div>

                {/* Directional accuracy bar */}
                <div className="mac-best-body">
                    <span className="mac-metric-label">Directional Accuracy</span>
                    <div className="mac-bar-wrap">
                        <div className="mac-bar-track">
                            <div className="mac-bar-midline" title="50% baseline" />
                            <div
                                className="mac-bar-fill"
                                style={{ width: `${pct}%`, background: barColor }}
                            />
                        </div>
                        <span className="mac-bar-label" style={{ color: barColor }}>
                            {pct.toFixed(1)}%
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
});

ModelAccuracyCards.displayName = "ModelAccuracyCards";
export default ModelAccuracyCards;