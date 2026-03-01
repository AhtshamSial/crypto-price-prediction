import React from "react";

export default function WarmUpBanner({ trainedCoins }) {
    const total = 5;
    const done  = trainedCoins.length;
    const pct   = Math.round((done / total) * 100);
    return (
        <div className="pred-warmup">
            <div className="pred-warmup-gears">⚙️</div>
            <h3 className="pred-warmup-title">Training AI Models…</h3>
            <p className="pred-warmup-desc">
                Building ensemble models for {total} coins. First run takes 2–5 minutes. Results are cached after that.
            </p>
            <div className="pred-warmup-progress">
                <div className="pred-warmup-bar">
                    <div className="pred-warmup-fill" style={{ width: `${pct}%` }} />
                </div>
                <span className="pred-warmup-stat">{done} / {total} coins ready</span>
            </div>
            {done > 0 && (
                <div className="pred-warmup-coins">
                    {trainedCoins.map(c => (
                        <span key={c} className="pred-warmup-coin">✓ {c}</span>
                    ))}
                </div>
            )}
        </div>
    );
}
