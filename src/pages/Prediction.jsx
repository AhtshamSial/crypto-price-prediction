import React, { useState } from "react";
import { Line, Bar, Pie } from "react-chartjs-2";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    Tooltip,
    Legend,
} from "chart.js";
import api from "../services/api"; // axios instance: baseURL -> your backend

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    Tooltip,
    Legend
);

export default function Prediction() {
    const [symbol, setSymbol] = useState("BTC/USDT");
    const [investment, setInvestment] = useState(100);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    // Demo fallback when backend unavailable
    const demoFallback = (sym, invest) => {
        const price = sym.includes("BTC") ? 64000 : sym.includes("ETH") ? 3400 : 520;
        const confidence = sym.includes("BTC") ? 0.82 : sym.includes("ETH") ? 0.78 : 0.72;
        const sentiment = 0.25;
        const atr = sym.includes("BTC") ? 150 : sym.includes("ETH") ? 8 : 3;
        const direction = confidence > 0.75 ? "LONG" : "SHORT";
        const now = new Date().toISOString();
        return {
            symbol: sym,
            price,
            direction,
            confidence,
            sentiment,
            atr,
            stop_loss: direction === "LONG" ? price - 1.5 * atr : price + 1.5 * atr,
            take_profit: direction === "LONG" ? price + 3 * atr : price - 3 * atr,
            leverage: 5,
            timestamp: now,
            demo: true,
        };
    };

    // Build a short synthetic trend (10 points) around price using ATR
    const buildTrendFromResult = (price, atr, points = 10) => {
        if (!price || price === null) return null;
        const trend = [];
        for (let i = points - 1; i >= 0; i--) {
            // Create slightly decaying / trending values: last point = price
            const noise = (Math.random() - 0.5) * atr * 0.4;
            const value = price * (1 - 0.005 * i) + noise;
            trend.push(Math.round(value * 100) / 100);
        }
        return trend;
    };

    const handlePredict = async () => {
        setLoading(true);
        setResult(null);
        try {
            const res = await api.post("/predict", { symbol, investment });

            // Handle backend error case
            if (res.data.error) {
                throw new Error(res.data.message || "Prediction engine error");
            }

            // If backend sends valid data
            if (!res.data.price) {
                throw new Error("Incomplete data from AI model");
            }

            setResult(res.data);
        } catch (err) {
            console.error("Prediction API error:", err);
            alert(`Prediction failed: ${err.message || "Server error. Please try again later."}`);
            // Optionally comment next line if you don’t want fallback at all
            // const fallback = demoFallback(symbol, investment);
            // fallback.note = "api_error_fallback";
            // setResult(fallback);
        } finally {
            setLoading(false);
        }
    };

    // Chart data generators (use result when available)
    const trendPrices = result ? buildTrendFromResult(result.price, result.atr, 10) : null;
    const trendData = {
        labels: trendPrices ? trendPrices.map((_, i) => `T${i + 1}`) : [],
        datasets: [
            {
                label: `${symbol} Predicted Trend`,
                data: trendPrices || [],
                borderColor: "#0d6efd",
                backgroundColor: "rgba(13,110,253,0.15)",
                fill: true,
                tension: 0.35,
                pointRadius: 0,
            },
        ],
    };

    // Bar chart: AI Confidence comparison for top coins.
    // Use result.confidence for selected symbol; mock others with small variation.
    const topCoins = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"];
    const barData = {
        labels: topCoins,
        datasets: [
            {
                label: "AI Confidence (%)",
                data: topCoins.map((c) => {
                    if (!result) return 75 + Math.floor(Math.random() * 10);
                    if (c === result.symbol) return Math.round(result.confidence * 100);
                    // mock based on selected result.confidence with small offset
                    const base = result.confidence * 100;
                    const offset = Math.floor((Math.random() - 0.5) * 10);
                    return Math.max(40, Math.min(99, Math.round(base + offset)));
                }),
                backgroundColor: topCoins.map((c) =>
                    c === result?.symbol ? "rgba(13,110,253,0.9)" : "rgba(13,110,253,0.5)"
                ),
            },
        ],
    };

    // Pie chart: sentiment distribution from result.sentiment (-1..1) -> convert to Positive/Neutral/Negative
    // We'll map sentiment score: >0.2 positive, <-0.2 negative, else neutral, then produce percentages
    let sentimentPie = { labels: ["Positive", "Neutral", "Negative"], datasets: [{ data: [60, 25, 15], backgroundColor: ["#198754", "#ffc107", "#dc3545"] }] };
    if (result) {
        const s = result.sentiment ?? 0;
        // Basic mapping to percentages
        const positive = s > 0 ? Math.min(90, 50 + Math.round(40 * s)) : Math.max(5, 45 + Math.round(10 * s));
        const negative = s < 0 ? Math.min(90, 50 + Math.round(-40 * s)) : Math.max(5, 5);
        // let neutral absorb remaining
        let neutral = 100 - positive - negative;
        if (neutral < 0) neutral = 0;
        sentimentPie = {
            labels: ["Positive", "Neutral", "Negative"],
            datasets: [
                {
                    data: [positive, neutral, negative],
                    backgroundColor: ["#198754", "#ffc107", "#dc3545"],
                },
            ],
        };
    }

    const commonOptions = {
        responsive: true,
        plugins: { legend: { position: "top" } },
    };

    return (
        <div className="container my-5">
            <div className="d-flex justify-content-between align-items-center mb-3">
                <div>
                    <h2 className="fw-bold mb-0">AI Market Prediction</h2>
                    <small className="text-muted">Real-time AI-backed prediction</small>
                </div>
                <div>
                    <button
                        className="btn btn-outline-secondary"
                        onClick={() => {
                            setSymbol("BTC/USDT");
                            setResult(null);
                        }}
                    >
                        Clear
                    </button>
                </div>
            </div>

            {/* Input Card */}
            <div className="card shadow-sm mb-4 p-4">
                <div className="row g-3 align-items-end">
                    <div className="col-md-5">
                        <label className="form-label fw-semibold">Symbol</label>
                        <select className="form-select" value={symbol} onChange={(e) => setSymbol(e.target.value)}>
                            <option>BTC/USDT</option>
                            <option>ETH/USDT</option>
                            <option>BNB/USDT</option>
                            <option>SOL/USDT</option>
                        </select>
                    </div>

                    <div className="col-md-5">
                        <label className="form-label fw-semibold">Investment ($)</label>
                        <input
                            type="number"
                            className="form-control"
                            value={investment}
                            onChange={(e) => setInvestment(e.target.value)}
                            min={1}
                        />
                    </div>

                    <div className="col-md-2 text-center">
                        <button className="btn btn-secondary w-100" onClick={handlePredict} disabled={loading}>
                            {loading ? "Predicting..." : "Predict"}
                        </button>
                    </div>
                </div>
            </div>

            {/* Result Area */}
            {result ? (
                <div className="card shadow-sm p-4 mb-4">
                    <div className="row gy-3">
                        <div className="col-md-4">
                            <div className="card border-primary h-100">
                                <div className="card-body text-center">
                                    <h6 className="text-muted">Symbol</h6>
                                    <h4 className="text-primary">{result.symbol}</h4>
                                    <small className="text-muted d-block mt-2">{new Date(result.timestamp).toLocaleString()}</small>
                                </div>
                            </div>
                        </div>

                        <div className="col-md-4">
                            <div className="card border-success h-100">
                                <div className="card-body text-center">
                                    <h6 className="text-muted">Direction</h6>
                                    <h4 className={result.direction === "LONG" ? "text-success" : "text-danger"}>
                                        {result.direction === "LONG" ? "Up / LONG" : result.direction === "SHORT" ? "Down / SHORT" : result.direction}
                                    </h4>
                                </div>
                            </div>
                        </div>

                        <div className="col-md-4">
                            <div className="card border-warning h-100">
                                <div className="card-body text-center">
                                    <h6 className="text-muted">Confidence</h6>
                                    <h4 className="text-warning">{result.confidence != null ? `${(result.confidence * 100).toFixed(2)}%` : "N/A"}</h4>
                                </div>
                            </div>
                        </div>

                        <div className="col-md-4 mt-3">
                            <div className="card border-info h-100">
                                <div className="card-body text-center">
                                    <h6 className="text-muted">Sentiment</h6>
                                    <h4 className="text-info">{result.sentiment != null ? result.sentiment.toFixed(2) : "0.00"}</h4>
                                </div>
                            </div>
                        </div>

                        <div className="col-md-4 mt-3">
                            <div className="card border-danger h-100">
                                <div className="card-body text-center">
                                    <h6 className="text-muted">ATR</h6>
                                    <h4 className="text-danger">{result.atr != null ? result.atr.toFixed(2) : "N/A"}</h4>
                                </div>
                            </div>
                        </div>

                        <div className="col-md-4 mt-3">
                            <div className="card border-secondary h-100">
                                <div className="card-body text-center">
                                    <h6 className="text-muted">Leverage</h6>
                                    <h4 className="text-secondary">{result.leverage ?? "-"}</h4>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Charts Row */}
                    <div className="row mt-4 g-4">
                        {/* Left: Sparkline / Trend */}
                        <div className="col-md-6">
                            <div className="card p-3 shadow-sm h-100">
                                <h6 className="mb-3">Predicted Short-Term Trend</h6>
                                {trendPrices && trendPrices.length > 0 ? (
                                    <Line data={trendData} options={{ ...commonOptions, plugins: { legend: { display: false } }, scales: { x: { display: false } } }} height={150} />
                                ) : (
                                    <div className="text-center text-muted py-5">No trend available</div>
                                )}
                                <div className="mt-3 d-flex justify-content-between">
                                    <small>Price: ${result.price ?? "N/A"}</small>
                                    <small>SL: ${result.stop_loss ? result.stop_loss.toFixed(2) : "N/A"}</small>
                                    <small>TP: ${result.take_profit ? result.take_profit.toFixed(2) : "N/A"}</small>
                                </div>
                            </div>
                        </div>

                        {/* Right: Bar + Pie stacked */}
                        <div className="col-md-6">
                            <div className="card p-3 shadow-sm h-100">
                                <h6 className="mb-3">AI Confidence (Top Coins)</h6>
                                <Bar data={barData} options={{ ...commonOptions, plugins: { legend: { display: false } } }} height={140} />
                                <hr />
                                <h6 className="mb-3 mt-3">Market Sentiment</h6>
                                <div style={{ maxWidth: 300, margin: "0 auto" }}>
                                    <Pie data={sentimentPie} options={{ ...commonOptions, plugins: { legend: { position: "bottom" } } }} />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            ) : (
                // no result yet
                <div className="card p-4 text-center text-muted">
                    <p className="mb-0">Enter symbol & investment, then click Predict to run the model.</p>
                </div>
            )}
        </div>
    );
}
