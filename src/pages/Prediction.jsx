import React, { useEffect, useState } from "react";
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
import api from "../services/api";

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

const COINS = ["BTC", "ETH", "BNB", "SOL"];

export default function Prediction() {
    const [activeCoin, setActiveCoin] = useState("BTC"); // BTC active by default
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [fakeTrend, setFakeTrend] = useState([]);

    // 🔄 Animated fake trend while loading
    useEffect(() => {
        if (!loading) return;

        const base = 50000 + Math.random() * 1000;
        let i = 0;

        const interval = setInterval(() => {
            setFakeTrend((prev) => {
                const next = base + Math.sin(i / 3) * 300 + Math.random() * 100;
                i++;
                return [...prev.slice(-14), Math.round(next * 100) / 100];
            });
        }, 300);

        return () => clearInterval(interval);
    }, [loading]);

    // ▶ Run prediction
    const handlePredict = async () => {
        setLoading(true);
        setResult(null);
        setFakeTrend([]);

        try {
            const res = await api.post("/predict", { symbol: activeCoin });

            if (res.data.error) {
                throw new Error(res.data.message);
            }

            setResult(res.data);
        } catch (err) {
            alert("Prediction failed. Check backend & CORS.");
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    /* ------------------ CHART DATA ------------------ */

    const trendPrices = loading
        ? fakeTrend
        : result
            ? Object.values(result.predictions).map((p) => p.predicted_price)
            : [];

    const trendData = {
        labels: trendPrices.map((_, i) => `T${i + 1}`),
        datasets: [
            {
                label: `${activeCoin} Price Trend`,
                data: trendPrices,
                borderWidth: 2,
                tension: 0.4,
                fill: true,
            },
        ],
    };

    const barData = result && {
        labels: COINS,
        datasets: [
            {
                label: "Signal Strength",
                data: COINS.map((c) =>
                    c === activeCoin ? 90 : 50 + Math.random() * 20
                ),
            },
        ],
    };

    const pieData = result && {
        labels: ["BUY", "HOLD", "SELL"],
        datasets: [
            {
                data: [50, 30, 20],
                backgroundColor: ["#198754", "#ffc107", "#dc3545"],
            },
        ],
    };

    /* ------------------ UI ------------------ */

    return (
        <div className="container my-5">
            {/* TOP BAR */}
            <div className="d-flex justify-content-between align-items-center mb-4">
                <h2 className="fw-bold mb-0">AI Crypto Prediction</h2>
                <button
                    className="btn btn-primary"
                    onClick={handlePredict}
                    disabled={loading}
                >
                    {loading ? "Predicting..." : "Predict"}
                </button>
            </div>

            {/* COIN BUTTONS */}
            <div className="mb-4 d-flex gap-2">
                {COINS.map((coin) => (
                    <button
                        key={coin}
                        className={`btn ${activeCoin === coin ? "btn-dark" : "btn-outline-dark"
                            }`}
                        onClick={() => setActiveCoin(coin)}
                        disabled={loading}
                    >
                        {coin}
                    </button>
                ))}
            </div>

            {/* STATUS */}
            {loading && (
                <div className="alert alert-info text-center">
                    Running AI model for {activeCoin}...
                </div>
            )}

            {/* RESULT */}
            {result && (
                <div className="row g-4 mb-4">
                    <div className="col-md-4">
                        <div className="card p-3 text-center">
                            <h6 className="text-muted">Current Price</h6>
                            <h3>${result.current_price}</h3>
                        </div>
                    </div>

                    {Object.entries(result.predictions).map(([h, p]) => (
                        <div key={h} className="col-md-4">
                            <div className="card p-3 text-center">
                                <h6 className="text-muted">{h}-Day Prediction</h6>
                                <h4>${p.predicted_price}</h4>
                                <span
                                    className={`badge ${p.signal === "BUY"
                                            ? "bg-success"
                                            : p.signal === "SELL"
                                                ? "bg-danger"
                                                : "bg-warning"
                                        }`}
                                >
                                    {p.signal}
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {/* CHARTS */}
            <div className="row g-4">
                <div className="col-md-6">
                    <div className="card p-3">
                        <h6 className="mb-2">Price Trend</h6>
                        <Line data={trendData} />
                    </div>
                </div>

                <div className="col-md-3">
                    <div className="card p-3">
                        <h6 className="mb-2">Signal Strength</h6>
                        {barData ? <Bar data={barData} /> : <p>No data</p>}
                    </div>
                </div>

                <div className="col-md-3">
                    <div className="card p-3">
                        <h6 className="mb-2">Market Bias</h6>
                        {pieData ? <Pie data={pieData} /> : <p>No data</p>}
                    </div>
                </div>
            </div>
        </div>
    );
}
