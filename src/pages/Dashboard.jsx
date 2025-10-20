import React, { useEffect, useState } from "react";
import axios from "axios";
import { Bar, Line } from "react-chartjs-2";
import CryptoTable from "../components/CryptoTable";
import CoinModal from "../components/CoinModal";
import LoadingSpinner from "../components/LoadingSpinner";
import GlobalStats from "../components/GlobalStates";

const Dashboard = () => {
    const [coins, setCoins] = useState([]);
    const [globalStats, setGlobalStats] = useState({});
    const [loading, setLoading] = useState(true);
    const [selectedCoin, setSelectedCoin] = useState(null);

    const fetchData = async () => {
        try {
            setLoading(true); // Start spinner
            const [coinsRes, globalRes] = await Promise.all([
                axios.get("https://api.coingecko.com/api/v3/coins/markets", {
                    params: {
                        vs_currency: "usd",
                        order: "market_cap_desc",
                        per_page: 20,
                        page: 1,
                        sparkline: true,
                    },
                }),
                axios.get("https://api.coingecko.com/api/v3/global"),
            ]);
            setCoins(coinsRes.data);
            setGlobalStats(globalRes.data.data);
            setTimeout(() => setLoading(false), 500); // Small delay for smooth fade
        } catch (err) {
            console.error("Error fetching data:", err);
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchData();
        const interval = setInterval(fetchData, 60000);
        return () => clearInterval(interval);
    }, []);

    // Keep spinner until data is fully available
    if (loading || coins.length === 0 || !globalStats.total_market_cap) {
        return <LoadingSpinner />;
    }

    // Charts
    const top5 = coins.slice(0, 5);
    const marketCapData = {
        labels: top5.map((c) => c.symbol.toUpperCase()),
        datasets: [
            {
                label: "Market Cap",
                data: top5.map((c) => c.market_cap),
                backgroundColor: ["#007bff", "#28a745", "#ffc107", "#dc3545", "#6f42c1"],
            },
        ],
    };

    const trendData = {
        labels: top5[0]?.sparkline_in_7d?.price.map((_, i) => i) || [],
        datasets: top5.map((c) => ({
            label: c.symbol.toUpperCase(),
            data: c.sparkline_in_7d.price,
            borderColor: "#" + Math.floor(Math.random() * 16777215).toString(16),
            fill: false,
            tension: 0.3,
        })),
    };

    return (
        <div className="dashboard-container px-lg-5">
            <GlobalStats stats={globalStats} />

            <div className="charts mb-4">
                <div className="chart-card">
                    <h6>Top 5 Market Caps</h6>
                    <Bar
                        data={marketCapData}
                        options={{ responsive: true, plugins: { legend: { display: false } } }}
                    />
                </div>
                <div className="chart-card">
                    <h6>Top 5 Sparkline Trends</h6>
                    <Line
                        data={trendData}
                        options={{ responsive: true, plugins: { legend: { position: "bottom" } } }}
                    />
                </div>
            </div>

            <CryptoTable coins={coins} onSelectCoin={setSelectedCoin} />

            {selectedCoin && (
                <CoinModal coin={selectedCoin} onClose={() => setSelectedCoin(null)} />
            )}
        </div>
    );
};

export default Dashboard;
