import React from "react";
import { Line } from "react-chartjs-2";
import "../App.css";

const CoinModal = ({ coin, onClose }) => {
    const chartData = {
        labels: coin.sparkline_in_7d.price.map((_, i) => i),
        datasets: [
            { label: coin.symbol.toUpperCase(), data: coin.sparkline_in_7d.price, borderColor: "#007bff", fill: false }
        ]
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <button className="close-btn" onClick={onClose}>×</button>
                <h4>{coin.name} ({coin.symbol.toUpperCase()})</h4>
                <p>Price: ${coin.current_price.toLocaleString()}</p>
                <p>Market Cap: ${coin.market_cap.toLocaleString()}</p>
                <p>24h Change: {coin.price_change_percentage_24h.toFixed(2)}%</p>
                <Line data={chartData} />
            </div>
        </div>
    );
};

export default CoinModal;
