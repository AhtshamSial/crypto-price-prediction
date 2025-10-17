import React from "react";
import { Line } from "react-chartjs-2";
import "../Styles/Table.css";

const CryptoTable = ({ coins, onSelectCoin }) => {
    return (
        <div className="crypto-table-container">
            <div className="crypto-table-header">
                <h2 className="text-center">Cryptocurrency Market</h2>
                <p className="text-center">
                    Track the top 20 coins with real-time price, market cap, volume, and 7-day trend.
                </p>
            </div>

            {/* Horizontal scroll wrapper */}
            <div className="crypto-table-wrapper-horizontal">
                <table className="crypto-table">
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Coin</th>
                            <th>Price</th>
                            <th>Market Cap</th>
                            <th>24h %</th>
                            <th>Volume</th>
                            <th>7d Trend</th>
                        </tr>
                    </thead>
                    <tbody>
                        {coins.map((coin, i) => {
                            const sparklineData = {
                                labels: coin.sparkline_in_7d.price.map((_, idx) => idx),
                                datasets: [
                                    {
                                        data: coin.sparkline_in_7d.price,
                                        borderColor: coin.price_change_percentage_24h > 0 ? "#28a745" : "#dc3545",
                                        borderWidth: 1,
                                        fill: false,
                                        tension: 0.3,
                                        pointRadius: 0,
                                    },
                                ],
                            };

                            const sparklineOptions = {
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: { legend: { display: false } },
                                elements: { line: { borderWidth: 1 } },
                                scales: { x: { display: false }, y: { display: false } },
                            };

                            return (
                                <tr key={coin.id} onClick={() => onSelectCoin(coin)} className="clickable-row">
                                    <td>{i + 1}</td>
                                    <td>
                                        <img src={coin.image} alt={coin.name} width="20" style={{ marginRight: "5px" }} />
                                        {coin.name} ({coin.symbol.toUpperCase()})
                                    </td>
                                    <td>${coin.current_price.toLocaleString()}</td>
                                    <td>${coin.market_cap.toLocaleString()}</td>
                                    <td className={coin.price_change_percentage_24h > 0 ? "green" : "red"}>
                                        {coin.price_change_percentage_24h.toFixed(2)}%
                                    </td>
                                    <td>${coin.total_volume.toLocaleString()}</td>
                                    <td>
                                        <div className="sparkline-cell">
                                            <Line data={sparklineData} options={sparklineOptions} />
                                        </div>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default CryptoTable;
