import React from "react";
import "../App.css";

const Footer = () => {
    return (
        <footer className="bg-dark text-light text-center py-3 mt-auto">
            <p className="mb-0">
                © {new Date().getFullYear()} CryptoVision | Powered by CoinGecko API
            </p>
        </footer>
    );
};

export default Footer;
