import React from "react";
import { Link } from "react-router-dom";
import "../Styles/Footer.css"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faChartBar } from "@fortawesome/free-solid-svg-icons";

const Footer = () => {
    const year = new Date().getFullYear();

    return (
        <footer className="cv-footer">
            {/* ── Top grid ── */}
            <div className="cv-footer__top">

                {/* Brand */}
                <div className="cv-footer__brand">
                    <Link to="/" className="cv-footer__brand-logo">
                        <div className="cv-footer__brand-icon">₿</div>
                        <span className="cv-footer__brand-name">CryptoVision</span>
                    </Link>
                    <p className="cv-footer__brand-desc">
                        AI-powered cryptocurrency analysis and prediction platform.
                        Track real-time prices, market trends, and get intelligent
                        forecasts for smarter decisions.
                    </p>
                    <div className="cv-footer__status">
                        <div className="cv-footer__status-dot" />
                        All systems operational
                    </div>
                </div>

                {/* Navigate — only real routes */}
                <div>
                    <div className="cv-footer__col-title">Navigate</div>
                    <ul className="cv-footer__links">
                        <li>
                            <Link to="/">
                                <FontAwesomeIcon icon={faChartBar} />
                                Dashboard
                            </Link>
                        </li>
                        <li>
                            <Link to="/prediction">
                                <span className="link-icon">🔮</span>
                                AI Predictions
                                <span className="cv-footer__badge cv-footer__badge--lock">Pro</span>
                            </Link>
                        </li>
                        <li>
                            <Link to="/auth">
                                <span className="link-icon">🔐</span>
                                Sign In / Register
                                <span className="cv-footer__badge cv-footer__badge--new">Free</span>
                            </Link>
                        </li>
                    </ul>
                </div>

                {/* Data & info */}
                <div>
                    <div className="cv-footer__col-title">Data & Info</div>
                    <ul className="cv-footer__links">
                        <li>
                            <a href="https://www.coingecko.com" target="_blank" rel="noreferrer">
                                <span className="link-icon">🦎</span>
                                Powered by CoinGecko
                            </a>
                        </li>
                        <li>
                            <a href="https://docs.coingecko.com" target="_blank" rel="noreferrer">
                                <span className="link-icon">📄</span>
                                API Documentation
                            </a>
                        </li>
                        <li>
                            <a href="https://github.com" target="_blank" rel="noreferrer">
                                <span className="link-icon">⚙️</span>
                                Source Code
                            </a>
                        </li>
                    </ul>
                </div>

            </div>

            {/* ── Divider ── */}
            <div className="cv-footer__divider"><hr /></div>

            {/* ── Bottom bar ── */}
            <div className="cv-footer__bottom">
                <p className="cv-footer__copy">
                    © {year} <span>CryptoVision</span> — Built with React & CoinGecko API
                </p>
                <p className="cv-footer__disclaimer">
                    <strong>⚠ Not financial advice.</strong> For informational purposes only.
                </p>
            </div>

        </footer>
    );
};

export default Footer;
