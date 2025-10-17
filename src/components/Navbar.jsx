import React from "react";
import { Link } from "react-router-dom";
import "../Styles/Navbar.css";

export default function Navbar() {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark px-5 py-2 mb-4">
      <div className="container-fluid">
        <Link className="navbar-brand fw-semi-bold fs-4" to="/">
          AI Crypto Predictor
        </Link>

        {/* Toggler button */}
        <button
          className="navbar-toggler custom-toggler"
          type="button"
          data-bs-toggle="collapse"
          data-bs-target="#navbarNav"
          aria-controls="navbarNav"
          aria-expanded="false"
          aria-label="Toggle navigation"
        >
          <div className="toggler-icon">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </button>

        {/* Collapsible content */}
        <div className="collapse navbar-collapse" id="navbarNav">
          <ul className="navbar-nav ms-auto">
            <li className="nav-item">
              <Link className="nav-link" to="/">
                Dashboard
              </Link>
            </li>
            <li className="nav-item">
              <Link className="nav-link" to="/prediction">
                Prediction
              </Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}
