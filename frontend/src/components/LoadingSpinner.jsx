import React from "react";
import "../Styles/Spinner.css";

const LoadingSpinner = () => {
    return (
        <div className="spinner-overlay">
            <div className="spinner-container">
                <div className="custom-spinner"></div>
            </div>
        </div>
    );
};

export default LoadingSpinner;
