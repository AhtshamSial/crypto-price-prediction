import React from "react";
import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import { AuthContextProvider } from "./components/AuthContext"; // or "./services/AuthContext"
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import ProtectedRoute from "./components/ProtectedRoute";

// Import pages
import Dashboard  from "./pages/Dashboard";
import CoinDetail from "./pages/CoinDetail";
import Prediction from "./pages/Prediction";
import LoginSignUp from "./pages/LoginSignUp";

function App() {
    return (
        <AuthContextProvider>
            <Router>
                <div className="app-wrapper d-flex flex-column min-vh-100">
                    {/* Navigation */}
                    <Navbar />

                    {/* Main Content */}
                    <div className="flex-grow-1">
                        <Routes>
                            {/* Public Routes */}
                            <Route path="/" element={<Dashboard />} />
                            <Route path="/coin/:coinId" element={<CoinDetail />} />
                            <Route path="/auth" element={<LoginSignUp />} />

                            {/* Protected Routes */}
                            <Route
                                path="/prediction"
                                element={
                                    <ProtectedRoute>
                                        <Prediction />
                                    </ProtectedRoute>
                                }
                            />

                            {/* Fallback */}
                            <Route path="*" element={<Navigate to="/" replace />} />
                        </Routes>
                    </div>

                    {/* Footer */}
                    <Footer />
                </div>
            </Router>
        </AuthContextProvider>
    );
}

export default App;