import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from "./AuthContext";

const ProtectedRoute = ({ children }) => {
    const { currentUser } = useAuth();

    if (currentUser === null) {
        // Loading state
        return <LoadingSpinner />;
    }

    if (!currentUser) {
        // Not authenticated - redirect
        return <Navigate to="/auth" replace />;
    }

    // Authenticated - show content
    return children;
};

export default ProtectedRoute;