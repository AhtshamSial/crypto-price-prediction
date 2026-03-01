import axios from "axios";

const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL || "http://127.0.0.1:8000",
    headers: {
        "Content-Type": "application/json",
    },
    timeout: 120000, // 2 min — model inference can be slow on first run
});

// Response interceptor for unified error handling
api.interceptors.response.use(
    (response) => response,
    (error) => {
        const detail =
            error.response?.data?.detail ||
            error.response?.data?.message ||
            error.message ||
            "Unknown error";
        return Promise.reject(new Error(detail));
    }
);

export default api;
