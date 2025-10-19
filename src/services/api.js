import axios from "axios";

const api = axios.create({
    baseURL: "http://127.0.0.1:5000", // or whatever port your Flask backend runs on
    headers: {
        "Content-Type": "application/json",
    },
});

export default api;
