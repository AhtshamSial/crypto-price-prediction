import axios from "axios";

const api = axios.create({
    baseURL: "http://127.0.0.1:5000", // Flask backend address
    timeout: 120000
});

export default api;
