import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000, // 60 second timeout
});

// Only set Content-Type for requests that have a body (POST, PUT, PATCH)
api.interceptors.request.use((config) => {
  if (["post", "put", "patch"].includes(config.method?.toLowerCase())) {
    config.headers["Content-Type"] = "application/json";
  }
  return config;
});

export default api;
