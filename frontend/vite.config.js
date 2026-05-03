import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Ensures the dev server handles client-side routing correctly
  server: {
    port: 5173,
  },
  // Makes environment variables available — VITE_ prefix required
  // VITE_API_URL is read in services/api.js
})
