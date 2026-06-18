import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,
    // 127.0.0.1 (not localhost) so it doesn't resolve to IPv6 ::1 on Windows (CLAUDE.md).
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true, // proxy the /api/v1/stream WebSocket too
      },
    },
  },
});
