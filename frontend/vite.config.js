import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Inline the (tiny) built CSS into index.html so it isn't a render-blocking request (FCP).
function inlineCss() {
  return {
    name: "inline-css",
    apply: "build",
    enforce: "post",
    transformIndexHtml(html, ctx) {
      if (!ctx.bundle) return html;
      let out = html;
      for (const [file, chunk] of Object.entries(ctx.bundle)) {
        if (file.endsWith(".css") && chunk.type === "asset") {
          const name = file.split("/").pop();
          out = out.replace(
            new RegExp(`<link[^>]+href="[^"]*${name}"[^>]*>`),
            `<style>${chunk.source}</style>`
          );
          delete ctx.bundle[file];
        }
      }
      return out;
    },
  };
}

export default defineConfig({
  plugins: [react(), inlineCss()],
  build: {
    rollupOptions: {
      output: {
        // Split big vendors into their own cacheable chunks (smaller initial parse).
        manualChunks: {
          react: ["react", "react-dom"],
          nivo: ["@nivo/core", "@nivo/bar", "@nivo/line"],
        },
      },
    },
    chunkSizeWarningLimit: 1200,
  },
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
