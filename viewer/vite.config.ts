import { defineConfig } from "vite";

export default defineConfig({
  server: { proxy: { "/ws": { target: "ws://127.0.0.1:8765", ws: true } } },
  build: { outDir: "dist", emptyOutDir: true, chunkSizeWarningLimit: 1200 },
});
