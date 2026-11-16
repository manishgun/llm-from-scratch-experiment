import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

/*
 * The backend port is read from the environment and threaded to BOTH the dev
 * proxy and the browser bundle, so a port change needs one variable rather than
 * edits in three files. The frontend reads it as `import.meta.env.VITE_BACKEND_PORT`
 * when opening its WebSocket, which cannot go through the HTTP proxy.
 */
const BACKEND_PORT = process.env.PORT || '3001';

export default defineConfig({
  plugins: [react()],
  define: {
    'import.meta.env.VITE_BACKEND_PORT': JSON.stringify(BACKEND_PORT),
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: Number(process.env.FRONTEND_PORT) || 5173,
    proxy: {
      '/api': {
        target: `http://localhost:${BACKEND_PORT}`,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist/frontend',
    // Source maps make a production stack trace readable, and this is a tool
    // people are meant to read and modify.
    sourcemap: true,
  },
});
