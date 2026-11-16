/// <reference types="vite/client" />

/**
 * Typed build-time environment, so `import.meta.env` is checked rather than
 * `any`. `VITE_BACKEND_PORT` is injected by vite.config.ts from the PORT
 * variable, keeping the dev proxy and the browser's WebSocket in agreement.
 */
interface ImportMetaEnv {
  readonly VITE_BACKEND_PORT: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
