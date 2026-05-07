import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'node:path'
import istanbul from 'vite-plugin-istanbul'

const coverageDev = process.env.VITE_COVERAGE === 'true'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    ...(coverageDev
      ? [
          istanbul({
            include: 'src/**/*',
            exclude: ['**/node_modules/**', '**/e2e/**', '**/*.test.ts', '**/*.test.tsx'],
            extension: ['.ts', '.tsx', '.js', '.jsx'],
            requireEnv: true,
          }),
        ]
      : []),
  ].filter(Boolean),
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    // istanbul 开发态打点时避免反复提示；仅在 VITE_COVERAGE 时开启
    sourcemap: process.env.VITE_COVERAGE === 'true',
  },
})
