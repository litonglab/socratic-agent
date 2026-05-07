import path from 'node:path'
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'node',
    environmentMatchGlobs: [['src/**/*.test.tsx', 'jsdom']],
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
    coverage: {
      provider: 'istanbul',
      reporter: ['text', 'text-summary', 'json'],
      reportsDirectory: 'coverage',
      include: ['src/**/*.{ts,tsx}'],
      exclude: ['**/node_modules/**', 'src/**/*.test.ts', 'src/**/*.test.tsx'],
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
