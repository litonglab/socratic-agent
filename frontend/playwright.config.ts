import { defineConfig, devices } from "@playwright/test"

/**
 * E2E：真实后端（默认 http://127.0.0.1:8000），通过 Vite 代理 `/api`。
 * 在 frontend 目录执行 `npm run test:e2e`（webServer 会启动或复用本目录的 dev）。
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: [["list"]],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://127.0.0.1:5173",
    trace: "on-first-retry",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev -- --host 127.0.0.1 --port 5173",
    url: "http://127.0.0.1:5173",
    // coverage 必须使用带 istanbul 打点的新 dev server，不能复用手动起的无打点进程
    reuseExistingServer: process.env.E2E_COVERAGE === "1" ? false : !process.env.CI,
    timeout: 120_000,
    env:
      process.env.E2E_COVERAGE === "1"
        ? { VITE_COVERAGE: "true" }
        : undefined,
  },
})
