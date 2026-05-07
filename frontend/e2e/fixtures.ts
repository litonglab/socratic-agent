import { mkdirSync, writeFileSync } from "node:fs"
import { randomUUID } from "node:crypto"
import path from "node:path"
import { test as base } from "@playwright/test"

/**
 * E2E_COVERAGE=1 时页面由 vite-plugin-istanbul 注入后的 __coverage__ 落盘，
 * 与 Vitest 的 istanbul 报告合并生成 coverage/combined/。
 */
export const test = base.extend({
  page: async ({ page }, use) => {
    await use(page)
    if (process.env.E2E_COVERAGE !== "1") return
    try {
      const raw = await page.evaluate(() => {
        try {
          const cov = (globalThis as unknown as { __coverage__?: Record<string, unknown> }).__coverage__
          if (!cov || typeof cov !== "object") return null
          return JSON.stringify(cov)
        } catch {
          return null
        }
      })
      if (!raw) return
      const obj = JSON.parse(raw) as Record<string, unknown>
      if (!obj || typeof obj !== "object" || Object.keys(obj).length === 0) return
      const dir = path.join(process.cwd(), ".nyc_output")
      mkdirSync(dir, { recursive: true })
      writeFileSync(path.join(dir, `e2e-${randomUUID()}.json`), raw)
    } catch {
      // 页面提前关闭或非浏览器上下文时静默跳过
    }
  },
})

export const expect = base.expect
