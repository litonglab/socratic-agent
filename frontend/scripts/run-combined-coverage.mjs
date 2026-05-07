/**
 * Vitest istanbul + Playwright istanbul（Vite instrumentation）合并报告。
 * 需要：后端可访问 DEEPSEEK 等就绪；会先清空 coverage 与 .nyc_output。
 */
import { execSync } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")
process.chdir(root)

for (const p of [path.join(root, "coverage"), path.join(root, ".nyc_output")]) {
  fs.rmSync(p, { recursive: true, force: true })
}

execSync("npx vitest run --coverage", { stdio: "inherit" })

execSync("npx playwright test --project=chromium", {
  stdio: "inherit",
  env: { ...process.env, E2E_COVERAGE: "1", VITE_COVERAGE: "true" },
})

execSync("node scripts/report-combined-istanbul.mjs", { stdio: "inherit" })
