#!/usr/bin/env node
/** 读取 coverage/coverage-final.json（Vitest）与 .nyc_output/*.json（Playwright），输出到 coverage/combined */

import fs from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import libCoverage from "istanbul-lib-coverage"
import libReport from "istanbul-lib-report"
import reports from "istanbul-reports"

const { createCoverageMap } = libCoverage

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..")
process.chdir(root)

const map = createCoverageMap({})

function mergeFile(absPath) {
  if (!fs.existsSync(absPath)) return
  try {
    const obj = JSON.parse(fs.readFileSync(absPath, "utf8"))
    if (typeof obj !== "object" || obj === null || Object.keys(obj).length === 0) return
    map.merge(createCoverageMap(obj))
  } catch {
    // skip invalid
  }
}

mergeFile(path.join(root, "coverage", "coverage-final.json"))

const nycOut = path.join(root, ".nyc_output")
if (fs.existsSync(nycOut)) {
  for (const name of fs.readdirSync(nycOut)) {
    if (name.endsWith(".json")) mergeFile(path.join(nycOut, name))
  }
}

const combinedDir = path.join(root, "coverage", "combined")
fs.rmSync(combinedDir, { recursive: true, force: true })
fs.mkdirSync(combinedDir, { recursive: true })

const context = libReport.createContext({
  dir: combinedDir,
  defaultSummarizer: "nested",
  coverageMap: map,
})

for (const name of ["text", "html", "lcov"]) {
  reports.create(name, {}).execute(context)
}

console.info("")
console.info("合并报告目录:", combinedDir.replace(root + path.sep, "") + "/")
