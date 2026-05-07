import type { Page } from "@playwright/test"
import { expect, test } from "./fixtures"
import { registerAndOpenChat } from "./helpers"

const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8000"

async function sendMessageAndWaitStream(
  page: Page,
  prompt: string,
): Promise<{ enable_websearch?: boolean }> {
  const reqPromise = page.waitForRequest(
    (r) => r.url().includes("/api/chat/stream") && r.method() === "POST",
    { timeout: 240_000 },
  )
  const respPromise = page.waitForResponse(
    (r) =>
      r.url().includes("/api/chat/stream") &&
      r.request().method() === "POST" &&
      r.status() === 200,
    { timeout: 240_000 },
  )
  await page.getByPlaceholder(/输入你的问题/).fill(prompt)
  const [, req] = await Promise.all([
    respPromise,
    reqPromise,
    page.getByRole("button", { name: "发送" }).click(),
  ])
  const body = req.postDataJSON() as { enable_websearch?: boolean }
  return body
}

test.describe("功能：联网搜索与侧栏", () => {
  test.beforeAll(async ({ request }) => {
    const res = await request.get(`${BACKEND}/health`)
    expect(res.ok(), `后端不可用（期望 ${BACKEND}/health）`).toBeTruthy()
  })

  test("默认开启联网搜索：请求体 enable_websearch 为 true", async ({ page }) => {
    await registerAndOpenChat(page)
    const body = await sendMessageAndWaitStream(page, `e2e 联网默认 ${Date.now()}`)
    expect(body.enable_websearch).toBe(true)
    await expect(page.getByRole("paragraph").filter({ hasText: /e2e 联网默认/ })).toBeVisible()
  })

  test("关闭联网搜索后：请求体 enable_websearch 为 false", async ({ page }) => {
    await registerAndOpenChat(page)
    await page.getByRole("switch", { name: /联网搜索/ }).click()
    await expect(page.getByRole("switch", { name: /联网搜索/ })).toHaveAttribute("aria-checked", "false")
    const body = await sendMessageAndWaitStream(page, `e2e 联网关闭 ${Date.now()}`)
    expect(body.enable_websearch).toBe(false)
  })

  test("再次打开联网搜索：请求体恢复为 true", async ({ page }) => {
    await registerAndOpenChat(page)
    const sw = page.getByRole("switch", { name: /联网搜索/ })
    await sw.click()
    await expect(sw).toHaveAttribute("aria-checked", "false")
    await sw.click()
    await expect(sw).toHaveAttribute("aria-checked", "true")
    const body = await sendMessageAndWaitStream(page, `e2e 联网再开 ${Date.now()}`)
    expect(body.enable_websearch).toBe(true)
  })

  test("桌面宽度下收起再展开侧栏", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 800 })
    await registerAndOpenChat(page)
    await page.getByRole("button", { name: "收起侧栏" }).click()
    await expect(page.getByRole("button", { name: "展开侧栏" })).toBeVisible()
    await page.getByRole("button", { name: "展开侧栏" }).click()
    await expect(page.getByRole("button", { name: "收起侧栏" })).toBeVisible()
  })
})
