import { test, expect } from "./fixtures"
import { registerAndOpenChat } from "./helpers"

const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8000"

test.describe("注册并对话（真实后端）", () => {
  test.beforeAll(async ({ request }) => {
    const res = await request.get(`${BACKEND}/health`)
    expect(res.ok(), `后端不可用：请先启动 backend（期望 ${BACKEND}/health）`).toBeTruthy()
  })

  test("注册新账号后发送一条消息并收到助手回复", async ({ page }) => {
    await registerAndOpenChat(page)
    const prompt = `e2e 你好 ${Date.now()}`
    await page.getByPlaceholder(/输入你的问题/).fill(prompt)

    await Promise.all([
      page.waitForResponse(
        (r) =>
          r.url().includes("/api/chat/stream") &&
          r.request().method() === "POST" &&
          r.status() === 200,
        { timeout: 240_000 },
      ),
      page.getByRole("button", { name: "发送" }).click(),
    ])

    await expect(page.getByRole("paragraph").filter({ hasText: prompt })).toBeVisible()
    await expect(page.locator(".prose-base").nth(1)).toBeVisible({ timeout: 180_000 })
  })
})
