import type { Page } from "@playwright/test"

export function uniqueUsername(): string {
  const rand = Math.random().toString(36).slice(2, 8)
  return (`u${Date.now().toString(36)}${rand}`).replace(/[^a-z0-9]/gi, "").slice(0, 10).padEnd(8, "x")
}

/** 完成注册并进入首页聊天（无 mock） */
export async function registerAndOpenChat(
  page: Page,
  password = "e2etest1",
): Promise<{ username: string }> {
  const username = uniqueUsername()
  await page.goto("/login")
  await page.getByRole("tab", { name: "注册" }).click()
  await page.locator("#reg-username").fill(username)
  await page.locator("#reg-password").fill(password)
  await page.getByRole("button", { name: "下一步" }).click()
  await page.locator("#reg-name").fill("E2E 用户")
  await page.locator("#reg-student_id").fill(`s${username}`)
  await page.locator("#reg-nickname").fill("e2e")
  await page.locator("#reg-class_name").fill("计算机网络试验班")
  await page.locator("#reg-email").fill(`${username}@example.com`)
  await page.getByRole("button", { name: "完成注册" }).click()
  await page.waitForURL("/", { timeout: 30_000 })
  await page.getByPlaceholder(/输入你的问题/).waitFor({ state: "visible", timeout: 30_000 })
  return { username }
}
