/**
 * @vitest-environment jsdom
 */
import { beforeEach, describe, expect, it, vi } from "vitest"
import { getToken, setToken } from "./api"

describe("api token 辅助", () => {
  beforeEach(() => {
    localStorage.clear()
    vi.restoreAllMocks()
  })

  it("setToken / getToken 读写 netruc_auth_token", () => {
    expect(getToken()).toBeNull()
    setToken("abc")
    expect(getToken()).toBe("abc")
    setToken(null)
    expect(getToken()).toBeNull()
  })
})
