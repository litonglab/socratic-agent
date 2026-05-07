import { describe, expect, it } from "vitest"
import { RegisterRules } from "./validate"

describe("RegisterRules（与前端注册校验一致）", () => {
  it("用户名 strict：合法短字母数字可通过", () => {
    expect(RegisterRules.username.strict("abc12")).toBeNull()
  })

  it("用户名 strict：超长或非法字符报错", () => {
    expect(RegisterRules.username.strict("abcdefghijk")).toMatch(/最多/)
    expect(RegisterRules.username.strict("bad_name")).not.toBeNull()
  })

  it("密码 strict：至少 8 位且包含字母与数字", () => {
    expect(RegisterRules.password.strict("short1")).not.toBeNull()
    expect(RegisterRules.password.strict("abcdefgh")).not.toBeNull()
    expect(RegisterRules.password.strict("abcd1234")).toBeNull()
  })
})
