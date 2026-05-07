import { describe, expect, it } from "vitest"
import { cn } from "./utils"

describe("cn", () => {
  it("合并冲突的 tailwind 类", () => {
    expect(cn("px-2 py-1", "px-4")).toMatch(/px-4/)
    expect(cn("px-2 py-1", "px-4")).not.toMatch(/px-2/)
  })
})
