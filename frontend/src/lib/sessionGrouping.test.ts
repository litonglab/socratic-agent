import { describe, expect, it } from "vitest"
import { groupSessionsByUpdatedAt, startOfDay } from "./sessionGrouping"
import type { SessionMeta } from "./api"

describe("sessionGrouping", () => {
  it("startOfDay 对齐本地自然日零点", () => {
    const d = new Date(2026, 4, 7, 15, 30, 0)
    expect(startOfDay(d)).toBe(new Date(2026, 4, 7, 0, 0, 0, 0).getTime())
  })

  it("无 updated_at 的会话进入「其他」桶", () => {
    const groups = groupSessionsByUpdatedAt([{ session_id: "a", title: "x" }])
    expect(groups).toHaveLength(1)
    expect(groups[0].key).toBe("unknown")
  })

  it("同一天内按 updated_at 倒序归入今天", () => {
    const now = new Date()
    const morning = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 8, 0, 1)
    const noon = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 14, 0, 2)
    const items: SessionMeta[] = [
      { session_id: "old", title: "a", updated_at: morning.toISOString() },
      { session_id: "new", title: "b", updated_at: noon.toISOString() },
    ]
    const groups = groupSessionsByUpdatedAt(items)
    const today = groups.find((g) => g.key === "today")
    expect(today).toBeDefined()
    expect(today!.items.map((s) => s.session_id)).toEqual(["new", "old"])
  })
})
