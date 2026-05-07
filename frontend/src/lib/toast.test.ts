/**
 * @vitest-environment jsdom
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { dismissToast, getToastSnapshot, subscribeToasts, toast } from "./toast"

describe("toast", () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
    while (getToastSnapshot().length) {
      dismissToast(getToastSnapshot()[0].id)
    }
  })

  it("info 入队并可被订阅接收", () => {
    const fn = vi.fn()
    const unsub = subscribeToasts(fn)
    toast.info("hi")
    expect(fn).toHaveBeenCalled()
    expect(getToastSnapshot().some((t) => t.text === "hi")).toBe(true)
    unsub()
  })

  it("定时后自动 dismiss", () => {
    toast.info("temp", 1000)
    const id = getToastSnapshot()[0].id
    expect(getToastSnapshot()).toHaveLength(1)
    vi.advanceTimersByTime(1000)
    expect(getToastSnapshot().some((t) => t.id === id)).toBe(false)
  })
})
