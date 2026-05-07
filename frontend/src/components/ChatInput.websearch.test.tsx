import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import ChatInput from "./ChatInput"

vi.mock("@/hooks/useHotkeys", () => ({
  useHotkeys: () => {},
}))

describe("ChatInput 联网搜索", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "ResizeObserver",
      class {
        observe() {}
        unobserve() {}
        disconnect() {}
      },
    )
  })

  it("switch 反映 websearch 并触发 onWebsearchChange", async () => {
    const user = userEvent.setup()
    const onWeb = vi.fn()
    const { rerender } = render(
      <ChatInput websearch onWebsearchChange={onWeb} onSend={vi.fn()} />,
    )
    const sw = screen.getByRole("switch", { name: /联网搜索/ })
    expect(sw).toHaveAttribute("aria-checked", "true")
    await user.click(sw)
    expect(onWeb).toHaveBeenCalledWith(false)

    rerender(<ChatInput websearch={false} onWebsearchChange={onWeb} onSend={vi.fn()} />)
    expect(screen.getByRole("switch", { name: /联网搜索/ })).toHaveAttribute("aria-checked", "false")
  })
})
