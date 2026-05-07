import React from "react"
import { describe, expect, it } from "vitest"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import TraceList from "./TraceList"

describe("TraceList", () => {
  it("展示联网搜索工具元信息", async () => {
    const user = userEvent.setup()
    render(
      <TraceList
        traces={[
          { tool: "web_search", input: "OSPF 基本概念" },
          { tool: "unknown_tool", output: "x".repeat(200) },
        ]}
      />,
    )
    await user.click(screen.getByText(/调用了 2 个工具/))
    expect(screen.getByText("联网搜索")).toBeInTheDocument()
    expect(screen.getByText("unknown_tool")).toBeInTheDocument()
    expect(screen.getByText(/输入：/)).toBeInTheDocument()
  })
})
