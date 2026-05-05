import { useEffect, useRef } from "react"

export interface HotkeyDef {
  /** 与 KeyboardEvent.key 比较，大小写不敏感。例：'b' / 'k' / '/' / 'Escape' */
  key: string
  /** 需要 Cmd (mac) 或 Ctrl (win/linux) */
  meta?: boolean
  shift?: boolean
  alt?: boolean
  /** 在 input/textarea/contenteditable 中是否仍然触发，默认 false */
  allowInInput?: boolean
}

export type HotkeyBinding = HotkeyDef & {
  handler: (e: KeyboardEvent) => void
}

/**
 * 全局快捷键 hook：
 * - bindings 用 ref 保留最新引用，避免每次渲染重新绑定 keydown 监听
 * - 默认在 input/textarea/contenteditable 内不触发，避免和打字冲突
 * - meta=true 表示需要 Cmd (mac) 或 Ctrl (win/linux)
 */
export function useHotkeys(bindings: HotkeyBinding[], enabled = true) {
  const bindingsRef = useRef<HotkeyBinding[]>(bindings)

  useEffect(() => {
    bindingsRef.current = bindings
  }, [bindings])

  useEffect(() => {
    if (!enabled) return
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null
      const isEditable =
        !!target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)

      for (const b of bindingsRef.current) {
        if (e.key.toLowerCase() !== b.key.toLowerCase()) continue
        const wantMeta = !!b.meta
        const hasMeta = e.metaKey || e.ctrlKey
        if (wantMeta !== hasMeta) continue
        if (!!b.shift !== e.shiftKey) continue
        if (!!b.alt !== e.altKey) continue
        if (isEditable && !b.allowInInput) continue
        e.preventDefault()
        b.handler(e)
        return
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [enabled])
}
