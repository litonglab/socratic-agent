import { useEffect, useState } from "react"
import { Copy, Check, RefreshCw, ThumbsUp, ThumbsDown } from "lucide-react"
import { cn } from "@/lib/utils"
import { toast } from "@/lib/toast"
import type { ChatMessage } from "./MessageList"

interface Props {
  m: ChatMessage
  onFeedback?: (messageId: string, next: "like" | "dislike" | "cancel") => void
  onRegenerate?: (messageId: string) => void
}

/**
 * 助手消息底部的操作条：复制 / 重新生成 / 反馈。
 * 仅对 message_id 已就绪、非 pending 的助手消息显示。
 */
export default function MessageActions({ m, onFeedback, onRegenerate }: Props) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const t = window.setTimeout(() => setCopied(false), 1500)
    return () => window.clearTimeout(t)
  }, [copied])

  async function handleCopy() {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(m.content || "")
      } else {
        // 兜底：localhost 之类非安全上下文
        const ta = document.createElement("textarea")
        ta.value = m.content || ""
        ta.style.position = "fixed"
        ta.style.opacity = "0"
        document.body.appendChild(ta)
        ta.select()
        document.execCommand("copy")
        document.body.removeChild(ta)
      }
      setCopied(true)
      toast.success("已复制到剪贴板")
    } catch {
      toast.error("复制失败")
    }
  }

  return (
    <div className="mt-1 flex items-center gap-1">
      <ActionBtn
        title="复制回答"
        onClick={handleCopy}
        icon={
          copied ? (
            <Check className="w-3.5 h-3.5 text-emerald-600" />
          ) : (
            <Copy className="w-3.5 h-3.5" />
          )
        }
      />
      {m.message_id && onRegenerate && (
        <ActionBtn
          title="重新生成"
          onClick={() => onRegenerate(m.message_id!)}
          icon={<RefreshCw className="w-3.5 h-3.5" />}
        />
      )}
      {m.message_id && onFeedback && (
        <>
          <ActionBtn
            title="赞"
            active={m.feedback === "like"}
            activeClassName="text-[hsl(var(--primary))] bg-[hsl(var(--primary)/0.10)]"
            onClick={() =>
              onFeedback(m.message_id!, m.feedback === "like" ? "cancel" : "like")
            }
            icon={<ThumbsUp className="w-3.5 h-3.5" />}
          />
          <ActionBtn
            title="踩"
            active={m.feedback === "dislike"}
            activeClassName="text-[hsl(var(--destructive))] bg-red-50"
            onClick={() =>
              onFeedback(m.message_id!, m.feedback === "dislike" ? "cancel" : "dislike")
            }
            icon={<ThumbsDown className="w-3.5 h-3.5" />}
          />
        </>
      )}
    </div>
  )
}

function ActionBtn({
  title,
  onClick,
  icon,
  active,
  activeClassName,
}: {
  title: string
  onClick: () => void
  icon: React.ReactNode
  active?: boolean
  activeClassName?: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-label={title}
      className={cn(
        "p-1.5 rounded-md transition-colors text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--primary))] hover:bg-[hsl(var(--accent))]",
        active && activeClassName,
      )}
    >
      {icon}
    </button>
  )
}

/** 苏格拉底层级 chip：把 hint_level (0..3) 映射成可读标签 */
export function HintLevelChip({
  hintLevel,
  category,
}: {
  hintLevel?: number
  category?: string
}) {
  if (hintLevel === undefined || hintLevel === null) return null
  const level = Math.max(0, Math.min(3, Math.floor(Number(hintLevel))))
  const labels = ["L0 直接回答", "L1 轻提示", "L2 引导", "L3 深度引导"]
  const tone = [
    "bg-[hsl(var(--accent))] text-[hsl(var(--ink-700,#4D3D3A))]",
    "bg-amber-50 text-amber-900 border-amber-200",
    "bg-orange-50 text-orange-900 border-orange-200",
    "bg-rose-50 text-rose-900 border-rose-200",
  ]
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1 text-[11px] px-2 py-0.5 rounded-full border border-[hsl(var(--border))]",
        tone[level],
      )}
      title={
        category
          ? `提示层级 ${level} · 问题类别：${category}`
          : `提示层级 ${level}`
      }
    >
      <span className="font-semibold">{labels[level]}</span>
      {category && <span className="opacity-70">· {category}</span>}
    </div>
  )
}
