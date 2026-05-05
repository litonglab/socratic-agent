import { useEffect, useRef, useState } from "react"
import { ThumbsUp, ThumbsDown, ChevronDown, ChevronRight, Brain, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import Markdown from "./Markdown"

export type FeedbackValue = "like" | "dislike" | null

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
  pending?: boolean
  thinking?: string
  message_id?: string | null
  feedback?: FeedbackValue
  images?: string[] // base64 data URLs（用户消息附带）
}

interface Props {
  messages: ChatMessage[]
  onFeedback?: (messageId: string, next: "like" | "dislike" | "cancel") => void
}

export default function MessageList({ messages, onFeedback }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = ref.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  return (
    <div ref={ref} className="flex-1 min-h-0 overflow-y-auto px-6 py-6">
      <div className="max-w-3xl mx-auto space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-[hsl(var(--muted-foreground))] py-20">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-gradient-to-b from-[#A42A1E] to-[#861A11] text-white text-xl font-extrabold shadow-md mb-3">
              N
            </div>
            <div className="text-2xl font-bold text-[hsl(var(--primary))]">NetRUC Agent</div>
          </div>
        )}
        {messages.map((m, i) => (
          <MessageRow key={i} m={m} onFeedback={onFeedback} />
        ))}
      </div>
    </div>
  )
}

function MessageRow({
  m,
  onFeedback,
}: {
  m: ChatMessage
  onFeedback?: (messageId: string, next: "like" | "dislike" | "cancel") => void
}) {
  const [thinkOpen, setThinkOpen] = useState(false)
  const isUser = m.role === "user"
  // 流式且还没开始输出最终回答时，显示"正在思考中"占位
  // （后端的 thinking 仅在 done 事件返回，无法流式增量）
  const showThinkingHint = !isUser && m.pending && !m.content
  // 是否渲染气泡：用户消息始终渲染；assistant 仅在有内容、或非 pending 时渲染
  // （避免"正在思考中"下面跟一个空气泡）
  const showBubble = isUser || !!m.content || !m.pending

  return (
    <div className={cn("flex flex-col", isUser ? "items-end" : "items-start")}>
      {/* assistant 思考区：放在气泡上方，浅灰斜体，无背景无边框，避免被误认为最终回答 */}
      {showThinkingHint && (
        <div className="flex items-center gap-1.5 text-xs italic text-[hsl(var(--muted-foreground))] mb-1.5 px-1">
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
          <span>正在思考中…</span>
        </div>
      )}
      {!isUser && !m.pending && m.thinking && (
        <button
          type="button"
          className="flex items-center gap-1 text-xs italic text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--primary))] mb-1 px-1"
          onClick={() => setThinkOpen((v) => !v)}
        >
          {thinkOpen ? (
            <ChevronDown className="w-3.5 h-3.5" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5" />
          )}
          <Brain className="w-3.5 h-3.5" />
          <span>思考过程</span>
        </button>
      )}
      {!isUser && !m.pending && m.thinking && thinkOpen && (
        <div className="max-w-[82%] mb-2 px-1">
          <Markdown
            content={m.thinking}
            className="text-xs italic text-[hsl(var(--muted-foreground))]"
          />
        </div>
      )}

      {showBubble && (
        <div
          className={cn(
            "max-w-[82%] rounded-2xl px-4 py-3 text-sm shadow-sm",
            isUser
              ? "bg-[hsl(var(--primary))] text-white"
              : "bg-white border border-[hsl(var(--border))]",
            m.pending && "opacity-90",
          )}
        >
          {/* 用户附带图片 */}
          {isUser && m.images && m.images.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-2">
              {m.images.map((src, idx) => (
                <img
                  key={idx}
                  src={src}
                  alt={`uploaded-${idx}`}
                  className="max-w-[180px] max-h-[180px] rounded-lg border border-white/30"
                />
              ))}
            </div>
          )}

          {/* 正文 */}
          {m.content ? (
            <Markdown content={m.content} inverse={isUser} />
          ) : m.pending ? (
            <div className="text-xs italic opacity-70">正在生成…</div>
          ) : null}
        </div>
      )}

      {/* assistant 反馈按钮 */}
      {!isUser && !m.pending && m.message_id && onFeedback && (
        <div className="mt-1 flex items-center gap-1">
          <button
            className={cn(
              "p-1.5 rounded-md transition-colors",
              m.feedback === "like"
                ? "text-[hsl(var(--primary))] bg-[hsl(var(--primary)/0.10)]"
                : "text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--primary))] hover:bg-[hsl(var(--accent))]",
            )}
            onClick={() =>
              onFeedback(m.message_id!, m.feedback === "like" ? "cancel" : "like")
            }
            title="赞"
          >
            <ThumbsUp className="w-3.5 h-3.5" />
          </button>
          <button
            className={cn(
              "p-1.5 rounded-md transition-colors",
              m.feedback === "dislike"
                ? "text-[hsl(var(--destructive))] bg-red-50"
                : "text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--destructive))] hover:bg-[hsl(var(--accent))]",
            )}
            onClick={() =>
              onFeedback(m.message_id!, m.feedback === "dislike" ? "cancel" : "dislike")
            }
            title="踩"
          >
            <ThumbsDown className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  )
}
