import { useEffect, useRef, useState, useCallback } from "react"
import {
  ChevronDown,
  ChevronRight,
  Brain,
  Loader2,
  ArrowDown,
  Sparkles,
  Wrench,
  BookOpen,
  Network,
  Globe,
  Pencil,
  X as XIcon,
  CornerDownLeft,
} from "lucide-react"
import { cn } from "@/lib/utils"
import Markdown from "./Markdown"
import MessageActions, { HintLevelChip } from "./MessageActions"
import TraceList from "./TraceList"
import BrandLogo from "./BrandLogo"
import type { ToolTrace, ChatStage, ChatState } from "@/lib/api"

const BOTTOM_THRESHOLD_PX = 80

export type FeedbackValue = "like" | "dislike" | null

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
  pending?: boolean
  thinking?: string
  /** 流式期间累积的思考增量，仅在 pending 助手消息上有意义；done 后切回 thinking */
  streaming_thinking?: string
  message_id?: string | null
  feedback?: FeedbackValue
  images?: string[] // base64 data URLs（用户消息附带）
  // assistant 专用：用于轨迹面板和苏格拉底层级 chip
  tool_traces?: ToolTrace[]
  state?: ChatState
  // 流式期间的当前阶段（仅在 pending 的助手消息上有意义）
  stage?: ChatStage
  stage_tools?: string[]
}

interface Props {
  messages: ChatMessage[]
  onFeedback?: (messageId: string, next: "like" | "dislike" | "cancel") => void
  onRegenerate?: (assistantMessageId: string) => void
  onEditAndResend?: (userMessageIndex: number, newText: string) => void
  /** 点击空状态的建议提问卡片：将文案填入输入框 */
  onPickSuggestion?: (text: string) => void
}

const SUGGESTIONS: Array<{ title: string; prompt: string; icon: React.ReactNode }> = [
  {
    title: "讲解子网划分",
    prompt: "我对实验 13 的子网划分思路有点困惑，可以引导我一步步思考吗？",
    icon: <BookOpen className="w-4 h-4" />,
  },
  {
    title: "拓扑结构问答",
    prompt: "请基于实验 13 的拓扑图说明各路由器之间的连接关系。",
    icon: <Network className="w-4 h-4" />,
  },
  {
    title: "联网搜索补充",
    prompt: "请结合最新的网络资料，比较 RIP 与 OSPF 在收敛速度上的差异。",
    icon: <Globe className="w-4 h-4" />,
  },
  {
    title: "错误排查",
    prompt: "我配置 NAT 后 ping 不通外网，请引导我排查可能的原因。",
    icon: <Wrench className="w-4 h-4" />,
  },
]

export default function MessageList({
  messages,
  onFeedback,
  onRegenerate,
  onEditAndResend,
  onPickSuggestion,
}: Props) {
  const ref = useRef<HTMLDivElement>(null)
  // 用户是否"贴近底部"。只有贴近底部时，新内容才自动滚；否则保留用户阅读位置。
  const [stickToBottom, setStickToBottom] = useState(true)

  const scrollToBottom = useCallback((behavior: ScrollBehavior = "smooth") => {
    const el = ref.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior })
  }, [])

  // 监听滚动，更新 stickToBottom
  useEffect(() => {
    const el = ref.current
    if (!el) return
    function onScroll() {
      if (!el) return
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight
      setStickToBottom(distance <= BOTTOM_THRESHOLD_PX)
    }
    el.addEventListener("scroll", onScroll, { passive: true })
    return () => el.removeEventListener("scroll", onScroll)
  }, [])

  // 仅在贴近底部时自动滚动；否则保留用户位置（可点"回到底部"主动跟进）
  useEffect(() => {
    if (stickToBottom) scrollToBottom("auto")
  }, [messages, stickToBottom, scrollToBottom])

  return (
    <div className="flex-1 min-h-0 relative">
      <div ref={ref} className="absolute inset-0 overflow-y-auto px-6 py-6">
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="flex justify-center mb-3">
                <BrandLogo size={56} roundedClass="rounded-2xl" />
              </div>
              <div className="text-2xl font-bold text-[hsl(var(--primary))]">
                NetRUC Agent
              </div>
              <div className="text-sm text-[hsl(var(--muted-foreground))] mt-1.5">
                计算机网络实验教学助手 · 基于课程文档与拓扑结构化数据回答
              </div>
              <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-xl mx-auto">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s.title}
                    type="button"
                    onClick={() => onPickSuggestion?.(s.prompt)}
                    className="text-left rounded-xl border border-[hsl(var(--border))] bg-white px-3 py-2.5 hover:border-[hsl(var(--primary)/0.4)] hover:bg-[hsl(var(--primary)/0.04)] transition-colors group"
                  >
                    <div className="flex items-center gap-2 text-[hsl(var(--primary))] mb-1">
                      {s.icon}
                      <span className="text-sm font-semibold">{s.title}</span>
                    </div>
                    <div className="text-xs text-[hsl(var(--muted-foreground))] line-clamp-2 group-hover:text-[hsl(var(--ink-700,#4D3D3A))]">
                      {s.prompt}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <MessageRow
              key={i}
              m={m}
              index={i}
              onFeedback={onFeedback}
              onRegenerate={onRegenerate}
              onEditAndResend={onEditAndResend}
            />
          ))}
        </div>
      </div>
      {!stickToBottom && messages.length > 0 && (
        <button
          type="button"
          onClick={() => {
            scrollToBottom("smooth")
            setStickToBottom(true)
          }}
          className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white border border-[hsl(var(--border))] shadow-md text-xs text-[hsl(var(--ink-700,#4D3D3A))] hover:text-[hsl(var(--primary))] hover:border-[hsl(var(--primary)/0.4)] transition-colors"
          aria-label="回到底部"
          title="回到底部"
        >
          <ArrowDown className="w-3.5 h-3.5" />
          回到底部
        </button>
      )}
    </div>
  )
}

const STAGE_META: Record<string, { icon: React.ReactNode; label: string }> = {
  analyzing: {
    icon: <Sparkles className="w-3.5 h-3.5" />,
    label: "分析问题中…",
  },
  tools: {
    icon: <Wrench className="w-3.5 h-3.5" />,
    label: "调用工具中…",
  },
  generating: {
    icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
    label: "生成回答中…",
  },
}

const TOOL_LABEL: Record<string, { label: string; icon: React.ReactNode }> = {
  retrieve_course_docs: { label: "检索文档", icon: <BookOpen className="w-3 h-3" /> },
  rag_retrieve: { label: "检索文档", icon: <BookOpen className="w-3 h-3" /> },
  retrieve: { label: "检索文档", icon: <BookOpen className="w-3 h-3" /> },
  get_topology_context: { label: "读取拓扑", icon: <Network className="w-3 h-3" /> },
  topology: { label: "读取拓扑", icon: <Network className="w-3 h-3" /> },
  web_search: { label: "联网搜索", icon: <Globe className="w-3 h-3" /> },
  websearch: { label: "联网搜索", icon: <Globe className="w-3 h-3" /> },
}

function StageIndicator({ stage, tools }: { stage?: ChatStage; tools?: string[] }) {
  const key = stage || "analyzing"
  const meta = STAGE_META[key] || {
    icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />,
    label: "处理中…",
  }
  return (
    <div className="flex items-center gap-2 text-xs italic text-[hsl(var(--muted-foreground))] mb-1.5 px-1 flex-wrap">
      <span className="text-[hsl(var(--primary))]">{meta.icon}</span>
      <span>{meta.label}</span>
      {key === "tools" && tools && tools.length > 0 && (
        <span className="not-italic flex items-center gap-1">
          {tools.slice(0, 4).map((t, i) => {
            const tm = TOOL_LABEL[t] || {
              label: t,
              icon: <Wrench className="w-3 h-3" />,
            }
            return (
              <span
                key={`${t}-${i}`}
                className="inline-flex items-center gap-1 rounded-full bg-[hsl(var(--accent))] px-1.5 py-0.5 text-[11px] text-[hsl(var(--ink-700,#4D3D3A))]"
              >
                {tm.icon}
                {tm.label}
              </span>
            )
          })}
        </span>
      )}
    </div>
  )
}

interface RowProps {
  m: ChatMessage
  index: number
  onFeedback?: (messageId: string, next: "like" | "dislike" | "cancel") => void
  onRegenerate?: (assistantMessageId: string) => void
  onEditAndResend?: (userMessageIndex: number, newText: string) => void
}

function MessageRow({ m, index, onFeedback, onRegenerate, onEditAndResend }: RowProps) {
  const [thinkOpen, setThinkOpen] = useState(false)
  // 流式思考块的展开状态：null=跟随默认（无 content 时展开，有 content 时折叠）；
  // boolean=用户主动覆盖。
  const [streamThinkOverride, setStreamThinkOverride] = useState<boolean | null>(null)
  const [editing, setEditing] = useState(false)
  const [editText, setEditText] = useState(m.content || "")
  const isUser = m.role === "user"
  const hasContent = !!m.content
  // 流式期间是否正在显示思考过程（pending 且已收到至少一段 thinking）
  const isStreamingThinking =
    !isUser && !!m.pending && !!m.streaming_thinking
  // 默认：内容尚未出现 → 展开（让用户看到一步步思考）；内容出现 → 自动折叠
  const streamThinkAutoOpen = !hasContent
  const streamThinkOpen =
    streamThinkOverride !== null ? streamThinkOverride : streamThinkAutoOpen
  // 阶段指示：pending 期间持续显示；当已经进入 generating 且气泡有正文时隐藏（避免冗余）
  const showStage =
    !isUser &&
    !!m.pending &&
    !(m.stage === "generating" && hasContent)
  // 用户消息始终渲染；assistant 仅在有内容、或非 pending 时渲染气泡
  const showBubble = isUser || hasContent || !m.pending

  return (
    <div className={cn("group/row flex flex-col", isUser ? "items-end" : "items-start")}>
      {showStage && <StageIndicator stage={m.stage} tools={m.stage_tools} />}

      {/* 流式思考过程：
          - content 未出现：展开显示，斜体浅色一步步出现 + 闪烁光标
          - content 出现：自动折叠成可点击按钮，气泡显示回答；用户可点击展开查看 */}
      {isStreamingThinking && (
        <div className="max-w-[82%] mb-1.5 px-1">
          <button
            type="button"
            onClick={() => setStreamThinkOverride((cur) => !(cur === null ? streamThinkAutoOpen : cur))}
            className="flex items-center gap-1 text-xs italic text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--primary))]"
            aria-expanded={streamThinkOpen}
          >
            {streamThinkOpen ? (
              <ChevronDown className="w-3.5 h-3.5" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5" />
            )}
            <Brain className="w-3.5 h-3.5 text-[hsl(var(--primary))]" />
            <span>{streamThinkOpen ? "思考过程" : "已思考，点击查看"}</span>
          </button>
          {streamThinkOpen && (
            <div className="mt-1 pl-5 flex items-start gap-1.5 text-xs italic text-[hsl(var(--muted-foreground))]">
              <div className="leading-relaxed whitespace-pre-wrap break-words">
                {m.streaming_thinking}
                {!hasContent && (
                  <span className="ml-1 inline-block w-1 h-3 align-[-2px] bg-[hsl(var(--muted-foreground))] animate-pulse" />
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {!isUser && !m.pending && (
        <div className="flex items-center gap-2 mb-1 px-1 flex-wrap">
          <HintLevelChip
            hintLevel={m.state?.hint_level}
            category={m.state?.question_category || m.state?.category}
          />
          {m.thinking && (
            <button
              type="button"
              className="flex items-center gap-1 text-xs italic text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--primary))]"
              onClick={() => setThinkOpen((v) => !v)}
              aria-expanded={thinkOpen}
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
        </div>
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
            "max-w-[82%] rounded-2xl px-4 py-3 text-[15px] leading-relaxed shadow-sm",
            isUser
              ? "bg-[hsl(var(--primary))] text-white"
              : "bg-white border border-[hsl(var(--border))]",
            m.pending && "opacity-90",
          )}
        >
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

          {editing ? (
            <div className="flex flex-col gap-2 min-w-[280px]">
              <textarea
                autoFocus
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
                    e.preventDefault()
                    const v = editText.trim()
                    if (!v) return
                    onEditAndResend?.(index, v)
                    setEditing(false)
                  } else if (e.key === "Escape") {
                    setEditing(false)
                    setEditText(m.content || "")
                  }
                }}
                rows={3}
                className="w-full resize-y rounded-lg bg-white/10 text-white placeholder-white/60 px-2 py-1.5 outline-none border border-white/30 focus:border-white/60 text-sm"
              />
              <div className="flex items-center justify-end gap-1">
                <button
                  type="button"
                  className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-white/15 hover:bg-white/25"
                  onClick={() => {
                    setEditing(false)
                    setEditText(m.content || "")
                  }}
                  title="取消"
                >
                  <XIcon className="w-3 h-3" />
                  取消
                </button>
                <button
                  type="button"
                  className="inline-flex items-center gap-1 text-xs px-2 py-1 rounded-md bg-white text-[hsl(var(--primary))] font-semibold hover:opacity-90"
                  onClick={() => {
                    const v = editText.trim()
                    if (!v) return
                    onEditAndResend?.(index, v)
                    setEditing(false)
                  }}
                  title="保存并重发 (⌘/Ctrl+Enter)"
                >
                  <CornerDownLeft className="w-3 h-3" />
                  发送
                </button>
              </div>
            </div>
          ) : m.content ? (
            <Markdown content={m.content} inverse={isUser} />
          ) : m.pending ? (
            <div className="text-xs italic opacity-70">正在生成…</div>
          ) : null}
        </div>
      )}

      {isUser && !editing && onEditAndResend && (
        <button
          type="button"
          onClick={() => {
            setEditing(true)
            setEditText(m.content || "")
          }}
          className="mt-1 inline-flex items-center gap-1 text-[11px] text-[hsl(var(--muted-foreground))] hover:text-[hsl(var(--primary))] opacity-0 group-hover/row:opacity-100 focus-visible:opacity-100 transition-opacity"
          title="编辑后重发"
          aria-label="编辑后重发"
        >
          <Pencil className="w-3 h-3" />
          编辑
        </button>
      )}

      {!isUser && !m.pending && (
        <>
          <MessageActions m={m} onFeedback={onFeedback} onRegenerate={onRegenerate} />
          <TraceList traces={m.tool_traces} />
        </>
      )}
    </div>
  )
}
