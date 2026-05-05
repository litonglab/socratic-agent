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
  Lightbulb,
} from "lucide-react"
import { cn } from "@/lib/utils"
import Markdown from "./Markdown"
import MessageActions, { HintLevelChip } from "./MessageActions"
import TraceList from "./TraceList"
import BrandLogo from "./BrandLogo"
import Lightbox from "./Lightbox"
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
  /** 仅前端维护：思考开始的 performance.now() 时间戳 */
  thinking_started_at?: number
  /** 仅前端维护：思考耗时（毫秒），思考结束后写入 */
  thinking_duration_ms?: number
}

interface Props {
  messages: ChatMessage[]
  /** 当前激活的 sessionId：用于按会话缓存滚动位置 */
  sessionId?: string | null
  /** 切换会话时正在拉取消息：渲染骨架屏，避免空白闪烁 */
  loading?: boolean
  onFeedback?: (messageId: string, next: "like" | "dislike" | "cancel") => void
  onRegenerate?: (assistantMessageId: string) => void
  onEditAndResend?: (userMessageIndex: number, newText: string) => void
  /** 点击空状态的建议提问卡片 / 助手消息底部追问 chip：将文案直接发送 */
  onPickSuggestion?: (text: string) => void
}

// 按问题类别给出"下一句问什么"的模板。
// 后端 category 实际取值：LAB_TROUBLESHOOTING / THEORY_CONCEPT / CONFIG_REVIEW / CALCULATION
const FOLLOW_UPS_BY_CATEGORY: Record<string, string[]> = {
  THEORY_CONCEPT: [
    "请举一个具体的例子说明",
    "和它最容易混淆的相近概念是什么？",
    "这个概念在实验里是怎么体现的？",
  ],
  CONFIG_REVIEW: [
    "请展示完整的命令序列",
    "这个配置常见的踩坑点有哪些？",
    "如何在真实设备上验证它生效了？",
  ],
  LAB_TROUBLESHOOTING: [
    "还有哪些可能的故障原因？",
    "如何用 ping / traceroute 进一步定位？",
    "请给我一份完整的排查清单",
  ],
  CALCULATION: [
    "请展示完整的计算过程",
    "再出一道类似的题让我练手",
    "这种题型还有哪些常见陷阱？",
  ],
}
const FOLLOW_UPS_DEFAULT = [
  "请举个例子说明",
  "请进一步深入这个点",
  "如何在实验中验证？",
]
// 当前还在被引导（hint_level >= 2）时，追加两个偏"互动控制"的快捷追问
const FOLLOW_UPS_HINT_HIGH = [
  "我卡住了，请给一个关键提示",
  "我已经想明白了，请直接确认答案",
]

function buildFollowUps(state?: ChatState): string[] {
  const cat = state?.question_category || state?.category || ""
  const base = FOLLOW_UPS_BY_CATEGORY[cat] || FOLLOW_UPS_DEFAULT
  const level = Number(state?.hint_level ?? 0)
  if (level >= 2) {
    // 深度引导时优先暴露互动控制，再附带 1 条 category 模板
    return [...FOLLOW_UPS_HINT_HIGH, base[0]].slice(0, 3)
  }
  return base.slice(0, 3)
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
  sessionId,
  loading,
  onFeedback,
  onRegenerate,
  onEditAndResend,
  onPickSuggestion,
}: Props) {
  const ref = useRef<HTMLDivElement>(null)
  // 用户是否"贴近底部"。只有贴近底部时，新内容才自动滚；否则保留用户阅读位置。
  const [stickToBottom, setStickToBottom] = useState(true)
  // 图片放大预览：所有消息共用一个 Lightbox
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null)
  // 按 sessionId 缓存滚动位置：切回老会话时恢复阅读进度
  const scrollPosRef = useRef<Record<string, number>>({})
  const prevSessionIdRef = useRef<string | null>(null)

  const scrollToBottom = useCallback((behavior: ScrollBehavior = "smooth") => {
    const el = ref.current
    if (!el) return
    el.scrollTo({ top: el.scrollHeight, behavior })
  }, [])

  // 监听滚动：更新 stickToBottom + 持续记录当前会话的 scrollTop
  // deps 包含 sessionId 是因为 onScroll 闭包要引用最新的 sessionId
  useEffect(() => {
    const el = ref.current
    if (!el) return
    function onScroll() {
      if (!el) return
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight
      setStickToBottom(distance <= BOTTOM_THRESHOLD_PX)
      if (sessionId) scrollPosRef.current[sessionId] = el.scrollTop
    }
    el.addEventListener("scroll", onScroll, { passive: true })
    return () => el.removeEventListener("scroll", onScroll)
  }, [sessionId])

  // 仅在贴近底部时自动滚动；切换会话期间（loading）不自动滚，让恢复逻辑接管
  useEffect(() => {
    if (loading) return
    if (stickToBottom) scrollToBottom("auto")
  }, [messages, stickToBottom, scrollToBottom, loading])

  // 切换会话瞬间记录前一个 session 的 scrollTop（onScroll 可能没来得及触发最新值）
  useEffect(() => {
    const prev = prevSessionIdRef.current
    if (prev && prev !== sessionId && ref.current) {
      scrollPosRef.current[prev] = ref.current.scrollTop
    }
    prevSessionIdRef.current = sessionId ?? null
  }, [sessionId])

  // loading: true → false 表示新会话消息已就绪 → 恢复缓存或贴底
  // 注意：放在"自动滚到底" useEffect 之后声明，确保它后执行从而能覆盖滚动位置
  useEffect(() => {
    if (loading) return
    if (!sessionId) return
    const el = ref.current
    if (!el) return
    const cached = scrollPosRef.current[sessionId]
    if (cached !== undefined) {
      el.scrollTo({ top: cached })
      const distance = el.scrollHeight - cached - el.clientHeight
      setStickToBottom(distance <= BOTTOM_THRESHOLD_PX)
    } else {
      el.scrollTo({ top: el.scrollHeight })
      setStickToBottom(true)
    }
  }, [loading, sessionId])

  return (
    <div className="flex-1 min-h-0 relative">
      <div ref={ref} className="absolute inset-0 overflow-y-auto px-6 py-6">
        <div className="max-w-3xl mx-auto space-y-4">
          {loading && <SkeletonRows />}
          {!loading && messages.length === 0 && (
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
          {!loading &&
            messages.map((m, i) => (
              <MessageRow
                key={i}
                m={m}
                index={i}
                isLastAssistant={
                  m.role === "assistant" && i === messages.length - 1 && !m.pending
                }
                onFeedback={onFeedback}
                onRegenerate={onRegenerate}
                onEditAndResend={onEditAndResend}
                onPickSuggestion={onPickSuggestion}
                onOpenImage={setLightboxSrc}
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
      <Lightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />
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
  /** 是否为对话中最后一条已完成的助手消息：仅它显示"建议追问"chip */
  isLastAssistant?: boolean
  onFeedback?: (messageId: string, next: "like" | "dislike" | "cancel") => void
  onRegenerate?: (assistantMessageId: string) => void
  onEditAndResend?: (userMessageIndex: number, newText: string) => void
  onPickSuggestion?: (text: string) => void
  /** 点击图片缩略图：把放大查看委托给上层 Lightbox */
  onOpenImage?: (src: string) => void
}

function MessageRow({
  m,
  index,
  isLastAssistant,
  onFeedback,
  onRegenerate,
  onEditAndResend,
  onPickSuggestion,
  onOpenImage,
}: RowProps) {
  const [thinkOpen, setThinkOpen] = useState(false)
  // 流式思考块的展开状态：null=跟随默认（无 content 时展开，有 content 时折叠）；
  // boolean=用户主动覆盖。
  const [streamThinkOverride, setStreamThinkOverride] = useState<boolean | null>(null)
  const [editing, setEditing] = useState(false)
  const [editText, setEditText] = useState(m.content || "")
  // 流式期间的"实时秒数"：由 setInterval 在 callback 内 setState 更新，
  // render 不调 performance.now() 这类 impure 函数，避免 react-hooks/purity 报错
  const [liveThinkSec, setLiveThinkSec] = useState<number | null>(null)
  const isUser = m.role === "user"
  const hasContent = !!m.content
  // 流式期间是否正在显示思考过程（pending 且已收到至少一段 thinking）
  const isStreamingThinking =
    !isUser && !!m.pending && !!m.streaming_thinking

  // 仅在"思考已开始 + 尚未结束"时启动 ticker；callback 内 setState 不触发
  // react-hooks/set-state-in-effect 规则
  useEffect(() => {
    const startedAt = m.thinking_started_at
    const finalDur = m.thinking_duration_ms
    if (startedAt === undefined || finalDur !== undefined) {
      return
    }
    const id = window.setInterval(() => {
      setLiveThinkSec((performance.now() - startedAt) / 1000)
    }, 200)
    return () => window.clearInterval(id)
  }, [m.thinking_started_at, m.thinking_duration_ms])

  const finalThinkSec =
    m.thinking_duration_ms !== undefined ? m.thinking_duration_ms / 1000 : null
  const isStillThinking =
    m.thinking_started_at !== undefined && m.thinking_duration_ms === undefined
  const thinkSecLabel =
    finalThinkSec !== null
      ? `${finalThinkSec.toFixed(1)}s`
      : isStillThinking && liveThinkSec !== null
        ? `${liveThinkSec.toFixed(1)}s`
        : null
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
            <span>
              {streamThinkOpen
                ? finalThinkSec !== null
                  ? `思考过程 · ${thinkSecLabel}`
                  : isStillThinking
                    ? thinkSecLabel
                      ? `思考中 · ${thinkSecLabel}`
                      : "思考中…"
                    : "思考过程"
                : finalThinkSec !== null
                  ? `已思考 ${thinkSecLabel}，点击查看`
                  : "已思考，点击查看"}
            </span>
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
              <span>
                {finalThinkSec !== null ? `思考过程 · ${thinkSecLabel}` : "思考过程"}
              </span>
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
                <button
                  key={idx}
                  type="button"
                  onClick={() => onOpenImage?.(src)}
                  className="block p-0 border-0 bg-transparent cursor-zoom-in"
                  aria-label="放大查看图片"
                  title="点击放大"
                >
                  <img
                    src={src}
                    alt={`uploaded-${idx}`}
                    className="max-w-[180px] max-h-[180px] rounded-lg border border-white/30 transition-transform hover:scale-[1.02]"
                  />
                </button>
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
          {isLastAssistant && hasContent && onPickSuggestion && (
            <FollowUpChips state={m.state} onPick={onPickSuggestion} />
          )}
          <TraceList traces={m.tool_traces} />
        </>
      )}
    </div>
  )
}

function FollowUpChips({
  state,
  onPick,
}: {
  state?: ChatState
  onPick: (text: string) => void
}) {
  const items = buildFollowUps(state)
  if (items.length === 0) return null
  return (
    <div className="mt-1.5 max-w-[82%] px-1">
      <div className="flex items-center gap-1 text-[11px] text-[hsl(var(--muted-foreground))] mb-1">
        <Lightbulb className="w-3 h-3 text-amber-500" />
        <span>可以这样接着问</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {items.map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => onPick(t)}
            className="text-left text-xs px-2.5 py-1 rounded-full border border-[hsl(var(--border))] bg-white text-[hsl(var(--ink-700,#4D3D3A))] hover:border-[hsl(var(--primary)/0.4)] hover:bg-[hsl(var(--primary)/0.04)] hover:text-[hsl(var(--primary))] transition-colors"
            title={t}
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  )
}

/** 切换会话期间的占位骨架，避免空白闪烁 */
function SkeletonRows() {
  const rows: Array<{ side: "left" | "right"; w: string; h: string }> = [
    { side: "right", w: "w-48", h: "h-12" },
    { side: "left", w: "w-72", h: "h-20" },
    { side: "right", w: "w-40", h: "h-10" },
    { side: "left", w: "w-80", h: "h-24" },
  ]
  return (
    <div className="space-y-4" aria-busy="true" aria-label="正在加载会话">
      {rows.map((s, i) => (
        <div
          key={i}
          className={cn("flex", s.side === "right" ? "justify-end" : "justify-start")}
        >
          <div
            className={cn(
              "rounded-2xl bg-[hsl(var(--accent))] animate-pulse",
              s.w,
              s.h,
            )}
          />
        </div>
      ))}
    </div>
  )
}
