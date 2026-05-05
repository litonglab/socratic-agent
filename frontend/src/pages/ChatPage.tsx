import { useEffect, useState, useCallback, useRef, useMemo } from "react"
import { useNavigate } from "react-router-dom"
import { PanelLeftOpen } from "lucide-react"
import { useHotkeys, type HotkeyBinding } from "@/hooks/useHotkeys"
import {
  chatStream,
  listSessions,
  deleteSession as apiDeleteSession,
  fetchSessionMessages,
  submitFeedback,
  archiveSession,
  unarchiveSession,
  renameSession as apiRenameSession,
  type SessionMeta,
} from "@/lib/api"
import { toast } from "@/lib/toast"
import type { AuthState } from "@/hooks/useAuth"
import Sidebar from "@/components/Sidebar"
import MessageList, { type ChatMessage, type FeedbackValue } from "@/components/MessageList"
import ChatInput, { type AttachedImage } from "@/components/ChatInput"
import { cn } from "@/lib/utils"

const SIDEBAR_KEY = "netruc_sidebar_open"

function readSidebarOpen(): boolean {
  if (typeof window === "undefined") return true
  const v = window.localStorage.getItem(SIDEBAR_KEY)
  if (v === null) {
    // 首次访问：移动端默认收起，桌面端默认展开
    return !window.matchMedia("(max-width: 767px)").matches
  }
  return v === "1"
}

interface Props {
  auth: AuthState
}

export default function ChatPage({ auth }: Props) {
  const navigate = useNavigate()
  const [sessions, setSessions] = useState<SessionMeta[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [streaming, setStreaming] = useState(false)
  const [messagesLoading, setMessagesLoading] = useState(false)
  const [websearch, setWebsearch] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(readSidebarOpen)
  // 流式请求的 AbortController：切换会话 / 卸载 / 用户主动停止时取消
  const abortRef = useRef<AbortController | null>(null)
  // 侧栏搜索框 ref：用于 Cmd/Ctrl+K 聚焦
  const sidebarSearchRef = useRef<HTMLInputElement | null>(null)

  const cancelStream = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
  }, [])

  useEffect(() => {
    return () => {
      // 组件卸载时也取消，避免悬挂请求
      if (abortRef.current) abortRef.current.abort()
    }
  }, [])

  useEffect(() => {
    try {
      window.localStorage.setItem(SIDEBAR_KEY, sidebarOpen ? "1" : "0")
    } catch {
      // 忽略 storage 错误（如隐私模式）
    }
  }, [sidebarOpen])

  // 未登录跳回 login
  useEffect(() => {
    if (!auth.loading && !auth.user) navigate("/login")
  }, [auth.loading, auth.user, navigate])

  // 注意：deps 故意为空。否则 activeId 一变 reloadSessions 就重建，
  // useEffect[reloadSessions] 会在流式期间触发；那时新 session 还没写库，
  // 老逻辑会判定"列表里没有 active"而把 messages / activeId 清空 → 出现
  // "发完消息停留在首页，得点击会话才看到回答"的 bug。
  // 真正"列表里没有该 session 就清空"的需求只在用户主动删除时存在，
  // 已经放在 deleteSess() 里处理。
  const reloadSessions = useCallback(async () => {
    try {
      const { sessions } = await listSessions()
      setSessions(sessions || [])
    } catch (e) {
      console.warn("[ChatPage] listSessions failed:", e)
    }
  }, [])

  useEffect(() => {
    if (!auth.user) return
    let cancelled = false
    ;(async () => {
      try {
        const { sessions } = await listSessions()
        if (!cancelled) setSessions(sessions || [])
      } catch (e) {
        console.warn("[ChatPage] listSessions failed:", e)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [auth.user])

  async function selectSession(id: string) {
    if (id === activeId) return
    // 切换会话时先取消正在进行的流式请求，避免新会话被旧 stream 写脏
    cancelStream()
    setStreaming(false)
    setActiveId(id)
    setMessagesLoading(true)
    try {
      const data = await fetchSessionMessages(id)
      setMessages(
        (data.messages || []).map((m) => ({
          role: m.role === "user" ? "user" : "assistant",
          content: m.content || "",
          thinking: m.thinking || undefined,
          message_id: m.message_id ?? null,
          feedback: (m.feedback as FeedbackValue) ?? null,
          images: m.image_b64?.map((b) => `data:image/png;base64,${b}`),
        })),
      )
    } catch (e) {
      console.warn("[ChatPage] fetchSessionMessages failed:", e)
      setMessages([])
    } finally {
      setMessagesLoading(false)
    }
  }

  const newSession = useCallback(() => {
    cancelStream()
    setStreaming(false)
    setActiveId(null)
    setMessages([])
  }, [cancelStream])

  // 全局快捷键：
  //  - Cmd/Ctrl+B    折叠/展开侧栏
  //  - Cmd/Ctrl+⇧+O  新建会话（避开浏览器原生 Cmd+N）
  //  - Cmd/Ctrl+K    展开侧栏并聚焦搜索框（在输入框内也允许触发）
  const hotkeys = useMemo<HotkeyBinding[]>(
    () => [
      {
        key: "b",
        meta: true,
        handler: () => setSidebarOpen((v) => !v),
      },
      {
        key: "o",
        meta: true,
        shift: true,
        handler: () => newSession(),
      },
      {
        key: "k",
        meta: true,
        allowInInput: true,
        handler: () => {
          setSidebarOpen(true)
          requestAnimationFrame(() => {
            sidebarSearchRef.current?.focus()
            sidebarSearchRef.current?.select()
          })
        },
      },
    ],
    [newSession],
  )
  useHotkeys(hotkeys)

  async function deleteSess(id: string) {
    try {
      await apiDeleteSession(id)
      toast.success("会话已删除")
    } catch (e) {
      console.warn("[ChatPage] delete failed:", e)
      toast.error(`删除失败：${(e as Error).message}`)
    }
    if (id === activeId) {
      setActiveId(null)
      setMessages([])
    }
    void reloadSessions()
  }

  async function archiveSess(id: string) {
    try {
      await archiveSession(id)
      toast.info("已归档")
    } catch (e) {
      console.warn("[ChatPage] archive failed:", e)
      toast.error(`归档失败：${(e as Error).message}`)
    }
    void reloadSessions()
  }

  async function unarchiveSess(id: string) {
    try {
      await unarchiveSession(id)
      toast.info("已恢复")
    } catch (e) {
      console.warn("[ChatPage] unarchive failed:", e)
      toast.error(`恢复失败：${(e as Error).message}`)
    }
    void reloadSessions()
  }

  async function renameSess(id: string, title: string) {
    // 乐观更新：先改本地，失败则回滚 + toast
    setSessions((prev) =>
      prev.map((s) => (s.session_id === id ? { ...s, title } : s)),
    )
    try {
      await apiRenameSession(id, title)
      toast.success("已重命名")
    } catch (e) {
      console.warn("[ChatPage] rename failed:", e)
      toast.error(`重命名失败：${(e as Error).message}`)
      void reloadSessions()
    }
  }

  /**
   * 重新生成：找到该 assistant 消息对应的上一条 user 消息，
   * 截断 stored_history 到 user 之前，再用同样的 user message 重发。
   */
  async function handleRegenerate(assistantMessageId: string) {
    if (streaming) return
    let assistantIdx = -1
    let userIdx = -1
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].message_id === assistantMessageId) {
        assistantIdx = i
        break
      }
    }
    if (assistantIdx < 0) return
    for (let i = assistantIdx - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        userIdx = i
        break
      }
    }
    if (userIdx < 0) return
    const userMsg = messages[userIdx]
    setMessages(messages.slice(0, userIdx))
    await send(userMsg.content || "", [], { truncateTo: userIdx })
  }

  /**
   * 编辑用户消息后重发：截断到该用户消息之前，使用新文本重发。
   * 旧的图片附件不带过去（用户编辑文字常意味着重启上下文）。
   */
  async function handleEditAndResend(userMessageIndex: number, newText: string) {
    if (streaming) return
    if (userMessageIndex < 0 || userMessageIndex >= messages.length) return
    if (messages[userMessageIndex].role !== "user") return
    setMessages(messages.slice(0, userMessageIndex))
    await send(newText, [], { truncateTo: userMessageIndex })
  }

  async function onFeedback(messageId: string, next: "like" | "dislike" | "cancel") {
    if (!activeId) return
    try {
      await submitFeedback({ session_id: activeId, message_id: messageId, feedback: next })
    } catch (e) {
      console.warn("[ChatPage] feedback failed:", e)
      return
    }
    setMessages((prev) =>
      prev.map((m) =>
        m.message_id === messageId
          ? { ...m, feedback: next === "cancel" ? null : (next as FeedbackValue) }
          : m,
      ),
    )
  }

  async function send(
    text: string,
    images: AttachedImage[],
    opts: { truncateTo?: number } = {},
  ) {
    if (streaming) return
    const userMsg: ChatMessage = {
      role: "user",
      content: text,
      images: images.map((i) => i.dataUrl),
    }
    const assistantPending: ChatMessage = { role: "assistant", content: "", pending: true }
    setMessages((prev) => [...prev, userMsg, assistantPending])
    setStreaming(true)

    const ctrl = new AbortController()
    abortRef.current = ctrl

    try {
      let resolvedSid: string | null = activeId
      let bufVisible = ""
      let respMessageId: string | null = null
      let respThinking = ""
      for await (const ev of chatStream(
        {
          message: text,
          session_id: activeId,
          enable_websearch: websearch,
          truncate_history_to: opts.truncateTo,
          images: images.length
            ? images.map((i) => ({ base64: i.base64, mime: i.mime }))
            : undefined,
        },
        ctrl.signal,
      )) {
        // 若用户切换到了别的会话，发起此次请求时的 sid 与当前 active 不一致，
        // 这次流式产物已不再属于当前视图，丢弃即可（实际请求也已被 abort）
        if (ctrl.signal.aborted) break
        if (ev.type === "meta") {
          resolvedSid = ev.session_id || resolvedSid
          respMessageId = ev.message_id || respMessageId
          if (resolvedSid && resolvedSid !== activeId) {
            setActiveId(resolvedSid)
          }
        } else if (ev.type === "stage") {
          // 流式阶段提示：analyzing / tools / generating
          setMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            if (last && last.role === "assistant" && last.pending) {
              next[next.length - 1] = {
                ...last,
                stage: ev.stage,
                stage_tools: ev.tools,
              }
            }
            return next
          })
        } else if (ev.type === "thinking_delta" && ev.content) {
          // 流式思考增量：累积到 streaming_thinking，气泡上方斜体浅色显示
          // 第一次到达即记录 thinking_started_at（performance.now 时间戳）
          setMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            if (last && last.role === "assistant" && last.pending) {
              next[next.length - 1] = {
                ...last,
                streaming_thinking: (last.streaming_thinking || "") + ev.content,
                thinking_started_at:
                  last.thinking_started_at ?? performance.now(),
              }
            }
            return next
          })
        } else if (ev.type === "token" && ev.content) {
          bufVisible += ev.content
          setMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            if (last && last.role === "assistant") {
              // 思考阶段结束、首次进入正文 → 记录 thinking_duration_ms
              const isFirstToken = !last.content
              const finalDur =
                last.thinking_duration_ms ??
                (isFirstToken && last.thinking_started_at !== undefined
                  ? performance.now() - last.thinking_started_at
                  : undefined)
              next[next.length - 1] = {
                ...last,
                content: bufVisible,
                pending: true,
                stage: "generating",
                thinking_duration_ms: finalDur,
              }
            }
            return next
          })
        } else if (ev.type === "done") {
          const finalReply = ev.reply || bufVisible
          respThinking = ev.thinking || ""
          setMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            if (last && last.role === "assistant") {
              // 兜底：done 之前可能完全没有 token（仅 thinking → done）
              const finalDur =
                last.thinking_duration_ms ??
                (last.thinking_started_at !== undefined
                  ? performance.now() - last.thinking_started_at
                  : undefined)
              next[next.length - 1] = {
                role: "assistant",
                content: finalReply,
                thinking: respThinking || undefined,
                message_id: respMessageId,
                feedback: null,
                tool_traces: ev.tool_traces,
                state: ev.state,
                thinking_started_at: last.thinking_started_at,
                thinking_duration_ms: finalDur,
              }
            }
            return next
          })
        } else if (ev.type === "error") {
          setMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            if (last && last.role === "assistant") {
              next[next.length - 1] = {
                role: "assistant",
                content: `生成失败：${ev.error || "未知错误"}`,
              }
            }
            return next
          })
        }
      }
    } catch (e) {
      // AbortError 是用户主动取消（切换会话 / 停止生成），不展示为"生成失败"
      const err = e as Error & { name?: string }
      if (err?.name === "AbortError") {
        // 取消时把最后一条 pending 的助手消息标记为已停止（若仍是 pending）
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last && last.role === "assistant" && last.pending) {
            next[next.length - 1] = {
              ...last,
              pending: false,
              content: last.content || "（已停止生成）",
            }
          }
          return next
        })
      } else {
        setMessages((prev) => {
          const next = [...prev]
          const last = next[next.length - 1]
          if (last && last.role === "assistant") {
            next[next.length - 1] = {
              role: "assistant",
              content: `生成失败：${err.message || "未知错误"}`,
            }
          }
          return next
        })
      }
    } finally {
      if (abortRef.current === ctrl) abortRef.current = null
      setStreaming(false)
      void reloadSessions()
    }
  }

  if (auth.loading || !auth.user) {
    return (
      <div className="min-h-screen grid place-items-center text-[hsl(var(--muted-foreground))]">
        加载中…
      </div>
    )
  }

  return (
    <div className="h-screen flex overflow-hidden relative">
      {/* sidebar 容器：
           - md+ 屏幕：侧栏内联占位，宽度由 sidebarOpen 控制（折叠动画）
           - <md 屏幕：侧栏脱离文档流变成 overlay drawer，配合遮罩 */}
      <div
        className={cn(
          "shrink-0 overflow-hidden transition-[width] duration-200 ease-out",
          "max-md:absolute max-md:inset-y-0 max-md:left-0 max-md:z-40 max-md:shadow-2xl",
          sidebarOpen ? "w-72" : "w-0",
        )}
      >
        <div className="w-72 h-full">
          <Sidebar
            user={auth.user}
            sessions={sessions}
            activeId={activeId}
            searchInputRef={sidebarSearchRef}
            onNew={() => {
              newSession()
              if (window.matchMedia("(max-width: 767px)").matches) {
                setSidebarOpen(false)
              }
            }}
            onSelect={(id) => {
              void selectSession(id)
              if (window.matchMedia("(max-width: 767px)").matches) {
                setSidebarOpen(false)
              }
            }}
            onDelete={deleteSess}
            onArchive={archiveSess}
            onUnarchive={unarchiveSess}
            onRename={renameSess}
            onCollapse={() => setSidebarOpen(false)}
            onLogout={() => {
              auth.logout()
              navigate("/login")
            }}
          />
        </div>
      </div>

      {/* 移动端 drawer 遮罩 */}
      {sidebarOpen && (
        <div
          className="md:hidden absolute inset-0 z-30 bg-black/30 backdrop-blur-[1px]"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      <main className="flex-1 min-w-0 flex flex-col relative bg-[#FAF6F4]/40">
        {!sidebarOpen && (
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            className="absolute top-3 left-3 z-30 grid place-items-center w-8 h-8 rounded-md bg-white border border-[hsl(var(--border))] text-[hsl(var(--muted-foreground))] hover:bg-[hsl(var(--accent))] hover:text-[hsl(var(--primary))] shadow-sm"
            title="展开侧栏"
            aria-label="展开侧栏"
          >
            <PanelLeftOpen className="w-4 h-4" />
          </button>
        )}
        <MessageList
          messages={messages}
          sessionId={activeId}
          loading={messagesLoading}
          onFeedback={onFeedback}
          onRegenerate={handleRegenerate}
          onEditAndResend={handleEditAndResend}
          onPickSuggestion={(t) => void send(t, [])}
        />
        <ChatInput
          disabled={streaming}
          streaming={streaming}
          websearch={websearch}
          onWebsearchChange={setWebsearch}
          onSend={(t, imgs) => void send(t, imgs)}
          onStop={cancelStream}
        />
      </main>
    </div>
  )
}
