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
import ChatInput, { type Attachment } from "@/components/ChatInput"
import { cn } from "@/lib/utils"

const SIDEBAR_KEY = "netruc_sidebar_open"

/**
 * 单个会话的运行时状态。每个 sid 一份，独立持有自己的 messages、streaming
 * 标志和 AbortController；切换会话时不再清空这份状态，从而支持后台多会话并发。
 */
type SessionRuntime = {
  messages: ChatMessage[]
  streaming: boolean
  abort: AbortController | null
}

// 草稿会话占位 sid：用户尚未发出第一条消息、还没拿到后端 session_id 时使用。
// 收到首个 meta 事件后会被 swap 成真实 sid。
function makeDraftId(): string {
  return `__draft_${Math.random().toString(36).slice(2, 10)}__`
}

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
  // 当前活跃会话的 messages 投影；runtimeRef 才是 source of truth。
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [messagesLoading, setMessagesLoading] = useState(false)
  // 哪些 sid 正在流式生成 → Sidebar 渲染右侧转圈。
  const [streamingSet, setStreamingSet] = useState<Set<string>>(() => new Set())
  // 哪些 sid 在用户离开期间完成了生成、尚未被查看 → Sidebar 红点提示。
  const [unseenSet, setUnseenSet] = useState<Set<string>>(() => new Set())
  const [websearch, setWebsearch] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(readSidebarOpen)
  // 各 sid 的运行时状态（messages / streaming / abort），跨切换不丢失。
  const runtimeRef = useRef<Map<string, SessionRuntime>>(new Map())
  // activeId 的同步 ref：异步 SSE 回调里读取最新值，避免闭包陈旧。
  const activeIdRef = useRef<string | null>(null)
  // 侧栏搜索框 ref：用于 Cmd/Ctrl+K 聚焦
  const sidebarSearchRef = useRef<HTMLInputElement | null>(null)

  // 当前 active 会话是否在 streaming（衍生量）：决定 ChatInput 的 disabled / Stop 按钮。
  const streaming = activeId !== null && streamingSet.has(activeId)

  // 仅停掉当前 active 会话的流；其他后台跑的会话不受影响。
  const cancelActiveStream = useCallback(() => {
    const sid = activeIdRef.current
    if (!sid) return
    const rt = runtimeRef.current.get(sid)
    if (rt?.abort) {
      rt.abort.abort()
      rt.abort = null
    }
  }, [])

  useEffect(() => {
    activeIdRef.current = activeId
  }, [activeId])

  useEffect(() => {
    const runtime = runtimeRef.current
    return () => {
      // 组件卸载：取消所有还在跑的会话流，避免悬挂请求。
      runtime.forEach((rt) => {
        if (rt.abort) rt.abort.abort()
      })
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

  // 切换会话：不再 cancelStream，旧会话保留在 runtimeRef 里继续后台跑。
  async function selectSession(id: string) {
    if (id === activeId) return
    // 把当前 active 的 messages 状态写回 ref，免得切回时丢失最近增量。
    if (activeId) {
      const rt = runtimeRef.current.get(activeId)
      if (rt) rt.messages = messages
    }
    setActiveId(id)
    activeIdRef.current = id
    // 进入该会话时清 unseen 红点
    setUnseenSet((prev) => {
      if (!prev.has(id)) return prev
      const next = new Set(prev)
      next.delete(id)
      return next
    })
    // 切到内存中已有 runtime 的会话：直接投影，不重新拉历史。
    const cached = runtimeRef.current.get(id)
    if (cached) {
      setMessages(cached.messages)
      setMessagesLoading(false)
      return
    }
    setMessagesLoading(true)
    try {
      const data = await fetchSessionMessages(id)
      const msgs: ChatMessage[] = (data.messages || []).map((m) => ({
        role: m.role === "user" ? "user" : "assistant",
        content: m.content || "",
        thinking: m.thinking || undefined,
        message_id: m.message_id ?? null,
        feedback: (m.feedback as FeedbackValue) ?? null,
        // 后端持久化的是压缩后的 JPEG 缩略图；浏览器对 dataUrl mime 不严格但仍写正确值
        images: m.image_b64?.map((b) => `data:image/jpeg;base64,${b}`),
        files: m.files,
      }))
      setMessages(msgs)
      // 同步进 ref：保证后续切走再回来仍能保留视图
      runtimeRef.current.set(id, { messages: msgs, streaming: false, abort: null })
    } catch (e) {
      console.warn("[ChatPage] fetchSessionMessages failed:", e)
      setMessages([])
    } finally {
      setMessagesLoading(false)
    }
  }

  // 新建会话：保留所有后台跑的会话不动，仅把当前 active 的 messages 写回 ref。
  const newSession = useCallback(() => {
    const cur = activeIdRef.current
    if (cur) {
      const rt = runtimeRef.current.get(cur)
      if (rt) rt.messages = messages
    }
    setActiveId(null)
    activeIdRef.current = null
    setMessages([])
  }, [messages])

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
    // 同时停掉该会话还在跑的流（如有），并清理 runtime / 标志位缓存
    const rt = runtimeRef.current.get(id)
    if (rt?.abort) rt.abort.abort()
    runtimeRef.current.delete(id)
    setStreamingSet((prev) => {
      if (!prev.has(id)) return prev
      const next = new Set(prev)
      next.delete(id)
      return next
    })
    setUnseenSet((prev) => {
      if (!prev.has(id)) return prev
      const next = new Set(prev)
      next.delete(id)
      return next
    })
    if (id === activeId) {
      setActiveId(null)
      activeIdRef.current = null
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
   * 旧的附件不带过去（用户编辑文字常意味着重启上下文）。
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
    attachments: Attachment[],
    opts: { truncateTo?: number } = {},
  ) {
    // 多会话并发：以"该 sid 是否正在跑"为准，不再用全局 streaming 拦截。
    const initialActiveId = activeIdRef.current
    // 草稿状态发首条消息时使用占位 sid；收到 meta 后会通过 realSid 切换到真实 sid。
    const sid = initialActiveId ?? makeDraftId()
    const existingRt = runtimeRef.current.get(sid)
    if (existingRt?.streaming) return

    const imageAttachments = attachments.filter((a) => a.kind === "image")
    const fileAttachments = attachments.filter((a) => a.kind === "file")
    const userMsg: ChatMessage = {
      role: "user",
      content: text,
      images: imageAttachments
        .map((i) => i.dataUrl)
        .filter((u): u is string => !!u),
      files: fileAttachments.length
        ? fileAttachments.map((f) => ({ name: f.name, size: f.size }))
        : undefined,
    }
    const assistantPending: ChatMessage = { role: "assistant", content: "", pending: true }

    // baseMessages：该 sid 的历史 messages（active 状态用 messages state；草稿则用空数组）
    const baseMessages =
      initialActiveId === sid
        ? messages
        : runtimeRef.current.get(sid)?.messages ?? []
    const newMessages = [...baseMessages, userMsg, assistantPending]

    const ctrl = new AbortController()
    runtimeRef.current.set(sid, {
      messages: newMessages,
      streaming: true,
      abort: ctrl,
    })

    // 草稿会话：立即把 activeId 切到占位 sid，让 streamingSet/输入 disable 都能基于它工作。
    if (!initialActiveId) {
      setActiveId(sid)
      activeIdRef.current = sid
    }
    if (activeIdRef.current === sid) {
      setMessages(newMessages)
    }
    setStreamingSet((prev) => {
      const next = new Set(prev)
      next.add(sid)
      return next
    })

    // 真实 sid（meta 事件会带回）。所有 SSE 回调通过它写 ref。
    let realSid = sid
    let bufVisible = ""
    let respMessageId: string | null = null
    let respThinking = ""

    // 写 ref + 条件性同步 UI：仅当该 sid 仍是当前 active 时才 setMessages。
    const updateMessages = (updater: (prev: ChatMessage[]) => ChatMessage[]) => {
      const rt = runtimeRef.current.get(realSid)
      if (!rt) return
      const next = updater(rt.messages)
      rt.messages = next
      if (activeIdRef.current === realSid) {
        setMessages(next)
      }
    }

    try {
      for await (const ev of chatStream(
        {
          message: text,
          session_id: initialActiveId, // 后端语义：null/空 → 新建会话
          enable_websearch: websearch,
          truncate_history_to: opts.truncateTo,
          images: imageAttachments.length
            ? imageAttachments.map((i) => ({ base64: i.base64, mime: i.mime }))
            : undefined,
          files: fileAttachments.length
            ? fileAttachments.map((f) => ({
                name: f.name,
                base64: f.base64,
                mime: f.mime,
              }))
            : undefined,
        },
        ctrl.signal,
      )) {
        if (ctrl.signal.aborted) break
        if (ev.type === "meta") {
          const newSid = ev.session_id || realSid
          respMessageId = ev.message_id || respMessageId
          // 草稿 → 真实 sid 的迁移：把 runtime / streamingSet / activeId 三处 key 同步换掉
          if (newSid !== realSid) {
            const rt = runtimeRef.current.get(realSid)
            if (rt) {
              runtimeRef.current.set(newSid, rt)
              runtimeRef.current.delete(realSid)
            }
            setStreamingSet((prev) => {
              if (!prev.has(realSid)) return prev
              const next = new Set(prev)
              next.delete(realSid)
              next.add(newSid)
              return next
            })
            if (activeIdRef.current === realSid) {
              setActiveId(newSid)
              activeIdRef.current = newSid
            }
            realSid = newSid
          }
          // 乐观把新会话立即插入侧栏：让用户即使切到草稿/别的会话，
          // 也能在 Sidebar 看到这条还在跑的会话（带转圈）。
          // 标题先用问题前 30 字占位，等 done 后 reloadSessions 会用后端摘要标题覆盖。
          setSessions((prev) => {
            if (prev.some((s) => s.session_id === realSid)) return prev
            const placeholderTitle = text.trim().slice(0, 30) || "新会话"
            return [
              {
                session_id: realSid,
                title: placeholderTitle,
                updated_at: new Date().toISOString(),
              },
              ...prev,
            ]
          })
        } else if (ev.type === "stage") {
          // 流式阶段提示：analyzing / tools / generating
          updateMessages((prev) => {
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
          updateMessages((prev) => {
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
          updateMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            if (last && last.role === "assistant") {
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
          updateMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            if (last && last.role === "assistant") {
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
          updateMessages((prev) => {
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
      const err = e as Error & { name?: string }
      if (err?.name === "AbortError") {
        // 用户主动停止：保留已生成的内容，标记 pending 关闭
        updateMessages((prev) => {
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
        updateMessages((prev) => {
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
      // 清掉该 sid 的 streaming + abort 句柄
      const rt = runtimeRef.current.get(realSid)
      if (rt) {
        rt.streaming = false
        rt.abort = null
      }
      setStreamingSet((prev) => {
        if (!prev.has(realSid)) return prev
        const next = new Set(prev)
        next.delete(realSid)
        return next
      })
      // 后台完成：用户已切走 → 标 unseen 红点 + toast 提示
      const wasAborted = ctrl.signal.aborted
      if (!wasAborted && activeIdRef.current !== realSid) {
        setUnseenSet((prev) => {
          if (prev.has(realSid)) return prev
          const next = new Set(prev)
          next.add(realSid)
          return next
        })
        const sessionTitle =
          sessions.find((s) => s.session_id === realSid)?.title || "新会话"
        toast.success(`「${sessionTitle}」已生成完成`)
      }
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
            streamingSessions={streamingSet}
            unseenSessions={unseenSet}
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
          onStop={cancelActiveStream}
        />
      </main>
    </div>
  )
}
