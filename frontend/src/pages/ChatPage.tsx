import { useEffect, useState, useCallback } from "react"
import { useNavigate } from "react-router-dom"
import { PanelLeftOpen } from "lucide-react"
import {
  chatStream,
  listSessions,
  deleteSession as apiDeleteSession,
  fetchSessionMessages,
  submitFeedback,
  archiveSession,
  unarchiveSession,
  type SessionMeta,
} from "@/lib/api"
import type { AuthState } from "@/hooks/useAuth"
import Sidebar from "@/components/Sidebar"
import MessageList, { type ChatMessage, type FeedbackValue } from "@/components/MessageList"
import ChatInput, { type AttachedImage } from "@/components/ChatInput"
import { cn } from "@/lib/utils"

const SIDEBAR_KEY = "netruc_sidebar_open"

function readSidebarOpen(): boolean {
  if (typeof window === "undefined") return true
  const v = window.localStorage.getItem(SIDEBAR_KEY)
  if (v === null) return true
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
  const [websearch, setWebsearch] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(readSidebarOpen)

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
    if (auth.user) void reloadSessions()
  }, [auth.user, reloadSessions])

  async function selectSession(id: string) {
    setActiveId(id)
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
    }
  }

  function newSession() {
    setActiveId(null)
    setMessages([])
  }

  async function deleteSess(id: string) {
    try {
      await apiDeleteSession(id)
    } catch (e) {
      console.warn("[ChatPage] delete failed:", e)
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
    } catch (e) {
      console.warn("[ChatPage] archive failed:", e)
    }
    void reloadSessions()
  }

  async function unarchiveSess(id: string) {
    try {
      await unarchiveSession(id)
    } catch (e) {
      console.warn("[ChatPage] unarchive failed:", e)
    }
    void reloadSessions()
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

  async function send(text: string, images: AttachedImage[]) {
    if (streaming) return
    const userMsg: ChatMessage = {
      role: "user",
      content: text,
      images: images.map((i) => i.dataUrl),
    }
    const assistantPending: ChatMessage = { role: "assistant", content: "", pending: true }
    setMessages((prev) => [...prev, userMsg, assistantPending])
    setStreaming(true)

    try {
      let resolvedSid: string | null = activeId
      let bufVisible = ""
      let respMessageId: string | null = null
      let respThinking = ""
      for await (const ev of chatStream({
        message: text,
        session_id: activeId,
        enable_websearch: websearch,
        images: images.length
          ? images.map((i) => ({ base64: i.base64, mime: i.mime }))
          : undefined,
      })) {
        if (ev.type === "meta") {
          resolvedSid = ev.session_id || resolvedSid
          respMessageId = ev.message_id || respMessageId
          if (resolvedSid && resolvedSid !== activeId) {
            setActiveId(resolvedSid)
          }
        } else if (ev.type === "token" && ev.content) {
          bufVisible += ev.content
          setMessages((prev) => {
            const next = [...prev]
            const last = next[next.length - 1]
            if (last && last.role === "assistant") {
              next[next.length - 1] = { ...last, content: bufVisible, pending: true }
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
              next[next.length - 1] = {
                role: "assistant",
                content: finalReply,
                thinking: respThinking || undefined,
                message_id: respMessageId,
                feedback: null,
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
      setMessages((prev) => {
        const next = [...prev]
        const last = next[next.length - 1]
        if (last && last.role === "assistant") {
          next[next.length - 1] = { role: "assistant", content: `生成失败：${(e as Error).message}` }
        }
        return next
      })
    } finally {
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
    <div className="h-screen flex overflow-hidden">
      {/* sidebar 容器：通过宽度+overflow 实现折叠动画 */}
      <div
        className={cn(
          "shrink-0 overflow-hidden transition-[width] duration-200 ease-out",
          sidebarOpen ? "w-72" : "w-0",
        )}
      >
        <div className="w-72 h-full">
          <Sidebar
            user={auth.user}
            sessions={sessions}
            activeId={activeId}
            onNew={newSession}
            onSelect={selectSession}
            onDelete={deleteSess}
            onArchive={archiveSess}
            onUnarchive={unarchiveSess}
            onCollapse={() => setSidebarOpen(false)}
            onLogout={() => {
              auth.logout()
              navigate("/login")
            }}
          />
        </div>
      </div>

      <main className="flex-1 min-w-0 flex flex-col relative">
        {/* sidebar 收起时，主区域左上角浮一个展开按钮（展开状态由 sidebar 内自带的折叠按钮控制） */}
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
        <MessageList messages={messages} onFeedback={onFeedback} />
        <ChatInput
          disabled={streaming}
          websearch={websearch}
          onWebsearchChange={setWebsearch}
          onSend={send}
        />
      </main>
    </div>
  )
}
