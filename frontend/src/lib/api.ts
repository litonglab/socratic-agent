// FastAPI 客户端：登录 / 注册 / sessions / 流式聊天等
// 通过 vite proxy 走到 http://localhost:8000

const TOKEN_KEY = "netruc_auth_token"

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string | null) {
  if (token) localStorage.setItem(TOKEN_KEY, token)
  else localStorage.removeItem(TOKEN_KEY)
}

function authHeaders(): HeadersInit {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

async function request<T>(url: string, opts: RequestInit = {}): Promise<T> {
  const res = await fetch(url, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
      ...(opts.headers || {}),
    },
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const data = await res.json()
      if (data?.detail) detail = String(data.detail)
    } catch {
      // ignore parse error
    }
    throw new Error(detail)
  }
  if (res.status === 204) return undefined as T
  return (await res.json()) as T
}

// ===== Auth =====

export interface UserProfile {
  name?: string
  student_id?: string
  nickname?: string
  class_name?: string
  email?: string
}

export interface AuthUser {
  id?: string
  username: string
  profile?: UserProfile
}

export interface AuthResponse {
  token: string
  user: AuthUser
}

export async function login(username: string, password: string): Promise<AuthResponse> {
  return request<AuthResponse>("/api/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  })
}

export async function register(payload: {
  username: string
  password: string
  name: string
  student_id: string
  nickname: string
  class_name: string
  email: string
}): Promise<AuthResponse> {
  return request<AuthResponse>("/api/register", {
    method: "POST",
    body: JSON.stringify(payload),
  })
}

export async function fetchMe(): Promise<AuthUser & { preferences?: Record<string, unknown> }> {
  return request("/api/me")
}

// ===== Sessions =====

export interface SessionMeta {
  session_id: string
  title: string
  archived?: boolean
  updated_at?: string
  created_at?: string
}

export async function listSessions(): Promise<{ sessions: SessionMeta[] }> {
  return request("/api/sessions")
}

export async function deleteSession(sessionId: string): Promise<void> {
  await request(`/api/sessions/${sessionId}`, { method: "DELETE" })
}

export async function archiveSession(sessionId: string) {
  return request(`/api/sessions/${sessionId}/archive`, { method: "POST" })
}

export async function unarchiveSession(sessionId: string) {
  return request(`/api/sessions/${sessionId}/unarchive`, { method: "POST" })
}

export async function fetchSessionMessages(sessionId: string): Promise<{
  session_id: string
  messages: Array<{
    role: string
    content: string
    message_id?: string | null
    thinking?: string
    image_b64?: string[]
    feedback?: string | null
  }>
}> {
  return request(`/api/sessions/${sessionId}`)
}

// ===== Feedback =====

export async function submitFeedback(payload: {
  session_id: string
  message_id: string
  feedback: "like" | "dislike" | "cancel"
}) {
  return request("/api/feedback", { method: "POST", body: JSON.stringify(payload) })
}

// ===== Streaming chat (SSE-style line-delimited JSON) =====

export interface StreamEvent {
  type: "meta" | "token" | "done" | "error"
  // meta
  message_id?: string
  session_id?: string
  // token
  content?: string
  // done
  reply?: string
  thinking?: string
  tool_traces?: Array<{ tool?: string; input?: string; output?: string }>
  // error
  error?: string
}

export interface ChatImageInput {
  base64: string
  mime?: string
}

export interface ChatStreamRequest {
  message: string
  session_id?: string | null
  enable_websearch?: boolean
  max_turns?: number
  history?: null | Array<{ role: string; content: string }>
  images?: ChatImageInput[]
}

/**
 * 调用 /api/chat/stream，按 SSE 协议解析事件流。
 * 后端格式：
 *   event: <name>
 *   data: {...json...}
 *   <空行>
 *
 * name 可能是 meta / delta / done / error。我们把它映射到统一的
 * StreamEvent { type: name, ...data }。
 *
 * 返回 async generator，调用方用 for-await-of 消费。
 */
export async function* chatStream(req: ChatStreamRequest): AsyncGenerator<StreamEvent> {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...authHeaders(),
    },
    body: JSON.stringify(req),
  })
  if (!res.ok || !res.body) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const data = await res.json()
      if (data?.detail) detail = String(data.detail)
    } catch {
      // ignore
    }
    throw new Error(detail)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder("utf-8")
  let buffer = ""

  function* parseBlocks(buf: string): Generator<StreamEvent> {
    // SSE 事件以两个换行分隔
    const blocks = buf.split(/\r?\n\r?\n/)
    // 注意：最后一个 block 可能不完整，由调用方处理；这里 generator 只产出已完整的
    for (let i = 0; i < blocks.length - 1; i++) {
      const block = blocks[i]
      if (!block.trim()) continue
      let evName = "message"
      const dataLines: string[] = []
      for (const rawLine of block.split(/\r?\n/)) {
        const line = rawLine.replace(/^\uFEFF/, "")
        if (line.startsWith("event:")) {
          evName = line.slice(6).trim()
        } else if (line.startsWith("data:")) {
          dataLines.push(line.slice(5).trim())
        }
        // 注释行（以 ":" 开头）和其它字段忽略
      }
      const dataStr = dataLines.join("\n")
      let data: Record<string, unknown> = {}
      if (dataStr) {
        try {
          data = JSON.parse(dataStr)
        } catch (err) {
          console.warn("[chatStream] failed to parse data:", dataStr, err)
        }
      }
      // 兼容：后端把 token 增量叫做 "delta"，前端历史代码判断的是 "token"
      const type = evName === "delta" ? "token" : (evName as StreamEvent["type"])
      // 兼容：后端 error 事件用 detail 字段，前端代码读 error
      const merged: Record<string, unknown> = { ...data }
      if (type === "error" && !merged.error && merged.detail) {
        merged.error = merged.detail
      }
      yield { type, ...merged } as StreamEvent
    }
  }

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    if (buffer.includes("\n\n") || buffer.includes("\r\n\r\n")) {
      const blocks = buffer.split(/\r?\n\r?\n/)
      // 保留最后一个未完成的 chunk
      const incomplete = blocks.pop() || ""
      const consumed = blocks.join("\n\n") + "\n\n"
      for (const ev of parseBlocks(consumed)) {
        yield ev
      }
      buffer = incomplete
    }
  }

  // flush 剩余
  if (buffer.trim()) {
    const flushed = buffer + "\n\n"
    for (const ev of parseBlocks(flushed)) {
      yield ev
    }
  }
}
