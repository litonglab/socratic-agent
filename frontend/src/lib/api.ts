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
 * 调用 /api/chat/stream，逐行解析 NDJSON 事件流。
 * 返回一个 async generator，使用 for-await-of 消费。
 */
export async function* chatStream(req: ChatStreamRequest): AsyncGenerator<StreamEvent> {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
    },
    body: JSON.stringify(req),
  })
  if (!res.ok || !res.body) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const data = await res.json()
      if (data?.detail) detail = String(data.detail)
    } catch {}
    throw new Error(detail)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder("utf-8")
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let nl: number
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim()
      buffer = buffer.slice(nl + 1)
      if (!line) continue
      try {
        yield JSON.parse(line) as StreamEvent
      } catch (e) {
        console.warn("[chatStream] failed to parse line:", line, e)
      }
    }
  }
  // 处理结尾残留
  const tail = buffer.trim()
  if (tail) {
    try {
      yield JSON.parse(tail) as StreamEvent
    } catch (e) {
      console.warn("[chatStream] failed to parse tail:", tail, e)
    }
  }
}
