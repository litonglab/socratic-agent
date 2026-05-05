import { useEffect, useState, useCallback } from "react"
import { fetchMe, getToken, setToken, type AuthUser } from "@/lib/api"

export interface AuthState {
  user: AuthUser | null
  loading: boolean
  refresh: () => Promise<void>
  logout: () => void
  setSession: (token: string, user: AuthUser) => void
}

/**
 * 内部纯异步函数：根据 token 拉取当前用户。
 * 失败时只清 localStorage 里的 token（副作用），不触碰 React state，
 * 让上层调用者统一决定 setState 时机，便于把 setState 放到 await 之后。
 */
async function fetchAuthUser(): Promise<AuthUser | null> {
  if (!getToken()) return null
  try {
    const me = await fetchMe()
    return { id: me.id, username: me.username, profile: me.profile }
  } catch (err) {
    console.warn("[useAuth] fetchMe failed, clearing token:", err)
    setToken(null)
    return null
  }
}

export function useAuth(): AuthState {
  const [user, setUser] = useState<AuthUser | null>(null)
  // 有 token 时初始即 loading=true，避免在 effect 同步路径上调用 setLoading(true)
  // —— 这正是 react-hooks/set-state-in-effect 规则希望我们做的：用 lazy 初值替代
  const [loading, setLoading] = useState<boolean>(() => Boolean(getToken()))

  const refresh = useCallback(async () => {
    setLoading(true)
    const u = await fetchAuthUser()
    setUser(u)
    setLoading(false)
  }, [])

  const logout = useCallback(() => {
    setToken(null)
    setUser(null)
  }, [])

  const setSession = useCallback((token: string, u: AuthUser) => {
    setToken(token)
    setUser(u)
  }, [])

  // mount 时初始化：所有 setState 都发生在 .then() 异步回调里，
  // effect 同步路径只做 cancelled flag 管理，不触发 setState
  useEffect(() => {
    let cancelled = false
    fetchAuthUser().then((u) => {
      if (cancelled) return
      setUser(u)
      setLoading(false)
    })
    return () => {
      cancelled = true
    }
  }, [])

  return { user, loading, refresh, logout, setSession }
}
