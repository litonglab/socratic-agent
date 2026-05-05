import { useEffect, useState, useCallback } from "react"
import { fetchMe, getToken, setToken, type AuthUser } from "@/lib/api"

export interface AuthState {
  user: AuthUser | null
  loading: boolean
  refresh: () => Promise<void>
  logout: () => void
  setSession: (token: string, user: AuthUser) => void
}

export function useAuth(): AuthState {
  const [user, setUser] = useState<AuthUser | null>(null)
  const [loading, setLoading] = useState<boolean>(Boolean(getToken()))

  const refresh = useCallback(async () => {
    if (!getToken()) {
      setUser(null)
      setLoading(false)
      return
    }
    setLoading(true)
    try {
      const me = await fetchMe()
      setUser({ id: me.id, username: me.username, profile: me.profile })
    } catch (err) {
      console.warn("[useAuth] fetchMe failed, clearing token:", err)
      setToken(null)
      setUser(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const logout = useCallback(() => {
    setToken(null)
    setUser(null)
  }, [])

  const setSession = useCallback((token: string, u: AuthUser) => {
    setToken(token)
    setUser(u)
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  return { user, loading, refresh, logout, setSession }
}
