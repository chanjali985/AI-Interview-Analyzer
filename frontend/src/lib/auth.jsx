import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react'

import { api, auth as tokenStore } from './api'

const AuthContext = createContext(null)

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    async function bootstrap() {
      try {
        const publicConfig = await api.config()
        if (!cancelled) setConfig(publicConfig)
      } catch {
        /* the API may still be starting; the app degrades to defaults */
      }
      if (tokenStore.token) {
        try {
          const profile = await api.me()
          if (!cancelled) setUser(profile)
        } catch {
          tokenStore.clear()
        }
      }
      if (!cancelled) setLoading(false)
    }
    bootstrap()
    return () => {
      cancelled = true
    }
  }, [])

  const login = useCallback(async (email, password) => {
    const response = await api.login(email, password)
    tokenStore.set(response.access_token)
    const profile = await api.me()
    setUser(profile)
    return profile
  }, [])

  const logout = useCallback(() => {
    tokenStore.clear()
    setUser(null)
  }, [])

  const value = useMemo(() => ({ user, config, loading, login, logout }), [user, config, loading, login, logout])
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useAuth must be used inside AuthProvider')
  return context
}
