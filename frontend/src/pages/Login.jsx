import { useState } from 'react'
import { Navigate, useLocation, useNavigate } from 'react-router-dom'

import { Alert, Spinner } from '../components/ui'
import { useAuth } from '../lib/auth'

export default function Login() {
  const { user, loading, login, config } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  if (!loading && user) {
    return <Navigate to={location.state?.from?.pathname || '/dashboard'} replace />
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      await login(email.trim().toLowerCase(), password)
      navigate(location.state?.from?.pathname || '/dashboard', { replace: true })
    } catch (err) {
      setError(err.message || 'Could not sign in')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="grid min-h-screen lg:grid-cols-2">
      <div className="flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-sm">
          <div className="mb-8">
            <div className="mb-5 grid h-11 w-11 place-items-center rounded-xl bg-ink-900 text-blue-400">
              <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v6a3 3 0 0 0 3 3Z" />
                <path d="M6 11v1a6 6 0 0 0 12 0v-1M12 18v3" strokeLinecap="round" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold tracking-tight">Sign in</h1>
            <p className="mt-1 text-sm text-ink-500">
              {config?.company_name || 'AI Interview Analyzer'} hiring workspace
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {error ? <Alert tone="error">{error}</Alert> : null}

            <div>
              <label className="label" htmlFor="email">
                Work email
              </label>
              <input
                id="email"
                type="email"
                autoComplete="username"
                required
                className="input"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                placeholder="you@company.com"
              />
            </div>

            <div>
              <label className="label" htmlFor="password">
                Password
              </label>
              <input
                id="password"
                type="password"
                autoComplete="current-password"
                required
                className="input"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="••••••••"
              />
            </div>

            <button type="submit" className="btn-dark w-full" disabled={busy}>
              {busy ? <Spinner className="h-4 w-4" /> : null}
              {busy ? 'Signing in…' : 'Sign in'}
            </button>
          </form>

          <p className="mt-6 text-xs leading-relaxed text-ink-500">
            Candidates do not sign in here — they receive a private interview link by email.
          </p>
        </div>
      </div>

      <div className="hidden bg-ink-950 lg:flex lg:items-center lg:justify-center lg:px-12">
        <div className="max-w-md text-white">
          <h2 className="text-2xl font-semibold leading-snug">
            Recorded interviews, scored consistently, reported the same day.
          </h2>
          <ul className="mt-7 space-y-4 text-sm text-ink-300">
            {[
              'Candidates answer by voice from any browser — no install, no scheduling.',
              'Every answer is transcribed and scored on five dimensions.',
              'Resume and spoken skills are compared to a relevance score.',
              'The candidate and your team both get the report by email automatically.',
            ].map((line) => (
              <li key={line} className="flex gap-3">
                <svg viewBox="0 0 24 24" className="mt-0.5 h-4 w-4 shrink-0 text-blue-400" fill="none" stroke="currentColor" strokeWidth="2.2">
                  <path d="m5 13 4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                {line}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  )
}
