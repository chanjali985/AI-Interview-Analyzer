import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { Alert, EmptyState, Loading, StatCard, StatusBadge } from '../components/ui'
import { api } from '../lib/api'
import { formatDateTime, scoreColor } from '../lib/format'

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [health, setHealth] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    Promise.all([api.dashboard(), api.health().catch(() => null)])
      .then(([dashboard, healthResult]) => {
        if (cancelled) return
        setStats(dashboard)
        setHealth(healthResult)
      })
      .catch((err) => !cancelled && setError(err.message))
    return () => {
      cancelled = true
    }
  }, [])

  if (error) return <Alert tone="error" title="Could not load the dashboard">{error}</Alert>
  if (!stats) return <Loading />

  const warnings = []
  if (health && !health.llm_ready) {
    warnings.push(`The language model (${health.llm_provider}) is not reachable — analysis will fail until it is running.`)
  }
  if (health && !health.transcription_ready) {
    warnings.push(`Transcription (${health.transcription_provider}) is not ready — recordings cannot be transcribed.`)
  }
  if (health && !health.mail_configured) {
    warnings.push('Email is not configured, so reports will not be delivered. Set SMTP_PASSWORD to a Gmail App Password.')
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <p className="mt-1 text-sm text-ink-500">Hiring pipeline at a glance.</p>
        </div>
        <div className="flex gap-2">
          <Link to="/roles/new" className="btn-ghost">
            New role
          </Link>
          <Link to="/interviews" className="btn-dark">
            Invite a candidate
          </Link>
        </div>
      </div>

      {warnings.length ? (
        <Alert tone="warning" title="Service checks">
          <ul className="ml-4 list-disc space-y-1">
            {warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </Alert>
      ) : null}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Interviews" value={stats.total_interviews} hint={`${stats.total_candidates} candidates`} />
        <StatCard label="Completed" value={stats.completed_interviews} hint={`${stats.pending_interviews} in flight`} tone="good" />
        <StatCard
          label="Average score"
          value={stats.average_score == null ? '—' : stats.average_score.toFixed(1)}
          hint="across analysed interviews"
        />
        <StatCard
          label="Above threshold"
          value={stats.shortlisted}
          hint="ready to shortlist"
          tone={stats.shortlisted ? 'good' : 'default'}
        />
      </div>

      {stats.failed_interviews ? (
        <Alert tone="error" title={`${stats.failed_interviews} interview(s) failed to analyse`}>
          Open the candidate, check the error, then use “Re-run analysis”. The recordings are kept.
        </Alert>
      ) : null}

      <section className="card overflow-hidden">
        <div className="flex items-center justify-between border-b border-ink-100 px-5 py-4">
          <h2 className="text-sm font-semibold">Recent activity</h2>
          <Link to="/interviews" className="text-sm font-medium text-blue-600 hover:text-blue-700">
            View all
          </Link>
        </div>

        {stats.recent.length === 0 ? (
          <EmptyState
            title="No interviews yet"
            description="Create a role with your question set, then invite your first candidate by email."
            action={
              <Link to="/roles/new" className="btn-primary mt-2">
                Create a role
              </Link>
            }
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-ink-100 bg-ink-50/60">
                  <th className="table-head px-5 py-2.5">Candidate</th>
                  <th className="table-head px-5 py-2.5">Role</th>
                  <th className="table-head px-5 py-2.5">Status</th>
                  <th className="table-head px-5 py-2.5 text-right">Score</th>
                  <th className="table-head px-5 py-2.5">Invited</th>
                </tr>
              </thead>
              <tbody>
                {stats.recent.map((interview) => (
                  <tr key={interview.id} className="border-b border-ink-50 last:border-0 hover:bg-ink-50/50">
                    <td className="px-5 py-3">
                      <Link to={`/reports/${interview.id}`} className="font-medium text-ink-900 hover:text-blue-600">
                        {interview.candidate.full_name}
                      </Link>
                      <div className="text-xs text-ink-500">{interview.candidate.email}</div>
                    </td>
                    <td className="px-5 py-3 text-ink-600">{interview.role.title}</td>
                    <td className="px-5 py-3">
                      <StatusBadge status={interview.status} />
                    </td>
                    <td className={`px-5 py-3 text-right font-semibold tabular-nums ${scoreColor(interview.overall_score)}`}>
                      {interview.overall_score == null ? '—' : interview.overall_score.toFixed(1)}
                    </td>
                    <td className="px-5 py-3 text-xs text-ink-500">{formatDateTime(interview.invited_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
