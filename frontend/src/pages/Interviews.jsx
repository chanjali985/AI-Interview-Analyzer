import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { Alert, EmptyState, Loading, Modal, Spinner, StatusBadge, useToast } from '../components/ui'
import { api } from '../lib/api'
import { formatDateTime, scoreColor } from '../lib/format'

const STATUS_OPTIONS = [
  ['', 'All statuses'],
  ['invited', 'Invited'],
  ['in_progress', 'In progress'],
  ['submitted', 'Submitted'],
  ['processing', 'Analysing'],
  ['completed', 'Completed'],
  ['failed', 'Failed'],
  ['expired', 'Expired'],
]

function InviteModal({ open, onClose, roles, onInvited }) {
  const toast = useToast()
  const [form, setForm] = useState({ full_name: '', email: '', phone: '', role_id: '', resume_text: '', send_email: true })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  const activeRoles = roles.filter((role) => role.is_active && role.question_count > 0)

  useEffect(() => {
    if (open) {
      setForm((current) => ({ ...current, role_id: current.role_id || activeRoles[0]?.id || '' }))
      setResult(null)
      setError('')
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, roles])

  async function submit(event) {
    event.preventDefault()
    setError('')
    setBusy(true)
    try {
      const response = await api.invite({ ...form, role_id: Number(form.role_id) })
      setResult(response)
      if (response.email_sent) toast.success(`Invite emailed to ${form.email}`)
      else toast.info('Invite created — copy the link below')
      onInvited()
      setForm({ full_name: '', email: '', phone: '', role_id: form.role_id, resume_text: '', send_email: true })
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Invite a candidate"
      width="max-w-xl"
      footer={
        result ? (
          <button type="button" className="btn-dark" onClick={onClose}>
            Done
          </button>
        ) : (
          <>
            <button type="button" className="btn-ghost" onClick={onClose}>
              Cancel
            </button>
            <button type="submit" form="invite-form" className="btn-dark" disabled={busy || !activeRoles.length}>
              {busy ? <Spinner className="h-4 w-4" /> : null}
              {busy ? 'Sending…' : 'Send invite'}
            </button>
          </>
        )
      }
    >
      {result ? (
        <div className="space-y-4">
          <Alert tone={result.email_sent ? 'success' : 'warning'}>
            {result.email_sent
              ? 'The invitation email has been sent.'
              : `The invite was created but not emailed${result.email_error ? `: ${result.email_error}` : ''}. Share the link below instead.`}
          </Alert>
          <div>
            <p className="label">Interview link</p>
            <div className="flex gap-2">
              <input readOnly className="input font-mono text-xs" value={result.invite_url} onFocus={(e) => e.target.select()} />
              <button
                type="button"
                className="btn-ghost shrink-0"
                onClick={() => {
                  navigator.clipboard?.writeText(result.invite_url)
                  toast.success('Link copied')
                }}
              >
                Copy
              </button>
            </div>
            <p className="mt-2 text-xs text-ink-500">
              Anyone with this link can take the interview, so share it only with the candidate.
            </p>
          </div>
        </div>
      ) : (
        <form id="invite-form" onSubmit={submit} className="space-y-4">
          {error ? <Alert tone="error">{error}</Alert> : null}
          {!activeRoles.length ? (
            <Alert tone="warning">
              You need an active role with at least one question before you can invite anyone.{' '}
              <Link to="/roles/new" className="font-semibold underline">
                Create one
              </Link>
              .
            </Alert>
          ) : null}

          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="label" htmlFor="full_name">
                Full name
              </label>
              <input
                id="full_name"
                className="input"
                required
                value={form.full_name}
                onChange={(event) => setForm({ ...form, full_name: event.target.value })}
              />
            </div>
            <div>
              <label className="label" htmlFor="candidate_email">
                Email
              </label>
              <input
                id="candidate_email"
                type="email"
                className="input"
                required
                value={form.email}
                onChange={(event) => setForm({ ...form, email: event.target.value })}
              />
            </div>
          </div>

          <div>
            <label className="label" htmlFor="role">
              Role
            </label>
            <select
              id="role"
              className="input"
              required
              value={form.role_id}
              onChange={(event) => setForm({ ...form, role_id: event.target.value })}
            >
              <option value="">Select a role…</option>
              {activeRoles.map((role) => (
                <option key={role.id} value={role.id}>
                  {role.title} ({role.question_count} questions)
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="label" htmlFor="resume">
              Resume text <span className="font-normal text-ink-400">(optional — the candidate can upload theirs)</span>
            </label>
            <textarea
              id="resume"
              rows={3}
              className="input"
              value={form.resume_text}
              onChange={(event) => setForm({ ...form, resume_text: event.target.value })}
              placeholder="Paste the resume you already have on file."
            />
          </div>

          <label className="flex items-center gap-2.5 text-sm text-ink-700">
            <input
              type="checkbox"
              className="h-4 w-4 rounded border-ink-300"
              checked={form.send_email}
              onChange={(event) => setForm({ ...form, send_email: event.target.checked })}
            />
            Email the invitation now
          </label>
        </form>
      )}
    </Modal>
  )
}

export default function Interviews() {
  const [interviews, setInterviews] = useState(null)
  const [roles, setRoles] = useState([])
  const [filters, setFilters] = useState({ status: '', role_id: '', search: '' })
  const [error, setError] = useState('')
  const [inviteOpen, setInviteOpen] = useState(false)

  const load = useCallback(async () => {
    try {
      const data = await api.listInterviews(filters)
      setInterviews(data)
    } catch (err) {
      setError(err.message)
    }
  }, [filters])

  useEffect(() => {
    api.listRoles().then(setRoles).catch(() => setRoles([]))
  }, [])

  useEffect(() => {
    const timer = setTimeout(load, 200)
    return () => clearTimeout(timer)
  }, [load])

  // Keep the list fresh while analyses are running.
  useEffect(() => {
    const pending = interviews?.some((item) => ['submitted', 'processing'].includes(item.status))
    if (!pending) return undefined
    const timer = setInterval(load, 8000)
    return () => clearInterval(timer)
  }, [interviews, load])

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Candidates</h1>
          <p className="mt-1 text-sm text-ink-500">Every invitation, recording and report.</p>
        </div>
        <button type="button" className="btn-dark" onClick={() => setInviteOpen(true)}>
          Invite a candidate
        </button>
      </div>

      <div className="card flex flex-wrap gap-3 p-4">
        <input
          className="input sm:max-w-xs"
          placeholder="Search name or email…"
          value={filters.search}
          onChange={(event) => setFilters({ ...filters, search: event.target.value })}
        />
        <select
          className="input sm:w-44"
          value={filters.status}
          onChange={(event) => setFilters({ ...filters, status: event.target.value })}
        >
          {STATUS_OPTIONS.map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
        <select
          className="input sm:w-56"
          value={filters.role_id}
          onChange={(event) => setFilters({ ...filters, role_id: event.target.value })}
        >
          <option value="">All roles</option>
          {roles.map((role) => (
            <option key={role.id} value={role.id}>
              {role.title}
            </option>
          ))}
        </select>
      </div>

      {error ? <Alert tone="error">{error}</Alert> : null}

      {!interviews ? (
        <Loading />
      ) : interviews.length === 0 ? (
        <EmptyState
          title="No candidates match"
          description="Invite someone, or clear the filters above."
          action={
            <button type="button" className="btn-primary mt-2" onClick={() => setInviteOpen(true)}>
              Invite a candidate
            </button>
          }
        />
      ) : (
        <div className="card overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-ink-100 bg-ink-50/60">
                <th className="table-head px-5 py-2.5">Candidate</th>
                <th className="table-head px-5 py-2.5">Role</th>
                <th className="table-head px-5 py-2.5">Status</th>
                <th className="table-head px-5 py-2.5 text-right">Score</th>
                <th className="table-head px-5 py-2.5 text-right">Match</th>
                <th className="table-head px-5 py-2.5">Submitted</th>
                <th className="px-5 py-2.5" />
              </tr>
            </thead>
            <tbody>
              {interviews.map((interview) => (
                <tr key={interview.id} className="border-b border-ink-50 last:border-0 hover:bg-ink-50/50">
                  <td className="px-5 py-3">
                    <div className="font-medium text-ink-900">{interview.candidate.full_name}</div>
                    <div className="text-xs text-ink-500">{interview.candidate.email}</div>
                  </td>
                  <td className="px-5 py-3 text-ink-600">{interview.role.title}</td>
                  <td className="px-5 py-3">
                    <StatusBadge status={interview.status} />
                  </td>
                  <td className={`px-5 py-3 text-right font-semibold tabular-nums ${scoreColor(interview.overall_score)}`}>
                    {interview.overall_score == null ? '—' : interview.overall_score.toFixed(1)}
                  </td>
                  <td className="px-5 py-3 text-right tabular-nums text-ink-600">
                    {interview.relevance_score == null ? '—' : `${Math.round(interview.relevance_score * 100)}%`}
                  </td>
                  <td className="px-5 py-3 text-xs text-ink-500">{formatDateTime(interview.submitted_at)}</td>
                  <td className="px-5 py-3 text-right">
                    <Link to={`/reports/${interview.id}`} className="text-sm font-medium text-blue-600 hover:text-blue-700">
                      Open
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <InviteModal open={inviteOpen} onClose={() => setInviteOpen(false)} roles={roles} onInvited={load} />
    </div>
  )
}
