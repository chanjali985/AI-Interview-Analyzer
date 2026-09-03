import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

import { Alert, EmptyState, Loading, useToast } from '../components/ui'
import { api } from '../lib/api'

export default function Roles() {
  const [roles, setRoles] = useState(null)
  const [error, setError] = useState('')
  const toast = useToast()

  useEffect(() => {
    api.listRoles().then(setRoles).catch((err) => setError(err.message))
  }, [])

  async function archive(role) {
    if (!window.confirm(`Archive “${role.title}”? Existing interviews are kept.`)) return
    try {
      await api.deleteRole(role.id)
      setRoles(await api.listRoles())
      toast.success('Role archived')
    } catch (err) {
      toast.error(err.message)
    }
  }

  if (error) return <Alert tone="error">{error}</Alert>
  if (!roles) return <Loading />

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Roles & questions</h1>
          <p className="mt-1 text-sm text-ink-500">
            Each role carries its own question set. Candidates answer those questions by voice.
          </p>
        </div>
        <Link to="/roles/new" className="btn-dark">
          New role
        </Link>
      </div>

      {roles.length === 0 ? (
        <EmptyState
          title="No roles yet"
          description="A role is a job opening plus the questions every candidate for it will answer."
          action={
            <Link to="/roles/new" className="btn-primary mt-2">
              Create your first role
            </Link>
          }
        />
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {roles.map((role) => (
            <div key={role.id} className="card flex flex-col p-5">
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <h2 className="truncate font-semibold text-ink-900">{role.title}</h2>
                  <p className="text-xs text-ink-500">{role.department || 'No department'}</p>
                </div>
                {role.is_active ? (
                  <span className="chip bg-emerald-50 text-emerald-700">Active</span>
                ) : (
                  <span className="chip bg-ink-100 text-ink-600">Archived</span>
                )}
              </div>

              <dl className="mt-4 flex gap-6 text-sm">
                <div>
                  <dt className="text-xs text-ink-500">Questions</dt>
                  <dd className="font-semibold tabular-nums">{role.question_count}</dd>
                </div>
                <div>
                  <dt className="text-xs text-ink-500">Interviews</dt>
                  <dd className="font-semibold tabular-nums">{role.interview_count}</dd>
                </div>
              </dl>

              <div className="mt-5 flex gap-2 border-t border-ink-100 pt-4">
                <Link to={`/roles/${role.id}`} className="btn-ghost flex-1 py-2 text-xs">
                  Open
                </Link>
                {role.is_active ? (
                  <button type="button" onClick={() => archive(role)} className="btn-ghost py-2 text-xs text-red-600">
                    Archive
                  </button>
                ) : null}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
