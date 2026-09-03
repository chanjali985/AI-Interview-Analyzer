import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import { Alert, Loading, Spinner, useToast } from '../components/ui'
import { api } from '../lib/api'

const BLANK_QUESTION = { text: '', category: 'general', time_limit_seconds: 180 }

const SUGGESTIONS = [
  'Tell us about your background and the work you are most proud of.',
  'Walk us through a technical problem you solved recently. What made it hard?',
  'Describe a time you disagreed with a teammate. How did you handle it?',
  'Which tools and technologies do you reach for most, and why?',
  'How do you decide what to work on when everything feels urgent?',
]

export default function RoleEditor() {
  const { id } = useParams()
  const isNew = !id
  const navigate = useNavigate()
  const toast = useToast()

  const [form, setForm] = useState({ title: '', department: '', description: '' })
  const [questions, setQuestions] = useState([{ ...BLANK_QUESTION }])
  const [locked, setLocked] = useState(false)
  const [loading, setLoading] = useState(!isNew)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (isNew) return
    api
      .getRole(id)
      .then((role) => {
        setForm({ title: role.title, department: role.department, description: role.description })
        setQuestions(
          role.questions.map((question) => ({
            text: question.text,
            category: question.category,
            time_limit_seconds: question.time_limit_seconds,
          })),
        )
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [id, isNew])

  useEffect(() => {
    if (isNew) return
    api
      .listRoles()
      .then((roles) => {
        const match = roles.find((role) => role.id === Number(id))
        setLocked(Boolean(match && match.interview_count > 0))
      })
      .catch(() => setLocked(false))
  }, [id, isNew])

  function updateQuestion(index, patch) {
    setQuestions((current) => current.map((question, i) => (i === index ? { ...question, ...patch } : question)))
  }

  function removeQuestion(index) {
    setQuestions((current) => current.filter((_, i) => i !== index))
  }

  function moveQuestion(index, direction) {
    setQuestions((current) => {
      const next = [...current]
      const target = index + direction
      if (target < 0 || target >= next.length) return current
      ;[next[index], next[target]] = [next[target], next[index]]
      return next
    })
  }

  async function save(event) {
    event.preventDefault()
    setError('')

    const cleaned = questions
      .map((question) => ({ ...question, text: question.text.trim() }))
      .filter((question) => question.text.length >= 5)

    if (!form.title.trim()) return setError('Give the role a title')
    if (cleaned.length === 0) return setError('Add at least one question of five characters or more')

    setSaving(true)
    try {
      if (isNew) {
        const role = await api.createRole({ ...form, questions: cleaned })
        toast.success('Role created')
        navigate(`/roles/${role.id}`, { replace: true })
      } else {
        await api.updateRole(id, locked ? form : { ...form, questions: cleaned })
        toast.success(locked ? 'Details saved (questions are locked)' : 'Role saved')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setSaving(false)
    }
  }

  if (loading) return <Loading />

  return (
    <form onSubmit={save} className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <Link to="/roles" className="text-sm text-ink-500 hover:text-ink-900">
            ← Roles
          </Link>
          <h1 className="mt-1 text-2xl font-bold tracking-tight">{isNew ? 'New role' : form.title || 'Edit role'}</h1>
        </div>
        <button type="submit" className="btn-dark" disabled={saving}>
          {saving ? <Spinner className="h-4 w-4" /> : null}
          {saving ? 'Saving…' : isNew ? 'Create role' : 'Save changes'}
        </button>
      </div>

      {error ? <Alert tone="error">{error}</Alert> : null}
      {locked ? (
        <Alert tone="info" title="Questions are locked">
          Candidates have already been invited to this role, so its questions cannot change — that would make past and
          future scores incomparable. Create a new role for a different question set.
        </Alert>
      ) : null}

      <section className="card space-y-4 p-5">
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label className="label" htmlFor="title">
              Role title
            </label>
            <input
              id="title"
              className="input"
              value={form.title}
              onChange={(event) => setForm({ ...form, title: event.target.value })}
              placeholder="Backend Engineer"
              required
            />
          </div>
          <div>
            <label className="label" htmlFor="department">
              Department
            </label>
            <input
              id="department"
              className="input"
              value={form.department}
              onChange={(event) => setForm({ ...form, department: event.target.value })}
              placeholder="Engineering"
            />
          </div>
        </div>
        <div>
          <label className="label" htmlFor="description">
            What candidates should know
          </label>
          <textarea
            id="description"
            rows={3}
            className="input"
            value={form.description}
            onChange={(event) => setForm({ ...form, description: event.target.value })}
            placeholder="Shown on the candidate's interview page before they start."
          />
        </div>
      </section>

      <section className="card p-5">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="font-semibold">Questions</h2>
            <p className="text-xs text-ink-500">Asked in this order. Candidates record one answer per question.</p>
          </div>
          {!locked ? (
            <button
              type="button"
              className="btn-ghost py-2 text-xs"
              onClick={() => setQuestions((current) => [...current, { ...BLANK_QUESTION }])}
            >
              Add question
            </button>
          ) : null}
        </div>

        <div className="space-y-3">
          {questions.map((question, index) => (
            <div key={index} className="rounded-lg border border-ink-200 p-3.5">
              <div className="flex items-start gap-3">
                <span className="mt-2 grid h-6 w-6 shrink-0 place-items-center rounded-full bg-ink-100 text-xs font-semibold text-ink-600">
                  {index + 1}
                </span>
                <div className="min-w-0 flex-1 space-y-3">
                  <textarea
                    rows={2}
                    className="input"
                    disabled={locked}
                    value={question.text}
                    onChange={(event) => updateQuestion(index, { text: event.target.value })}
                    placeholder="Ask something a candidate can answer out loud in two minutes."
                  />
                  <div className="flex flex-wrap items-center gap-3">
                    <select
                      className="input w-auto py-1.5 text-xs"
                      disabled={locked}
                      value={question.category}
                      onChange={(event) => updateQuestion(index, { category: event.target.value })}
                    >
                      <option value="general">General</option>
                      <option value="technical">Technical</option>
                      <option value="behavioural">Behavioural</option>
                      <option value="background">Background</option>
                    </select>
                    <label className="flex items-center gap-2 text-xs text-ink-500">
                      Time limit
                      <select
                        className="input w-auto py-1.5 text-xs"
                        disabled={locked}
                        value={question.time_limit_seconds}
                        onChange={(event) => updateQuestion(index, { time_limit_seconds: Number(event.target.value) })}
                      >
                        {[60, 90, 120, 180, 240, 300].map((seconds) => (
                          <option key={seconds} value={seconds}>
                            {seconds / 60} min
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                </div>
                {!locked ? (
                  <div className="flex shrink-0 flex-col gap-1">
                    <button type="button" onClick={() => moveQuestion(index, -1)} className="rounded px-2 py-1 text-ink-400 hover:bg-ink-50" aria-label="Move up">
                      ↑
                    </button>
                    <button type="button" onClick={() => moveQuestion(index, 1)} className="rounded px-2 py-1 text-ink-400 hover:bg-ink-50" aria-label="Move down">
                      ↓
                    </button>
                    <button
                      type="button"
                      onClick={() => removeQuestion(index)}
                      disabled={questions.length === 1}
                      className="rounded px-2 py-1 text-red-400 hover:bg-red-50 disabled:opacity-30"
                      aria-label="Remove question"
                    >
                      ×
                    </button>
                  </div>
                ) : null}
              </div>
            </div>
          ))}
        </div>

        {!locked ? (
          <div className="mt-5 border-t border-ink-100 pt-4">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-500">Need ideas?</p>
            <div className="flex flex-wrap gap-2">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => setQuestions((current) => [...current, { ...BLANK_QUESTION, text: suggestion }])}
                  className="rounded-full border border-ink-200 px-3 py-1.5 text-xs text-ink-600 hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700"
                >
                  + {suggestion.slice(0, 42)}…
                </button>
              ))}
            </div>
          </div>
        ) : null}
      </section>
    </form>
  )
}
