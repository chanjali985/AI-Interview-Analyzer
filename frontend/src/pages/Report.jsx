import { useCallback, useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'

import { Alert, Loading, ScoreBar, ScoreRing, Spinner, StatusBadge, useToast } from '../components/ui'
import { api, authedBlobUrl, downloadWithAuth } from '../lib/api'
import { DIMENSION_LABELS, VERDICT_LABELS, formatDateTime, formatDuration } from '../lib/format'

function AudioPlayer({ interviewId, answerId }) {
  const [url, setUrl] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => () => url && URL.revokeObjectURL(url), [url])

  if (url) return <audio controls src={url} className="mt-3 h-9 w-full max-w-md" />

  return (
    <div className="mt-3">
      <button
        type="button"
        className="btn-ghost py-1.5 text-xs"
        disabled={loading}
        onClick={async () => {
          setLoading(true)
          try {
            setUrl(await authedBlobUrl(api.audioUrl(interviewId, answerId)))
          } catch (err) {
            setError(err.message)
          } finally {
            setLoading(false)
          }
        }}
      >
        {loading ? <Spinner className="h-3.5 w-3.5" /> : '▶'} Play recording
      </button>
      {error ? <p className="mt-1 text-xs text-red-600">{error}</p> : null}
    </div>
  )
}

export default function Report() {
  const { id } = useParams()
  const navigate = useNavigate()
  const toast = useToast()
  const [interview, setInterview] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState('')

  const load = useCallback(async () => {
    try {
      setInterview(await api.getInterview(id))
    } catch (err) {
      setError(err.message)
    }
  }, [id])

  useEffect(() => {
    load()
  }, [load])

  useEffect(() => {
    if (!interview || !['submitted', 'processing'].includes(interview.status)) return undefined
    const timer = setInterval(load, 5000)
    return () => clearInterval(timer)
  }, [interview, load])

  async function run(action, label, fn) {
    setBusy(action)
    try {
      await fn()
      await load()
      toast.success(label)
    } catch (err) {
      toast.error(err.message)
    } finally {
      setBusy('')
    }
  }

  if (error) return <Alert tone="error">{error}</Alert>
  if (!interview) return <Loading />

  const done = interview.status === 'completed'
  const questionById = Object.fromEntries(interview.role.questions.map((question) => [question.id, question]))

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link to="/interviews" className="text-sm text-ink-500 hover:text-ink-900">
            ← Candidates
          </Link>
          <h1 className="mt-1 text-2xl font-bold tracking-tight">{interview.candidate.full_name}</h1>
          <p className="mt-1 text-sm text-ink-500">
            {interview.candidate.email} · {interview.role.title}
            {interview.role.department ? ` · ${interview.role.department}` : ''}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <StatusBadge status={interview.status} />
          {done ? (
            <>
              <button
                type="button"
                className="btn-ghost py-2 text-xs"
                onClick={() => downloadWithAuth(api.reportPdfUrl(interview.id), `report-${interview.candidate.full_name}.pdf`)}
              >
                Download PDF
              </button>
              <button
                type="button"
                className="btn-ghost py-2 text-xs"
                disabled={busy === 'resend'}
                onClick={() => run('resend', 'Report emailed to the candidate', () => api.resendReport(interview.id))}
              >
                {busy === 'resend' ? <Spinner className="h-3.5 w-3.5" /> : null} Email candidate
              </button>
            </>
          ) : null}
          {['completed', 'failed'].includes(interview.status) ? (
            <button
              type="button"
              className="btn-ghost py-2 text-xs"
              disabled={busy === 'rerun'}
              onClick={() => run('rerun', 'Analysis queued', () => api.reanalyze(interview.id))}
            >
              {busy === 'rerun' ? <Spinner className="h-3.5 w-3.5" /> : null} Re-run analysis
            </button>
          ) : null}
          <button
            type="button"
            className="btn-ghost py-2 text-xs text-red-600"
            onClick={async () => {
              if (!window.confirm('Delete this interview and its recordings? This cannot be undone.')) return
              await api.deleteInterview(interview.id)
              toast.success('Interview deleted')
              navigate('/interviews')
            }}
          >
            Delete
          </button>
        </div>
      </div>

      {interview.status === 'failed' ? (
        <Alert tone="error" title="Analysis failed">
          {interview.error_message || 'Unknown error.'} The recordings are kept — fix the cause and re-run.
        </Alert>
      ) : null}

      {['submitted', 'processing'].includes(interview.status) ? (
        <Alert tone="info" title="Analysis in progress">
          Transcribing and scoring the recordings. This page refreshes itself.
        </Alert>
      ) : null}

      {['invited', 'in_progress', 'expired'].includes(interview.status) ? (
        <Alert tone="warning" title="Not submitted yet">
          Invited {formatDateTime(interview.invited_at)}. The report appears once the candidate submits.
        </Alert>
      ) : null}

      {done ? (
        <>
          <section className="card p-6">
            <div className="flex flex-col gap-8 lg:flex-row lg:items-center">
              <div className="flex items-center gap-6">
                <ScoreRing score={interview.overall_score} />
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wide text-ink-500">Recommendation</p>
                  <p className="mt-1 text-lg font-semibold">{VERDICT_LABELS[interview.verdict] || '—'}</p>
                  <p className="mt-3 text-xs font-semibold uppercase tracking-wide text-ink-500">Skill match</p>
                  <p className="mt-1 text-lg font-semibold tabular-nums text-teal-600">
                    {Math.round((interview.relevance_score ?? 0) * 100)}%
                  </p>
                  <p className="mt-3 text-xs text-ink-500">Submitted {formatDateTime(interview.submitted_at)}</p>
                </div>
              </div>

              <div className="min-w-0 flex-1 space-y-2.5 lg:border-l lg:border-ink-100 lg:pl-8">
                {Object.entries(DIMENSION_LABELS).map(([key, label]) => (
                  <ScoreBar key={key} label={label} value={interview.dimension_scores?.[key]} />
                ))}
                <ScoreBar
                  label="Resume ↔ answers"
                  value={(interview.relevance_score ?? 0) * 10}
                  suffix={`${Math.round((interview.relevance_score ?? 0) * 100)}%`}
                />
              </div>
            </div>
          </section>

          <div className="grid gap-5 lg:grid-cols-3">
            <section className="card p-5 lg:col-span-2">
              <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-ink-500">Assessment</h2>
              <ul className="space-y-2 text-sm text-ink-700">
                {(interview.summary || []).map((point, index) => (
                  <li key={index} className="flex gap-2.5">
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500" />
                    {point}
                  </li>
                ))}
              </ul>

              {interview.strengths?.length ? (
                <>
                  <h3 className="mb-2 mt-6 text-sm font-semibold uppercase tracking-wide text-emerald-700">Strengths</h3>
                  <ul className="space-y-2 text-sm text-ink-700">
                    {interview.strengths.map((point, index) => (
                      <li key={index} className="flex gap-2.5">
                        <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-500" />
                        {point}
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}

              {interview.improvements?.length ? (
                <>
                  <h3 className="mb-2 mt-6 text-sm font-semibold uppercase tracking-wide text-amber-700">Areas to improve</h3>
                  <ul className="space-y-2 text-sm text-ink-700">
                    {interview.improvements.map((point, index) => (
                      <li key={index} className="flex gap-2.5">
                        <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-500" />
                        {point}
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}
            </section>

            <section className="card space-y-5 p-5">
              <div>
                <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-ink-500">Skills in answers</h2>
                <div className="flex flex-wrap gap-1.5">
                  {(interview.skills_answer || []).map((skill) => (
                    <span key={skill} className="chip">
                      {skill}
                    </span>
                  ))}
                  {!interview.skills_answer?.length ? <p className="text-sm text-ink-400">None detected</p> : null}
                </div>
              </div>
              <div>
                <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-ink-500">Skills on resume</h2>
                <div className="flex flex-wrap gap-1.5">
                  {(interview.skills_resume || []).map((skill) => (
                    <span key={skill} className="chip bg-ink-100 text-ink-700">
                      {skill}
                    </span>
                  ))}
                  {!interview.skills_resume?.length ? <p className="text-sm text-ink-400">None detected</p> : null}
                </div>
              </div>
              <div className="border-t border-ink-100 pt-4 text-xs text-ink-500">
                Report emailed to candidate:{' '}
                <span className="font-medium text-ink-700">
                  {interview.candidate_email_sent_at ? formatDateTime(interview.candidate_email_sent_at) : 'not sent'}
                </span>
              </div>
            </section>
          </div>
        </>
      ) : null}

      {interview.answers?.length ? (
        <section className="card p-5">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-ink-500">Answers</h2>
          <div className="space-y-5">
            {interview.answers.map((answer, index) => (
              <div key={answer.id} className="rounded-lg border border-ink-200 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <p className="font-medium text-ink-900">
                    Q{index + 1}. {questionById[answer.question_id]?.text || 'Question removed'}
                  </p>
                  <span className="text-xs text-ink-500">{formatDuration(answer.duration_seconds)}</span>
                </div>

                {answer.scores ? (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {Object.entries(answer.scores).map(([key, value]) => (
                      <span key={key} className="rounded-md bg-ink-50 px-2 py-1 text-xs text-ink-600">
                        {DIMENSION_LABELS[key] || key}: <strong className="tabular-nums">{Number(value).toFixed(1)}</strong>
                      </span>
                    ))}
                  </div>
                ) : null}

                <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-ink-700">
                  {answer.transcript || <span className="italic text-ink-400">No transcript yet.</span>}
                </p>

                {answer.feedback ? (
                  <p className="mt-3 rounded-md bg-blue-50 px-3 py-2 text-sm text-blue-900">{answer.feedback}</p>
                ) : null}

                <AudioPlayer interviewId={interview.id} answerId={answer.id} />
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  )
}
