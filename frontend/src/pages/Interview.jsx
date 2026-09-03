import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'

import { Alert, Loading, Spinner } from '../components/ui'
import { api } from '../lib/api'
import { formatDuration } from '../lib/format'
import { extensionFor, recordingSupported, useRecorder } from '../lib/useRecorder'

/* -------------------------------------------------------------------- shell */
function Shell({ company, children, step, totalSteps }) {
  return (
    <div className="min-h-screen bg-ink-50">
      <header className="border-b border-ink-200 bg-white">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-5 py-4">
          <div className="flex items-center gap-2.5">
            <div className="grid h-8 w-8 place-items-center rounded-lg bg-ink-900 text-blue-400">
              <svg viewBox="0 0 24 24" className="h-4.5 w-4.5" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v6a3 3 0 0 0 3 3Z" />
                <path d="M6 11v1a6 6 0 0 0 12 0v-1M12 18v3" strokeLinecap="round" />
              </svg>
            </div>
            <span className="text-sm font-semibold">{company}</span>
          </div>
          {step != null ? (
            <span className="text-xs font-medium text-ink-500">
              Step {step} of {totalSteps}
            </span>
          ) : null}
        </div>
      </header>
      <main className="mx-auto max-w-3xl px-5 py-8 sm:py-12">{children}</main>
      <footer className="mx-auto max-w-3xl px-5 pb-10 text-center text-xs text-ink-400">
        Your answers are analysed automatically and reviewed by a person before any decision is made.
      </footer>
    </div>
  )
}

/* ----------------------------------------------------------------- recorder */
function QuestionRecorder({ question, index, total, existing, onSaved, token }) {
  const [blob, setBlob] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [savedSeconds, setSavedSeconds] = useState(existing?.seconds ?? null)
  const lastSecondsRef = useRef(0)

  const handleStop = useCallback((recorded, meta) => {
    lastSecondsRef.current = meta.seconds
    setBlob({ data: recorded, mimeType: meta.mimeType, seconds: meta.seconds })
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current)
      return URL.createObjectURL(recorded)
    })
  }, [])

  const recorder = useRecorder({ maxSeconds: question.time_limit_seconds, onStop: handleStop })

  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl])

  async function upload() {
    if (!blob) return
    setError('')
    setUploading(true)
    try {
      const form = new FormData()
      form.append('audio', blob.data, `answer-q${question.id}.${extensionFor(blob.mimeType)}`)
      form.append('duration_seconds', String(blob.seconds || 0))
      await api.uploadAnswer(token, question.id, form)
      setSavedSeconds(blob.seconds)
      onSaved(question.id, blob.seconds)
      setBlob(null)
      recorder.reset()
    } catch (err) {
      setError(err.message)
    } finally {
      setUploading(false)
    }
  }

  const remaining = question.time_limit_seconds - recorder.seconds

  return (
    <div className="card p-6">
      <div className="mb-1 flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wide text-ink-500">
          Question {index + 1} of {total}
        </span>
        <span className="text-xs text-ink-500">Up to {Math.round(question.time_limit_seconds / 60)} min</span>
      </div>
      <h2 className="text-lg font-semibold leading-snug text-ink-900">{question.text}</h2>

      {error ? (
        <div className="mt-4">
          <Alert tone="error">{error}</Alert>
        </div>
      ) : null}
      {recorder.error ? (
        <div className="mt-4">
          <Alert tone="error">{recorder.error}</Alert>
        </div>
      ) : null}

      <div className="mt-6 flex flex-col items-center gap-4 rounded-xl border border-dashed border-ink-200 bg-ink-50/60 px-5 py-8">
        {recorder.isRecording ? (
          <>
            <div className="relative grid h-20 w-20 place-items-center">
              <span
                className="absolute inset-0 rounded-full bg-red-400/40 animate-pulseRing"
                style={{ transform: `scale(${1 + recorder.level * 0.5})` }}
              />
              <span className="relative grid h-16 w-16 place-items-center rounded-full bg-red-500 text-white">
                <svg viewBox="0 0 24 24" className="h-7 w-7" fill="currentColor">
                  <rect x="7" y="7" width="10" height="10" rx="2" />
                </svg>
              </span>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold tabular-nums">{formatDuration(recorder.seconds)}</p>
              <p className="text-xs text-ink-500">
                {remaining <= 30 ? `${remaining}s left — wrap up` : 'Recording… speak naturally'}
              </p>
            </div>
            <button type="button" onClick={recorder.stop} className="btn-dark">
              Stop recording
            </button>
          </>
        ) : blob ? (
          <>
            <audio controls src={previewUrl} className="w-full max-w-sm" />
            <p className="text-xs text-ink-500">
              {formatDuration(blob.seconds)} recorded. Listen back before you save it.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              <button
                type="button"
                className="btn-ghost"
                onClick={() => {
                  setBlob(null)
                  recorder.reset()
                }}
                disabled={uploading}
              >
                Record again
              </button>
              <button type="button" className="btn-primary" onClick={upload} disabled={uploading}>
                {uploading ? <Spinner className="h-4 w-4" /> : null}
                {uploading ? 'Saving…' : 'Save answer'}
              </button>
            </div>
          </>
        ) : savedSeconds != null ? (
          <>
            <div className="grid h-14 w-14 place-items-center rounded-full bg-emerald-100 text-emerald-600">
              <svg viewBox="0 0 24 24" className="h-7 w-7" fill="none" stroke="currentColor" strokeWidth="2.2">
                <path d="m5 13 4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <p className="text-sm font-medium text-ink-900">Answer saved ({formatDuration(savedSeconds)})</p>
            <button type="button" className="btn-ghost" onClick={recorder.start}>
              Re-record this answer
            </button>
          </>
        ) : (
          <>
            <button
              type="button"
              onClick={recorder.start}
              disabled={recorder.state === 'requesting'}
              className="grid h-20 w-20 place-items-center rounded-full bg-blue-600 text-white transition hover:bg-blue-700 disabled:opacity-60"
              aria-label="Start recording"
            >
              {recorder.state === 'requesting' ? (
                <Spinner className="h-7 w-7" />
              ) : (
                <svg viewBox="0 0 24 24" className="h-8 w-8" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 1 0-6 0v6a3 3 0 0 0 3 3Z" fill="currentColor" stroke="none" />
                  <path d="M6 11v1a6 6 0 0 0 12 0v-1M12 18v3" strokeLinecap="round" />
                </svg>
              )}
            </button>
            <p className="text-sm text-ink-500">
              {recorder.state === 'requesting' ? 'Waiting for microphone access…' : 'Tap to start recording'}
            </p>
          </>
        )}
      </div>
    </div>
  )
}

/* -------------------------------------------------------------------- page */
export default function Interview() {
  const { token } = useParams()
  const [session, setSession] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [stage, setStage] = useState('intro') // intro | resume | questions | review | done
  const [answered, setAnswered] = useState({})
  const [current, setCurrent] = useState(0)
  const [resumeText, setResumeText] = useState('')
  const [resumeFile, setResumeFile] = useState(null)
  const [resumeSaved, setResumeSaved] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [status, setStatus] = useState(null)

  useEffect(() => {
    api
      .session(token)
      .then((data) => {
        setSession(data)
        setResumeSaved(data.resume_on_file)
        if (['submitted', 'processing', 'completed', 'failed'].includes(data.status)) setStage('done')
      })
      .catch((err) => setLoadError(err.message))
  }, [token])

  // While the analysis runs, poll so the candidate sees it finish.
  useEffect(() => {
    if (stage !== 'done') return undefined
    let cancelled = false
    const poll = () =>
      api
        .candidateStatus(token)
        .then((data) => !cancelled && setStatus(data))
        .catch(() => {})
    poll()
    const timer = setInterval(() => {
      if (status && ['completed', 'failed'].includes(status.status)) return
      poll()
    }, 6000)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [stage, token, status])

  const questions = useMemo(() => session?.questions || [], [session])
  const allAnswered = useMemo(
    () => questions.length > 0 && questions.every((question) => answered[question.id] != null),
    [questions, answered],
  )

  async function begin() {
    setBusy(true)
    setError('')
    try {
      await api.startSession(token)
      setStage(session.resume_on_file ? 'questions' : 'resume')
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  async function saveResume(event) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const form = new FormData()
      if (resumeFile) form.append('resume_file', resumeFile)
      else form.append('resume_text', resumeText)
      await api.uploadResume(token, form)
      setResumeSaved(true)
      setStage('questions')
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  async function submit() {
    setBusy(true)
    setError('')
    try {
      await api.submit(token)
      setStage('done')
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  if (loadError) {
    return (
      <Shell company="Interview">
        <div className="card p-8 text-center">
          <h1 className="text-lg font-semibold">This interview link is not valid</h1>
          <p className="mt-2 text-sm text-ink-500">{loadError}</p>
          <p className="mt-4 text-sm text-ink-500">
            Links expire for security. Reply to the invitation email and ask for a fresh one.
          </p>
        </div>
      </Shell>
    )
  }

  if (!session) {
    return (
      <Shell company="Interview">
        <Loading label="Opening your interview…" />
      </Shell>
    )
  }

  const stepNumber = { intro: 1, resume: 2, questions: 3, review: 4, done: 4 }[stage]

  /* ------------------------------------------------------------------ done */
  if (stage === 'done') {
    const finished = status?.status === 'completed'
    const failed = status?.status === 'failed'
    return (
      <Shell company={session.company_name}>
        <div className="card p-8 text-center">
          <div
            className={`mx-auto grid h-14 w-14 place-items-center rounded-full ${
              failed ? 'bg-amber-100 text-amber-600' : 'bg-emerald-100 text-emerald-600'
            }`}
          >
            {failed ? (
              <span className="text-2xl">!</span>
            ) : (
              <svg viewBox="0 0 24 24" className="h-7 w-7" fill="none" stroke="currentColor" strokeWidth="2.2">
                <path d="m5 13 4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            )}
          </div>
          <h1 className="mt-5 text-xl font-bold">
            {failed ? 'We hit a problem analysing your answers' : 'Thanks — your interview is in.'}
          </h1>
          <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-ink-600">
            {failed
              ? 'Your recordings were received, but the analysis did not complete. The hiring team has been notified and will follow up.'
              : finished
                ? `Your feedback report has been emailed to ${session.candidate_email}. Check your spam folder if it has not arrived in a few minutes.`
                : `We are transcribing and scoring your answers now. Your report will arrive at ${session.candidate_email} shortly — you can close this page.`}
          </p>

          {!finished && !failed ? (
            <div className="mx-auto mt-6 flex max-w-xs items-center justify-center gap-3 rounded-lg bg-ink-50 px-4 py-3 text-sm text-ink-600">
              <Spinner className="h-4 w-4" />
              {status?.message || 'Analysing your answers…'}
            </div>
          ) : null}

          {finished && status?.overall_score != null ? (
            <div className="mx-auto mt-6 max-w-xs rounded-xl bg-ink-900 px-6 py-5 text-white">
              <p className="text-xs uppercase tracking-wide text-ink-400">Your overall score</p>
              <p className="mt-1 text-4xl font-bold tabular-nums">{status.overall_score.toFixed(1)}<span className="text-lg opacity-60">/10</span></p>
              <p className="mt-1 text-xs text-ink-300">Full breakdown is in your email.</p>
            </div>
          ) : null}
        </div>
      </Shell>
    )
  }

  return (
    <Shell company={session.company_name} step={stepNumber} totalSteps={4}>
      {error ? (
        <div className="mb-5">
          <Alert tone="error" onDismiss={() => setError('')}>
            {error}
          </Alert>
        </div>
      ) : null}

      {/* ----------------------------------------------------------- intro */}
      {stage === 'intro' ? (
        <div className="card p-7 sm:p-9">
          <p className="text-sm font-medium text-blue-600">{session.company_name}</p>
          <h1 className="mt-1 text-2xl font-bold tracking-tight">Hi {session.candidate_name.split(' ')[0]},</h1>
          <p className="mt-3 text-sm leading-relaxed text-ink-600">
            This is your recorded interview for the <strong>{session.role_title}</strong> role. You will answer{' '}
            {questions.length} question{questions.length === 1 ? '' : 's'} out loud, one at a time. There is no live
            interviewer and you can re-record any answer before you submit.
          </p>

          {session.role_description ? (
            <div className="mt-5 rounded-lg bg-ink-50 px-4 py-3.5 text-sm leading-relaxed text-ink-700">
              {session.role_description}
            </div>
          ) : null}

          <ul className="mt-6 space-y-3 text-sm text-ink-700">
            {[
              'Find a quiet room and use a headset if you have one.',
              'Allow microphone access when your browser asks.',
              'Answer naturally — you can listen back and re-record.',
              'Your report is emailed to you as soon as the analysis finishes.',
            ].map((line) => (
              <li key={line} className="flex gap-3">
                <svg viewBox="0 0 24 24" className="mt-0.5 h-4 w-4 shrink-0 text-blue-500" fill="none" stroke="currentColor" strokeWidth="2.2">
                  <path d="m5 13 4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                {line}
              </li>
            ))}
          </ul>

          {!recordingSupported ? (
            <div className="mt-6">
              <Alert tone="error" title="This browser cannot record audio">
                Please open this link in the latest Chrome, Edge, Firefox or Safari — on a device with a microphone.
              </Alert>
            </div>
          ) : null}

          <button type="button" onClick={begin} className="btn-dark mt-7 w-full sm:w-auto" disabled={busy || !recordingSupported}>
            {busy ? <Spinner className="h-4 w-4" /> : null}
            {busy ? 'Starting…' : 'Start interview'}
          </button>
        </div>
      ) : null}

      {/* ---------------------------------------------------------- resume */}
      {stage === 'resume' ? (
        <form onSubmit={saveResume} className="card p-7">
          <h1 className="text-xl font-bold tracking-tight">Add your resume</h1>
          <p className="mt-2 text-sm text-ink-600">
            We compare the skills on your resume with the ones you talk about, so this step matters for your score.
          </p>

          <div className="mt-6">
            <label className="label" htmlFor="resume-file">
              Upload a file (PDF, DOCX or TXT)
            </label>
            <input
              id="resume-file"
              type="file"
              accept=".pdf,.txt,.md,.docx"
              className="block w-full text-sm text-ink-600 file:mr-3 file:rounded-lg file:border-0 file:bg-ink-900 file:px-4 file:py-2.5 file:text-sm file:font-semibold file:text-white hover:file:bg-ink-800"
              onChange={(event) => setResumeFile(event.target.files?.[0] || null)}
            />
          </div>

          <div className="my-5 flex items-center gap-3 text-xs uppercase tracking-wide text-ink-400">
            <span className="h-px flex-1 bg-ink-200" /> or paste it <span className="h-px flex-1 bg-ink-200" />
          </div>

          <textarea
            rows={8}
            className="input"
            placeholder="Paste your resume text here…"
            value={resumeText}
            onChange={(event) => setResumeText(event.target.value)}
            disabled={Boolean(resumeFile)}
          />

          <button
            type="submit"
            className="btn-dark mt-6 w-full sm:w-auto"
            disabled={busy || (!resumeFile && resumeText.trim().length < 40)}
          >
            {busy ? <Spinner className="h-4 w-4" /> : null}
            {busy ? 'Saving…' : 'Continue to questions'}
          </button>
        </form>
      ) : null}

      {/* ------------------------------------------------------- questions */}
      {stage === 'questions' ? (
        <div className="space-y-5">
          <div className="flex items-center gap-2">
            {questions.map((question, index) => (
              <button
                key={question.id}
                type="button"
                onClick={() => setCurrent(index)}
                className={`h-1.5 flex-1 rounded-full transition-colors ${
                  answered[question.id] != null
                    ? 'bg-emerald-500'
                    : index === current
                      ? 'bg-blue-500'
                      : 'bg-ink-200'
                }`}
                aria-label={`Go to question ${index + 1}`}
              />
            ))}
          </div>

          <QuestionRecorder
            key={questions[current].id}
            token={token}
            question={questions[current]}
            index={current}
            total={questions.length}
            existing={answered[questions[current].id] != null ? { seconds: answered[questions[current].id] } : null}
            onSaved={(questionId, seconds) => {
              setAnswered((current_) => ({ ...current_, [questionId]: seconds }))
              if (current < questions.length - 1) setTimeout(() => setCurrent((value) => value + 1), 600)
            }}
          />

          <div className="flex items-center justify-between gap-3">
            <button
              type="button"
              className="btn-ghost"
              disabled={current === 0}
              onClick={() => setCurrent((value) => value - 1)}
            >
              Previous
            </button>
            {current < questions.length - 1 ? (
              <button type="button" className="btn-ghost" onClick={() => setCurrent((value) => value + 1)}>
                Next question
              </button>
            ) : (
              <button type="button" className="btn-primary" disabled={!allAnswered} onClick={() => setStage('review')}>
                Review &amp; submit
              </button>
            )}
          </div>

          {!allAnswered ? (
            <p className="text-center text-xs text-ink-500">
              {questions.length - Object.keys(answered).length} question(s) still need a recording.
            </p>
          ) : null}
        </div>
      ) : null}

      {/* ---------------------------------------------------------- review */}
      {stage === 'review' ? (
        <div className="card p-7">
          <h1 className="text-xl font-bold tracking-tight">Ready to submit</h1>
          <p className="mt-2 text-sm text-ink-600">
            Once you submit, your answers are analysed and the report is emailed to{' '}
            <strong>{session.candidate_email}</strong>. You cannot re-record after this point.
          </p>

          <ul className="mt-6 divide-y divide-ink-100 border-y border-ink-100">
            {questions.map((question, index) => (
              <li key={question.id} className="flex items-center gap-3 py-3">
                <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-emerald-100 text-xs font-semibold text-emerald-700">
                  ✓
                </span>
                <span className="min-w-0 flex-1 truncate text-sm text-ink-700">
                  Q{index + 1}. {question.text}
                </span>
                <span className="shrink-0 text-xs tabular-nums text-ink-500">
                  {formatDuration(answered[question.id])}
                </span>
              </li>
            ))}
          </ul>

          <div className="mt-4 flex items-center gap-2 text-sm text-ink-600">
            <span className="grid h-6 w-6 place-items-center rounded-full bg-emerald-100 text-xs font-semibold text-emerald-700">✓</span>
            Resume {resumeSaved ? 'on file' : 'missing'}
          </div>

          <div className="mt-7 flex flex-wrap gap-2">
            <button type="button" className="btn-ghost" onClick={() => setStage('questions')} disabled={busy}>
              Back to questions
            </button>
            <button type="button" className="btn-dark" onClick={submit} disabled={busy || !allAnswered}>
              {busy ? <Spinner className="h-4 w-4" /> : null}
              {busy ? 'Submitting…' : 'Submit interview'}
            </button>
          </div>
        </div>
      ) : null}
    </Shell>
  )
}
