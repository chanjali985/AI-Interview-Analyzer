export const VERDICT_LABELS = {
  strong_hire: 'Strong match',
  shortlist: 'Shortlisted',
  review: 'Needs review',
  not_recommended: 'Not a match',
}

export const STATUS_LABELS = {
  invited: 'Invited',
  in_progress: 'In progress',
  submitted: 'Submitted',
  processing: 'Analysing',
  completed: 'Completed',
  failed: 'Failed',
  expired: 'Expired',
}

export const STATUS_STYLES = {
  invited: 'bg-ink-100 text-ink-700',
  in_progress: 'bg-amber-100 text-amber-800',
  submitted: 'bg-blue-100 text-blue-800',
  processing: 'bg-blue-100 text-blue-800',
  completed: 'bg-emerald-100 text-emerald-800',
  failed: 'bg-red-100 text-red-800',
  expired: 'bg-ink-200 text-ink-600',
}

export const DIMENSION_LABELS = {
  communication: 'Communication',
  technical_relevance: 'Technical relevance',
  confidence: 'Confidence',
  clarity: 'Clarity',
  overall_quality: 'Overall quality',
}

export function scoreColor(score) {
  if (score == null) return 'text-ink-400'
  if (score >= 8) return 'text-emerald-600'
  if (score >= 6.5) return 'text-blue-600'
  if (score >= 5) return 'text-amber-600'
  return 'text-red-600'
}

export function scoreBarColor(score) {
  if (score >= 8) return 'bg-emerald-500'
  if (score >= 6.5) return 'bg-blue-500'
  if (score >= 5) return 'bg-amber-500'
  return 'bg-red-500'
}

export function formatDate(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}

export function formatDateTime(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleString(undefined, {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return '—'
  const total = Math.round(seconds)
  const minutes = Math.floor(total / 60)
  const rest = total % 60
  return `${minutes}:${String(rest).padStart(2, '0')}`
}

export function initials(name = '') {
  return name
    .split(' ')
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0].toUpperCase())
    .join('')
}
