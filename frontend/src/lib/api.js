const TOKEN_KEY = 'aia.token'

export class ApiError extends Error {
  constructor(message, status, payload) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.payload = payload
  }
}

export const auth = {
  get token() {
    try {
      return localStorage.getItem(TOKEN_KEY)
    } catch {
      return null
    }
  },
  set(token) {
    try {
      localStorage.setItem(TOKEN_KEY, token)
    } catch {
      /* private mode: session-only auth */
    }
  },
  clear() {
    try {
      localStorage.removeItem(TOKEN_KEY)
    } catch {
      /* ignore */
    }
  },
}

async function parse(response) {
  const text = await response.text()
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

async function request(path, { method = 'GET', body, headers = {}, form, interviewToken, signal } = {}) {
  const finalHeaders = { ...headers }
  const token = auth.token
  if (token && !interviewToken) finalHeaders.Authorization = `Bearer ${token}`
  if (interviewToken) finalHeaders['X-Interview-Token'] = interviewToken

  let payload
  if (form) {
    payload = form
  } else if (body !== undefined) {
    finalHeaders['Content-Type'] = 'application/json'
    payload = JSON.stringify(body)
  }

  const response = await fetch(`/api${path}`, { method, headers: finalHeaders, body: payload, signal })
  const data = await parse(response)

  if (!response.ok) {
    if (response.status === 401 && !interviewToken) auth.clear()
    const detail =
      (data && typeof data === 'object' && (data.detail?.detail || data.detail)) ||
      (typeof data === 'string' ? data : null) ||
      `Request failed (${response.status})`
    throw new ApiError(typeof detail === 'string' ? detail : JSON.stringify(detail), response.status, data)
  }
  return data
}

export const api = {
  // meta
  config: () => request('/config'),
  health: () => request('/health'),

  // auth
  login: (email, password) => request('/auth/login', { method: 'POST', body: { email, password } }),
  me: () => request('/auth/me'),

  // roles
  listRoles: () => request('/roles'),
  getRole: (id) => request(`/roles/${id}`),
  createRole: (payload) => request('/roles', { method: 'POST', body: payload }),
  updateRole: (id, payload) => request(`/roles/${id}`, { method: 'PATCH', body: payload }),
  deleteRole: (id) => request(`/roles/${id}`, { method: 'DELETE' }),

  // interviews
  dashboard: () => request('/dashboard'),
  listInterviews: (params = {}) => {
    const query = new URLSearchParams(
      Object.entries(params).filter(([, value]) => value !== '' && value != null),
    ).toString()
    return request(`/interviews${query ? `?${query}` : ''}`)
  },
  getInterview: (id) => request(`/interviews/${id}`),
  invite: (payload) => request('/interviews/invite', { method: 'POST', body: payload }),
  regenerateLink: (id, sendEmail = false) =>
    request(`/interviews/${id}/invite-link?send_email=${sendEmail}`, { method: 'POST' }),
  reanalyze: (id) => request(`/interviews/${id}/reanalyze`, { method: 'POST' }),
  resendReport: (id) => request(`/interviews/${id}/resend-report`, { method: 'POST' }),
  deleteInterview: (id) => request(`/interviews/${id}`, { method: 'DELETE' }),
  audioUrl: (interviewId, answerId) => `/api/interviews/${interviewId}/answers/${answerId}/audio`,
  reportPdfUrl: (id) => `/api/interviews/${id}/report.pdf`,
  emailStatus: () => request('/health/email'),

  // candidate (token-authenticated)
  session: (token) => request('/public/session', { interviewToken: token }),
  startSession: (token) => request('/public/session/start', { method: 'POST', interviewToken: token }),
  uploadResume: (token, form) => request('/public/resume', { method: 'POST', form, interviewToken: token }),
  uploadAnswer: (token, questionId, form) =>
    request(`/public/answers/${questionId}`, { method: 'POST', form, interviewToken: token }),
  submit: (token) => request('/public/submit', { method: 'POST', interviewToken: token }),
  candidateStatus: (token) => request('/public/status', { interviewToken: token }),
}

/**
 * Download a protected file (PDF, audio) using the bearer token.
 */
export async function downloadWithAuth(url, filename) {
  const response = await fetch(url, { headers: { Authorization: `Bearer ${auth.token}` } })
  if (!response.ok) throw new ApiError('Download failed', response.status)
  const blob = await response.blob()
  const objectUrl = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = objectUrl
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(objectUrl)
}

export async function authedBlobUrl(url) {
  const response = await fetch(url, { headers: { Authorization: `Bearer ${auth.token}` } })
  if (!response.ok) throw new ApiError('Could not load recording', response.status)
  return URL.createObjectURL(await response.blob())
}
