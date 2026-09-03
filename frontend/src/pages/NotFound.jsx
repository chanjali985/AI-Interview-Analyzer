import { Link } from 'react-router-dom'

export default function NotFound() {
  return (
    <div className="grid min-h-screen place-items-center px-5">
      <div className="text-center">
        <p className="text-sm font-semibold text-blue-600">404</p>
        <h1 className="mt-2 text-2xl font-bold tracking-tight">We could not find that page</h1>
        <p className="mt-2 text-sm text-ink-500">
          If you were sent an interview link, open it exactly as it appears in your email.
        </p>
        <Link to="/dashboard" className="btn-dark mt-6">
          Go to the dashboard
        </Link>
      </div>
    </div>
  )
}
