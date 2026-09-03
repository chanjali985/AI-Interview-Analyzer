import { Navigate, Route, Routes, useLocation } from 'react-router-dom'

import Layout from './components/Layout'
import { Loading } from './components/ui'
import { useAuth } from './lib/auth'
import Dashboard from './pages/Dashboard'
import Interview from './pages/Interview'
import Interviews from './pages/Interviews'
import Login from './pages/Login'
import NotFound from './pages/NotFound'
import Report from './pages/Report'
import RoleEditor from './pages/RoleEditor'
import Roles from './pages/Roles'

function RequireAuth({ children }) {
  const { user, loading } = useAuth()
  const location = useLocation()

  if (loading) return <Loading label="Starting up…" />
  if (!user) return <Navigate to="/login" state={{ from: location }} replace />
  return children
}

export default function App() {
  return (
    <Routes>
      {/* Candidate-facing, no account needed — the invite token is the credential. */}
      <Route path="/interview/:token" element={<Interview />} />

      <Route path="/login" element={<Login />} />

      <Route
        element={
          <RequireAuth>
            <Layout />
          </RequireAuth>
        }
      >
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/roles" element={<Roles />} />
        <Route path="/roles/new" element={<RoleEditor />} />
        <Route path="/roles/:id" element={<RoleEditor />} />
        <Route path="/interviews" element={<Interviews />} />
        <Route path="/reports/:id" element={<Report />} />
      </Route>

      <Route path="*" element={<NotFound />} />
    </Routes>
  )
}
