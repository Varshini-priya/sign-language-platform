import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Navbar from './components/shared/Navbar'
import LoginPage from './pages/LoginPage'
import RegisterPage from './pages/RegisterPage'
import DashboardPage from './pages/DashboardPage'
import RecognitionPage from './pages/RecognitionPage'
import HistoryPage from './pages/HistoryPage'
import SettingsPage from './pages/SettingsPage'
import { useAuthStore } from './store/authStore'

function ProtectedRoute({ children }) {
  const token = useAuthStore((s) => s.token)
  return token ? children : <Navigate to="/login" replace />
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100">
        <Navbar />
        <main className="pt-16">
          <Routes>
            {/* Public */}
            <Route path="/login"    element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />

            {/* Protected */}
            <Route path="/dashboard"  element={<ProtectedRoute><DashboardPage /></ProtectedRoute>} />
            <Route path="/recognize"  element={<ProtectedRoute><RecognitionPage /></ProtectedRoute>} />
            <Route path="/history"    element={<ProtectedRoute><HistoryPage /></ProtectedRoute>} />
            <Route path="/settings"   element={<ProtectedRoute><SettingsPage /></ProtectedRoute>} />

            {/* Default */}
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
