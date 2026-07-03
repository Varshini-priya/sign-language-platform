import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuthStore } from '../../store/authStore'

const NAV_LINKS = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/recognize',  label: 'Recognize' },
  { to: '/history',   label: 'History' },
  { to: '/settings',  label: 'Settings' },
]

export default function Navbar() {
  const { pathname } = useLocation()
  const navigate = useNavigate()
  const { user, token, logout } = useAuthStore()

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <nav className="fixed top-0 inset-x-0 z-50 h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 flex items-center px-6 gap-8">
      {/* Logo */}
      <Link to="/" className="flex items-center gap-2 font-semibold text-brand-600 text-lg flex-shrink-0">
        <span className="text-2xl">🤟</span>
        <span>SignAI</span>
      </Link>

      {/* Links — only show when logged in */}
      {token && (
        <div className="flex items-center gap-1 flex-1">
          {NAV_LINKS.map(({ to, label }) => (
            <Link
              key={to}
              to={to}
              className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                pathname === to
                  ? 'bg-brand-50 text-brand-700 dark:bg-brand-900/30 dark:text-brand-300 font-medium'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800'
              }`}
            >
              {label}
            </Link>
          ))}
        </div>
      )}

      {/* Right side */}
      <div className="ml-auto flex items-center gap-3">
        {token ? (
          <>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {user?.username || 'User'}
            </span>
            <button
              onClick={handleLogout}
              className="text-sm px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
            >
              Logout
            </button>
          </>
        ) : (
          <Link
            to="/login"
            className="text-sm px-4 py-1.5 bg-brand-600 hover:bg-brand-700 text-white rounded-lg transition-colors"
          >
            Login
          </Link>
        )}
      </div>
    </nav>
  )
}
