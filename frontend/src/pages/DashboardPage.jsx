export default function DashboardPage() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      <h1 className="text-2xl font-semibold mb-2">Dashboard</h1>
      <p className="text-gray-500 dark:text-gray-400 mb-8">Your session analytics will appear here in Week 11.</p>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[
          { label: 'Total Sessions', value: '0', icon: '📊' },
          { label: 'Signs Recognized', value: '0', icon: '🤟' },
          { label: 'Avg Accuracy', value: '—', icon: '🎯' },
        ].map(({ label, value, icon }) => (
          <div key={label} className="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-5">
            <div className="text-2xl mb-2">{icon}</div>
            <div className="text-3xl font-semibold">{value}</div>
            <div className="text-sm text-gray-500 dark:text-gray-400 mt-1">{label}</div>
          </div>
        ))}
      </div>
    </div>
  )
}