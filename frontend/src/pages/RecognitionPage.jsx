export default function RecognitionPage() {
  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      <h1 className="text-2xl font-semibold mb-2">Recognition</h1>
      <p className="text-gray-500 dark:text-gray-400">
        Live sign language recognition will be wired here in Week 9.
        For now, the camera and MediaPipe pipeline is built in Python (Week 2).
      </p>
      <div className="mt-6 h-80 bg-gray-900 rounded-2xl flex items-center justify-center text-gray-600">
        Camera feed placeholder
      </div>
    </div>
  )
}