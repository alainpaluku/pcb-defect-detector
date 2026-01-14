interface ToggleProps {
  enabled: boolean
  onChange: (value: boolean) => void
}

export function Toggle({ enabled, onChange }: ToggleProps) {
  return (
    <button
      onClick={() => onChange(!enabled)}
      className={`w-12 h-6 rounded-full transition-colors ${enabled ? 'bg-lime-500' : 'bg-gray-600'}`}
    >
      <div className={`w-5 h-5 rounded-full bg-white shadow transform transition-transform ${enabled ? 'translate-x-6' : 'translate-x-0.5'}`} />
    </button>
  )
}
