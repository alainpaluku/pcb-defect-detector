interface ProgressBarProps {
  label: string
  value: number
  color: string
  darkMode: boolean
}

export function ProgressBar({ label, value, color, darkMode }: ProgressBarProps) {
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className={`flex items-center gap-1.5 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
          <span className={`w-2 h-2 rounded-full ${color}`} />
          {label}
        </span>
        <span className={darkMode ? 'text-gray-300' : 'text-gray-700'}>
          {(value * 100).toFixed(1)}%
        </span>
      </div>
      <div className={`h-1.5 rounded-full ${darkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>
        <div
          className={`h-full rounded-full ${color} transition-all duration-500`}
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  )
}
