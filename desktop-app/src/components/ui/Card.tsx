import { ReactNode } from 'react'

interface CardProps {
  children: ReactNode
  darkMode: boolean
  className?: string
}

export function Card({ children, darkMode, className = '' }: CardProps) {
  const baseClass = darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white border-gray-200'
  return (
    <div className={`rounded-xl border ${baseClass} ${className}`}>
      {children}
    </div>
  )
}

interface CardHeaderProps {
  children: ReactNode
  darkMode: boolean
  actions?: ReactNode
}

export function CardHeader({ children, darkMode, actions }: CardHeaderProps) {
  return (
    <div className={`px-4 py-3 border-b flex items-center justify-between ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
      <h2 className={`font-semibold flex items-center gap-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
        {children}
      </h2>
      {actions}
    </div>
  )
}

interface CardContentProps {
  children: ReactNode
  className?: string
}

export function CardContent({ children, className = '' }: CardContentProps) {
  return <div className={`p-4 ${className}`}>{children}</div>
}
