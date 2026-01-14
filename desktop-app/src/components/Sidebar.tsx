import {
  CpuChipIcon,
  ClockIcon,
  Cog6ToothIcon,
  InformationCircleIcon,
  HomeIcon,
  SunIcon,
  MoonIcon,
} from '@heroicons/react/24/outline'
import { HomeIcon as HomeIconSolid } from '@heroicons/react/24/solid'
import { Page } from '../types'

interface NavItem {
  id: Page
  icon: React.ElementType
  iconActive: React.ElementType
  label: string
}

const NAV_ITEMS: NavItem[] = [
  { id: 'home', icon: HomeIcon, iconActive: HomeIconSolid, label: 'Inspection' },
  { id: 'history', icon: ClockIcon, iconActive: ClockIcon, label: 'Historique' },
  { id: 'settings', icon: Cog6ToothIcon, iconActive: Cog6ToothIcon, label: 'Paramètres' },
  { id: 'about', icon: InformationCircleIcon, iconActive: InformationCircleIcon, label: 'À propos' },
]

interface SidebarProps {
  currentPage: Page
  setCurrentPage: (page: Page) => void
  darkMode: boolean
  setDarkMode: (v: boolean) => void
  modelLoaded: boolean
}

export function Sidebar({ currentPage, setCurrentPage, darkMode, setDarkMode, modelLoaded }: SidebarProps) {
  return (
    <aside className={`w-64 h-screen flex flex-col border-r ${darkMode ? 'bg-gray-900 border-gray-800' : 'bg-white border-gray-200'}`}>
      {/* Logo */}
      <div className={`p-5 border-b ${darkMode ? 'border-gray-800' : 'border-gray-200'}`}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-lime-400 to-emerald-500 rounded-xl flex items-center justify-center">
            <CpuChipIcon className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className={`font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>PCB Inspector</h1>
            <div className="flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${modelLoaded ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'}`} />
              <span className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                {modelLoaded ? 'Prêt' : 'Chargement...'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-3">
        <ul className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const isActive = currentPage === item.id
            const Icon = isActive ? item.iconActive : item.icon
            return (
              <li key={item.id}>
                <button
                  onClick={() => setCurrentPage(item.id)}
                  className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? darkMode ? 'bg-lime-500/10 text-lime-400' : 'bg-lime-50 text-lime-600'
                      : darkMode ? 'text-gray-400 hover:bg-gray-800 hover:text-gray-200' : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  {item.label}
                </button>
              </li>
            )
          })}
        </ul>
      </nav>

      {/* Theme Toggle */}
      <div className={`p-3 border-t ${darkMode ? 'border-gray-800' : 'border-gray-200'}`}>
        <button
          onClick={() => setDarkMode(!darkMode)}
          className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
            darkMode ? 'text-gray-400 hover:bg-gray-800' : 'text-gray-600 hover:bg-gray-100'
          }`}
        >
          {darkMode ? <SunIcon className="w-5 h-5" /> : <MoonIcon className="w-5 h-5" />}
          {darkMode ? 'Mode clair' : 'Mode sombre'}
        </button>
      </div>

      {/* Version */}
      <div className={`px-5 py-3 text-xs ${darkMode ? 'text-gray-600' : 'text-gray-400'}`}>
        Version 1.0.0
      </div>
    </aside>
  )
}
