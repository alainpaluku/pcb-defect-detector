import { useState } from 'react'
import { Card, CardContent, Toggle } from '../components/ui'

interface SettingsPageProps {
  darkMode: boolean
}

export function SettingsPage({ darkMode }: SettingsPageProps) {
  const [confidenceThreshold, setConfidenceThreshold] = useState(70)
  const [autoSave, setAutoSave] = useState(true)
  const [showNotifications, setShowNotifications] = useState(true)
  const [historyLimit, setHistoryLimit] = useState(50)

  const inputClass = darkMode 
    ? 'bg-gray-700 border-gray-600 text-white' 
    : 'bg-white border-gray-300 text-gray-900'

  return (
    <div className="flex-1 p-5 overflow-y-auto">
      <div className="max-w-2xl space-y-5">
        {/* Détection */}
        <Card darkMode={darkMode} className="p-5">
          <h3 className={`font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Paramètres de détection
          </h3>
          
          <div className="space-y-4">
            <div>
              <label className={`block text-sm mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                Seuil de confiance pour défaut: <span className="font-semibold text-lime-500">{confidenceThreshold}%</span>
              </label>
              <input
                type="range"
                min="50"
                max="95"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-lime-500"
              />
              <p className={`text-xs mt-1 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                Une image est considérée défectueuse si la confiance dépasse ce seuil
              </p>
            </div>
          </div>
        </Card>

        {/* Historique */}
        <Card darkMode={darkMode} className="p-5">
          <h3 className={`font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Historique
          </h3>
          
          <div className="space-y-4">
            <div>
              <label className={`block text-sm mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                Nombre maximum d'inspections conservées
              </label>
              <select
                value={historyLimit}
                onChange={(e) => setHistoryLimit(Number(e.target.value))}
                className={`w-full px-3 py-2 rounded-lg border ${inputClass}`}
              >
                <option value={25}>25 inspections</option>
                <option value={50}>50 inspections</option>
                <option value={100}>100 inspections</option>
                <option value={200}>200 inspections</option>
              </select>
            </div>

            <SettingRow
              title="Sauvegarde automatique"
              description="Sauvegarder les résultats automatiquement"
              darkMode={darkMode}
            >
              <Toggle enabled={autoSave} onChange={setAutoSave} />
            </SettingRow>
          </div>
        </Card>

        {/* Notifications */}
        <Card darkMode={darkMode} className="p-5">
          <h3 className={`font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Notifications
          </h3>
          
          <SettingRow
            title="Alertes de défauts"
            description="Afficher une notification quand un défaut est détecté"
            darkMode={darkMode}
          >
            <Toggle enabled={showNotifications} onChange={setShowNotifications} />
          </SettingRow>
        </Card>

        {/* Modèle */}
        <Card darkMode={darkMode} className="p-5">
          <h3 className={`font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Modèle IA
          </h3>
          
          <div className="space-y-3">
            <InfoRow label="Architecture" value="MobileNetV2" darkMode={darkMode} />
            <InfoRow label="Format" value="ONNX" darkMode={darkMode} />
            <InfoRow label="Runtime" value="ONNX Runtime Web" darkMode={darkMode} />
            <InfoRow label="Classes" value="6 types de défauts" darkMode={darkMode} />
            <InfoRow label="Taille d'entrée" value="224 × 224 px" darkMode={darkMode} />
          </div>
        </Card>
      </div>
    </div>
  )
}

function SettingRow({ title, description, darkMode, children }: {
  title: string
  description: string
  darkMode: boolean
  children: React.ReactNode
}) {
  return (
    <div className="flex items-center justify-between">
      <div>
        <p className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>{title}</p>
        <p className={`text-xs ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>{description}</p>
      </div>
      {children}
    </div>
  )
}

function InfoRow({ label, value, darkMode }: { label: string; value: string; darkMode: boolean }) {
  return (
    <div className="flex justify-between">
      <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>{label}</span>
      <span className={`text-sm font-medium ${darkMode ? 'text-white' : 'text-gray-900'}`}>{value}</span>
    </div>
  )
}
