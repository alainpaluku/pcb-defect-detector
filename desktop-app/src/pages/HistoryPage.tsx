import { ClockIcon, TrashIcon } from '@heroicons/react/24/outline'
import { InspectionItem } from '../types'
import { DEFECT_LABELS } from '../constants'
import { Card, CardHeader, CardContent } from '../components/ui'

interface HistoryPageProps {
  darkMode: boolean
  history: InspectionItem[]
  setHistory: React.Dispatch<React.SetStateAction<InspectionItem[]>>
}

export function HistoryPage({ darkMode, history, setHistory }: HistoryPageProps) {
  return (
    <div className="flex-1 p-5 overflow-hidden">
      <Card darkMode={darkMode} className="h-full flex flex-col">
        <CardHeader 
          darkMode={darkMode}
          actions={history.length > 0 && (
            <button
              onClick={() => setHistory([])}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                darkMode ? 'text-gray-400 hover:text-red-400 hover:bg-red-500/10' : 'text-gray-500 hover:text-red-500 hover:bg-red-50'
              }`}
            >
              <TrashIcon className="w-4 h-4" />
              Tout effacer
            </button>
          )}
        >
          <ClockIcon className="w-5 h-5 text-purple-500" />
          Historique des inspections
          <span className={`text-sm font-normal ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            ({history.length})
          </span>
        </CardHeader>

        <CardContent className="flex-1 overflow-y-auto">
          {history.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <ClockIcon className={`w-12 h-12 mx-auto mb-2 ${darkMode ? 'text-gray-600' : 'text-gray-300'}`} />
                <p className={darkMode ? 'text-gray-500' : 'text-gray-400'}>Aucune inspection</p>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-4 gap-4">
              {history.map((item) => (
                <HistoryCard key={item.id} item={item} darkMode={darkMode} />
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}

function HistoryCard({ item, darkMode }: { item: InspectionItem; darkMode: boolean }) {
  const isDefect = item.prediction.confidence > 0.7
  
  return (
    <div className={`rounded-lg overflow-hidden border transition-transform hover:scale-[1.02] ${
      darkMode ? 'bg-gray-800 border-gray-700' : 'bg-gray-50 border-gray-200'
    }`}>
      <div className="relative">
        <img src={item.imageSrc} alt="" className="w-full h-32 object-cover" />
        <div className={`absolute top-2 right-2 px-2 py-0.5 rounded text-xs font-medium ${
          isDefect ? 'bg-red-500 text-white' : 'bg-emerald-500 text-white'
        }`}>
          {isDefect ? 'Défaut' : 'OK'}
        </div>
      </div>
      <div className="p-3">
        <p className={`text-sm font-medium truncate ${darkMode ? 'text-white' : 'text-gray-900'}`}>
          {item.fileName}
        </p>
        <p className={`text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          {DEFECT_LABELS[item.prediction.defectType]} • {(item.prediction.confidence * 100).toFixed(0)}%
        </p>
        <p className={`text-xs mt-1 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          {item.timestamp.toLocaleString('fr-FR')}
        </p>
      </div>
    </div>
  )
}
