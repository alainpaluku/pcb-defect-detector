import { useState, useEffect } from 'react'
import { Page, InspectionItem } from './types'
import { Sidebar } from './components/Sidebar'
import { HomePage, HistoryPage, SettingsPage, AboutPage } from './pages'
import { loadOnnxModel } from './utils'

export default function App() {
  const [darkMode, setDarkMode] = useState(true)
  const [currentPage, setCurrentPage] = useState<Page>('home')
  const [modelLoaded, setModelLoaded] = useState(false)
  const [history, setHistory] = useState<InspectionItem[]>([])

  // Charger le modèle ONNX au démarrage
  useEffect(() => {
    const initModel = async () => {
      try {
        const success = await loadOnnxModel('/model/pcb_model.onnx')
        setModelLoaded(true)
        console.log(success ? '✅ Modèle ONNX prêt' : '⚠️ Mode démo (modèle non trouvé)')
      } catch (error) {
        console.error('Erreur chargement modèle:', error)
        setModelLoaded(true) // Permettre l'utilisation en mode mock
      }
    }
    
    initModel()
  }, [])

  return (
    <div className={`h-screen flex ${darkMode ? 'bg-gray-900' : 'bg-gray-100'}`}>
      <Sidebar
        currentPage={currentPage}
        setCurrentPage={setCurrentPage}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
        modelLoaded={modelLoaded}
      />

      <main className="flex-1 flex flex-col overflow-hidden">
        {currentPage === 'home' && (
          <HomePage 
            darkMode={darkMode} 
            modelLoaded={modelLoaded} 
            setHistory={setHistory} 
          />
        )}
        {currentPage === 'history' && (
          <HistoryPage 
            darkMode={darkMode} 
            history={history} 
            setHistory={setHistory} 
          />
        )}
        {currentPage === 'settings' && (
          <SettingsPage darkMode={darkMode} />
        )}
        {currentPage === 'about' && (
          <AboutPage darkMode={darkMode} />
        )}
      </main>
    </div>
  )
}
