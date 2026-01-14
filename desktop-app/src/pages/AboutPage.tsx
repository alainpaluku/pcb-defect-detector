import { CpuChipIcon } from '@heroicons/react/24/outline'
import { Card } from '../components/ui'
import { DEFECT_LABELS, DEFECT_COLORS } from '../constants'

interface AboutPageProps {
  darkMode: boolean
}

const TECHNOLOGIES = ['React', 'TypeScript', 'Tailwind CSS', 'ONNX Runtime', 'Tauri', 'MobileNetV2']

export function AboutPage({ darkMode }: AboutPageProps) {
  const openGitHub = () => {
    window.open('https://github.com/alainpaluku/pcb-defect-detector', '_blank')
  }

  return (
    <div className="flex-1 p-5 overflow-y-auto">
      <div className="max-w-2xl space-y-5">
        {/* Header */}
        <Card darkMode={darkMode} className="p-6">
          <div className="flex items-start gap-4">
            <div className="w-16 h-16 bg-gradient-to-br from-lime-400 to-emerald-500 rounded-2xl flex items-center justify-center flex-shrink-0">
              <CpuChipIcon className="w-9 h-9 text-white" />
            </div>
            <div>
              <h2 className={`text-xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                PCB Inspector
              </h2>
              <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                Version 1.0.0
              </p>
              <p className={`mt-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                Application de détection automatique de défauts sur circuits imprimés (PCB) 
                utilisant l'intelligence artificielle et le deep learning.
              </p>
            </div>
          </div>
        </Card>

        {/* Description */}
        <Card darkMode={darkMode} className="p-5">
          <h3 className={`font-semibold mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            À propos du projet
          </h3>
          <div className={`space-y-3 text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            <p>
              PCB Inspector est un système d'inspection optique automatisée (AOI) qui utilise 
              un réseau de neurones convolutif (CNN) basé sur MobileNetV2 pour détecter 
              les défauts de fabrication sur les circuits imprimés.
            </p>
            <p>
              Le modèle a été entraîné sur le dataset PCB Defects de Kaggle et peut identifier 
              6 types de défauts différents avec une précision élevée.
            </p>
          </div>
        </Card>

        {/* Défauts détectés */}
        <Card darkMode={darkMode} className="p-5">
          <h3 className={`font-semibold mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Types de défauts détectés
          </h3>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(DEFECT_LABELS).map(([key, label]) => (
              <div key={key} className="flex items-center gap-2">
                <span className={`w-2.5 h-2.5 rounded-full ${DEFECT_COLORS[key]}`} />
                <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>{label}</span>
              </div>
            ))}
          </div>
        </Card>

        {/* Technologies */}
        <Card darkMode={darkMode} className="p-5">
          <h3 className={`font-semibold mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Technologies utilisées
          </h3>
          <div className="flex flex-wrap gap-2">
            {TECHNOLOGIES.map((tech) => (
              <span
                key={tech}
                className={`px-3 py-1 rounded-full text-sm ${
                  darkMode ? 'bg-gray-700 text-gray-300' : 'bg-gray-100 text-gray-700'
                }`}
              >
                {tech}
              </span>
            ))}
          </div>
        </Card>

        {/* GitHub */}
        <Card darkMode={darkMode} className="p-5">
          <h3 className={`font-semibold mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Code source
          </h3>
          <p className={`text-sm mb-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            Le projet est open source et disponible sur GitHub.
          </p>
          <button
            onClick={openGitHub}
            className="flex items-center gap-2 px-4 py-2 bg-gray-900 hover:bg-gray-800 text-white rounded-lg transition-colors"
          >
            <GitHubIcon />
            Voir sur GitHub
          </button>
        </Card>

        {/* Auteur */}
        <Card darkMode={darkMode} className="p-5">
          <h3 className={`font-semibold mb-3 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            Auteur
          </h3>
          <p className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            Développé par <strong>Alain Paluku</strong>
          </p>
          <p className={`text-xs mt-2 ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
            © 2024 - Licence MIT
          </p>
        </Card>
      </div>
    </div>
  )
}

function GitHubIcon() {
  return (
    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
      <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
    </svg>
  )
}
