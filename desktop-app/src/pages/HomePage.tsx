import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import {
  CloudArrowUpIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon,
  FolderOpenIcon,
  DocumentChartBarIcon,
  CpuChipIcon,
  EyeIcon,
  EyeSlashIcon,
} from '@heroicons/react/24/outline'
import { CheckCircleIcon as CheckCircleSolid } from '@heroicons/react/24/solid'

import { InspectionItem, PredictionResult, BoundingBox } from '../types'
import { DEFECT_LABELS, DEFECT_COLORS } from '../constants'
import { predictImage, detectDefects } from '../utils'
import { Card, CardHeader, CardContent, ProgressBar } from '../components/ui'
import { ImageWithBoxes } from '../components/ImageWithBoxes'

interface HomePageProps {
  darkMode: boolean
  modelLoaded: boolean
  setHistory: React.Dispatch<React.SetStateAction<InspectionItem[]>>
}

export function HomePage({ darkMode, modelLoaded, setHistory }: HomePageProps) {
  const [loading, setLoading] = useState(false)
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [boundingBoxes, setBoundingBoxes] = useState<BoundingBox[]>([])
  const [showBoxes, setShowBoxes] = useState(true)

  const onDrop = useCallback(
    async (files: File[]) => {
      if (!modelLoaded || files.length === 0) return
      const file = files[0]
      const reader = new FileReader()

      reader.onload = async (e) => {
        const imageSrc = e.target?.result as string
        setCurrentImage(imageSrc)
        setLoading(true)
        setBoundingBoxes([])
        
        try {
          // Lancer classification et détection en parallèle
          const [prediction, boxes] = await Promise.all([
            predictImage(imageSrc),
            detectDefects(imageSrc)
          ])
          
          // Ajouter les bounding boxes au résultat
          const fullPrediction: PredictionResult = {
            ...prediction,
            boundingBoxes: boxes
          }
          
          setResult(fullPrediction)
          setBoundingBoxes(boxes)
          
          setHistory((prev) => [
            { id: Date.now().toString(), imageSrc, fileName: file.name, prediction: fullPrediction, timestamp: new Date() },
            ...prev,
          ].slice(0, 50))
        } finally {
          setLoading(false)
        }
      }
      reader.readAsDataURL(file)
    },
    [modelLoaded, setHistory]
  )

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.bmp'] },
    multiple: false,
    disabled: !modelLoaded || loading,
    noClick: true,
  })

  const isDefect = result && result.confidence > 0.7
  const hasDetections = boundingBoxes.length > 0

  return (
    <div className="flex-1 flex gap-5 p-5 overflow-hidden">
      {/* Left Panel - Upload */}
      <Card darkMode={darkMode} className="flex-1 flex flex-col">
        <CardHeader darkMode={darkMode} actions={
          <div className="flex items-center gap-2">
            {/* Toggle bounding boxes */}
            {hasDetections && (
              <button
                onClick={() => setShowBoxes(!showBoxes)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 text-sm font-medium rounded-lg transition-colors ${
                  showBoxes 
                    ? darkMode ? 'bg-purple-500/20 text-purple-400' : 'bg-purple-100 text-purple-600'
                    : darkMode ? 'bg-gray-700 text-gray-400' : 'bg-gray-200 text-gray-600'
                }`}
                title={showBoxes ? 'Masquer les zones' : 'Afficher les zones'}
              >
                {showBoxes ? <EyeIcon className="w-4 h-4" /> : <EyeSlashIcon className="w-4 h-4" />}
                Zones
              </button>
            )}
            <button
              onClick={open}
              disabled={!modelLoaded || loading}
              className="flex items-center gap-2 px-3 py-1.5 bg-lime-500 hover:bg-lime-600 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors"
            >
              <FolderOpenIcon className="w-4 h-4" />
              Parcourir
            </button>
          </div>
        }>
          <CloudArrowUpIcon className="w-5 h-5 text-lime-500" />
          Image PCB
        </CardHeader>

        <div {...getRootProps()} className="flex-1 p-4">
          <input {...getInputProps()} />
          <div
            className={`h-full rounded-lg border-2 border-dashed flex flex-col items-center justify-center transition-colors ${
              isDragActive
                ? 'border-lime-400 bg-lime-400/10'
                : darkMode
                  ? 'border-gray-700 hover:border-gray-600'
                  : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            {currentImage ? (
              <div className="w-full h-full flex flex-col items-center justify-center p-4">
                {/* Image avec bounding boxes */}
                <ImageWithBoxes
                  imageSrc={currentImage}
                  boundingBoxes={boundingBoxes}
                  showBoxes={showBoxes}
                  darkMode={darkMode}
                  className="max-h-[400px] w-full"
                />
                
                {loading && (
                  <div className="flex items-center justify-center gap-2 mt-4 text-lime-500">
                    <ArrowPathIcon className="w-5 h-5 animate-spin" />
                    <span className="font-medium">Analyse en cours...</span>
                  </div>
                )}
                
                {/* Indicateur de détections */}
                {!loading && hasDetections && (
                  <div className={`mt-3 text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    🎯 {boundingBoxes.length} zone{boundingBoxes.length > 1 ? 's' : ''} de défaut{boundingBoxes.length > 1 ? 's' : ''} détectée{boundingBoxes.length > 1 ? 's' : ''}
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center p-8">
                <CloudArrowUpIcon className={`w-16 h-16 mx-auto mb-4 ${darkMode ? 'text-gray-600' : 'text-gray-300'}`} />
                <p className={`text-lg mb-1 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  {isDragActive ? 'Déposez l\'image ici' : 'Glissez une image PCB'}
                </p>
                <p className={`text-sm ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                  Formats supportés: JPG, PNG, BMP
                </p>
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Right Panel - Results */}
      <Card darkMode={darkMode} className="w-80 flex flex-col">
        <CardHeader darkMode={darkMode}>
          <DocumentChartBarIcon className="w-5 h-5 text-purple-500" />
          Résultats
        </CardHeader>

        <CardContent className="flex-1 overflow-y-auto">
          {result ? (
            <div className="space-y-4">
              {/* Status */}
              <div className={`p-4 rounded-lg ${isDefect 
                ? darkMode ? 'bg-red-500/10 border border-red-500/30' : 'bg-red-50 border border-red-200'
                : darkMode ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-emerald-50 border border-emerald-200'
              }`}>
                <div className="flex items-center gap-3">
                  {isDefect ? (
                    <ExclamationTriangleIcon className="w-8 h-8 text-red-500" />
                  ) : (
                    <CheckCircleSolid className="w-8 h-8 text-emerald-500" />
                  )}
                  <div>
                    <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Statut</p>
                    <p className={`font-bold ${isDefect ? 'text-red-500' : 'text-emerald-500'}`}>
                      {isDefect ? 'Défaut détecté' : 'Conforme'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Détections count */}
              {hasDetections && (
                <div className={`p-3 rounded-lg ${darkMode ? 'bg-purple-500/10 border border-purple-500/30' : 'bg-purple-50 border border-purple-200'}`}>
                  <div className="flex items-center justify-between">
                    <span className={`text-sm ${darkMode ? 'text-purple-300' : 'text-purple-600'}`}>
                      Zones détectées
                    </span>
                    <span className={`text-lg font-bold ${darkMode ? 'text-purple-400' : 'text-purple-600'}`}>
                      {boundingBoxes.length}
                    </span>
                  </div>
                </div>
              )}

              {/* Defect Type */}
              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700/50' : 'bg-gray-50'}`}>
                <p className={`text-sm mb-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Type principal</p>
                <p className={`text-lg font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                  {DEFECT_LABELS[result.defectType]}
                </p>
                <p className={`text-2xl font-bold ${isDefect ? 'text-red-500' : 'text-emerald-500'}`}>
                  {(result.confidence * 100).toFixed(1)}%
                </p>
              </div>

              {/* Detected defects list */}
              {hasDetections && (
                <div>
                  <p className={`text-sm font-medium mb-2 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Défauts localisés
                  </p>
                  <div className="space-y-2">
                    {boundingBoxes.map((box, idx) => (
                      <div 
                        key={idx}
                        className={`flex items-center justify-between p-2 rounded ${
                          darkMode ? 'bg-gray-700/30' : 'bg-gray-100'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <span className={`w-3 h-3 rounded-sm ${DEFECT_COLORS[box.className]}`} />
                          <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                            {DEFECT_LABELS[box.className]}
                          </span>
                        </div>
                        <span className={`text-xs font-medium ${darkMode ? 'text-gray-500' : 'text-gray-400'}`}>
                          {(box.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Probabilities */}
              <div>
                <p className={`text-sm font-medium mb-3 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Probabilités globales
                </p>
                <div className="space-y-2.5">
                  {Object.entries(result.allScores)
                    .sort(([, a], [, b]) => b - a)
                    .map(([cls, score]) => (
                      <ProgressBar
                        key={cls}
                        label={DEFECT_LABELS[cls]}
                        value={score}
                        color={DEFECT_COLORS[cls]}
                        darkMode={darkMode}
                      />
                    ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <CpuChipIcon className={`w-12 h-12 mx-auto mb-2 ${darkMode ? 'text-gray-600' : 'text-gray-300'}`} />
                <p className={darkMode ? 'text-gray-500' : 'text-gray-400'}>En attente d'analyse</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
