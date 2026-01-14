export interface PredictionResult {
  defectType: string
  confidence: number
  allScores: Record<string, number>
  boundingBoxes?: BoundingBox[]  // Zones de défauts détectées
}

export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
  width: number
  height: number
  className: string
  confidence: number
}

export interface InspectionItem {
  id: string
  imageSrc: string
  fileName: string
  prediction: PredictionResult
  timestamp: Date
}

export type Page = 'home' | 'history' | 'settings' | 'about'

export interface AppSettings {
  confidenceThreshold: number
  historyLimit: number
  autoSave: boolean
  showNotifications: boolean
  showBoundingBoxes: boolean  // Afficher les encadrements
}
