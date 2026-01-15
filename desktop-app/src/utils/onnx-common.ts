/**
 * Module commun pour les opérations ONNX Runtime
 * Factorise le code partagé entre detection.ts et prediction.ts
 */
import * as ort from 'onnxruntime-web'
import { DEFECT_CLASSES } from '../constants'

// Options ONNX par défaut
const DEFAULT_SESSION_OPTIONS: ort.InferenceSession.SessionOptions = {
  executionProviders: ['wasm'],
  graphOptimizationLevel: 'all',
}

/**
 * Gestionnaire de session ONNX réutilisable
 */
export class OnnxSessionManager {
  private session: ort.InferenceSession | null = null
  private modelName: string

  constructor(modelName: string) {
    this.modelName = modelName
  }

  async load(modelPath: string): Promise<boolean> {
    try {
      this.session = await ort.InferenceSession.create(modelPath, DEFAULT_SESSION_OPTIONS)
      console.log(`✅ ${this.modelName} chargé`)
      return true
    } catch (error) {
      console.error(`❌ Erreur chargement ${this.modelName}:`, error)
      return false
    }
  }

  isReady(): boolean {
    return this.session !== null
  }

  getSession(): ort.InferenceSession | null {
    return this.session
  }

  async run(inputs: Record<string, ort.Tensor>): Promise<ort.InferenceSession.OnnxValueMapType> {
    if (!this.session) throw new Error(`${this.modelName} non chargé`)
    return this.session.run(inputs)
  }

  get inputNames(): readonly string[] {
    return this.session?.inputNames ?? []
  }

  get outputNames(): readonly string[] {
    return this.session?.outputNames ?? []
  }

  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release()
      this.session = null
    }
  }
}

/**
 * Charge une image et retourne ses données pixel
 */
export function loadImage(imageSrc: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error('Erreur chargement image'))
    img.src = imageSrc
  })
}

/**
 * Crée un canvas avec contexte 2D
 */
export function createCanvas(width: number, height: number): { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D } {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  if (!ctx) throw new Error('Impossible de créer le contexte canvas')
  return { canvas, ctx }
}

/**
 * Extrait les données RGB normalisées d'un ImageData
 * @param format 'NHWC' pour [batch, height, width, channels] ou 'NCHW' pour [batch, channels, height, width]
 */
export function extractRgbData(
  imageData: ImageData,
  format: 'NHWC' | 'NCHW' = 'NHWC'
): Float32Array {
  const { data, width, height } = imageData
  const pixelCount = width * height
  const floatData = new Float32Array(pixelCount * 3)

  if (format === 'NHWC') {
    for (let i = 0; i < pixelCount; i++) {
      const offset = i * 4
      floatData[i * 3] = data[offset] / 255.0
      floatData[i * 3 + 1] = data[offset + 1] / 255.0
      floatData[i * 3 + 2] = data[offset + 2] / 255.0
    }
  } else {
    // NCHW: channels first
    for (let i = 0; i < pixelCount; i++) {
      const offset = i * 4
      floatData[i] = data[offset] / 255.0
      floatData[pixelCount + i] = data[offset + 1] / 255.0
      floatData[2 * pixelCount + i] = data[offset + 2] / 255.0
    }
  }

  return floatData
}

/**
 * Softmax pour convertir logits en probabilités
 */
export function softmax(arr: Float32Array | number[]): number[] {
  const values = Array.from(arr)
  const max = Math.max(...values)
  const exp = values.map(x => Math.exp(x - max))
  const sum = exp.reduce((a, b) => a + b, 0)
  return exp.map(x => x / sum)
}

/**
 * Vérifie si softmax est nécessaire (détecte les logits)
 */
export function needsSoftmax(values: Float32Array | number[]): boolean {
  const arr = Array.from(values)
  
  // Vérifier si une valeur est hors de [0, 1]
  if (arr.some(v => v < 0 || v > 1)) return true
  
  // Vérifier si la somme est proche de 1
  const sum = arr.reduce((a, b) => a + b, 0)
  return Math.abs(sum - 1.0) > 0.1
}

/**
 * Génère des scores aléatoires normalisés pour le mode mock
 */
export function generateMockScores(): Record<string, number> {
  const scores: Record<string, number> = {}
  let total = 0

  DEFECT_CLASSES.forEach(cls => {
    scores[cls] = Math.random()
    total += scores[cls]
  })

  // Normaliser
  Object.keys(scores).forEach(k => (scores[k] /= total))
  return scores
}

/**
 * Trouve la classe avec le score maximum
 */
export function findMaxClass(scores: Record<string, number>): { className: string; confidence: number } {
  const entries = Object.entries(scores)
  const [className, confidence] = entries.reduce((max, curr) => 
    curr[1] > max[1] ? curr : max
  )
  return { className, confidence }
}

/**
 * Délai pour simulation
 */
export function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}
