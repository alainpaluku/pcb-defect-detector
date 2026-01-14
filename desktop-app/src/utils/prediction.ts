import * as ort from 'onnxruntime-web'
import { PredictionResult } from '../types'
import { DEFECT_CLASSES } from '../constants'

// Configuration
const MODEL_CONFIG = {
  inputSize: 224,
  inputChannels: 3,
} as const

// Session ONNX Runtime
let session: ort.InferenceSession | null = null

/**
 * Charge le modèle ONNX avec warmup
 */
export async function loadOnnxModel(modelPath = '/model/pcb_model.onnx'): Promise<boolean> {
  try {
    const options: ort.InferenceSession.SessionOptions = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }
    
    session = await ort.InferenceSession.create(modelPath, options)
    console.log('✅ Modèle ONNX chargé')
    
    // Warmup pour initialiser le modèle
    await warmupModel()
    
    return true
  } catch (error) {
    console.error('❌ Erreur chargement modèle:', error)
    return false
  }
}

/**
 * Warmup du modèle avec une inférence à vide
 */
async function warmupModel(): Promise<void> {
  if (!session) return
  
  const { inputSize, inputChannels } = MODEL_CONFIG
  const dummyData = new Float32Array(1 * inputSize * inputSize * inputChannels).fill(0)
  const tensor = new ort.Tensor('float32', dummyData, [1, inputSize, inputSize, inputChannels])
  
  const inputName = session.inputNames[0]
  await session.run({ [inputName]: tensor })
  
  console.log('✅ Warmup terminé')
}

/**
 * Vérifie si le modèle est chargé
 */
export function isModelReady(): boolean {
  return session !== null
}

/**
 * Prétraite une image pour l'inférence ONNX
 * Redimensionne à 224x224 et normalise [0, 1]
 */
async function preprocessImage(imageSrc: string): Promise<Float32Array> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    
    img.onload = () => {
      const { inputSize, inputChannels } = MODEL_CONFIG
      const canvas = document.createElement('canvas')
      canvas.width = inputSize
      canvas.height = inputSize
      
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        reject(new Error('Impossible de créer le contexte canvas'))
        return
      }
      
      ctx.drawImage(img, 0, 0, inputSize, inputSize)
      const { data } = ctx.getImageData(0, 0, inputSize, inputSize)
      
      // Format NHWC: [1, 224, 224, 3], normalisé [0, 1]
      const pixelCount = inputSize * inputSize
      const floatData = new Float32Array(pixelCount * inputChannels)
      
      for (let i = 0; i < pixelCount; i++) {
        const offset = i * 4 // RGBA
        floatData[i * 3] = data[offset] / 255.0       // R
        floatData[i * 3 + 1] = data[offset + 1] / 255.0 // G
        floatData[i * 3 + 2] = data[offset + 2] / 255.0 // B
      }
      
      resolve(floatData)
    }
    
    img.onerror = () => reject(new Error('Erreur chargement image'))
    img.src = imageSrc
  })
}

/**
 * Softmax pour convertir logits en probabilités
 */
function softmax(arr: Float32Array): number[] {
  const max = Math.max(...arr)
  const exp = Array.from(arr).map(x => Math.exp(x - max))
  const sum = exp.reduce((a, b) => a + b, 0)
  return exp.map(x => x / sum)
}

/**
 * Prédit le défaut sur une image avec le modèle ONNX
 */
export async function predictImage(imageSrc?: string): Promise<PredictionResult> {
  // Mode mock si pas de modèle ou pas d'image
  if (!session || !imageSrc) {
    return mockPrediction()
  }

  try {
    const { inputSize, inputChannels } = MODEL_CONFIG
    const inputData = await preprocessImage(imageSrc)
    const inputTensor = new ort.Tensor('float32', inputData, [1, inputSize, inputSize, inputChannels])
    
    const inputName = session.inputNames[0]
    const outputName = session.outputNames[0]
    
    const results = await session.run({ [inputName]: inputTensor })
    const output = results[outputName].data as Float32Array
    
    // Appliquer softmax si les valeurs sont des logits (hors [0,1])
    const probs = needsSoftmax(output) ? softmax(output) : Array.from(output)
    
    return buildPredictionResult(probs)
  } catch (error) {
    console.error('Erreur prédiction:', error)
    return mockPrediction()
  }
}

/**
 * Vérifie si softmax est nécessaire (détecte les logits)
 */
function needsSoftmax(values: Float32Array): boolean {
  return values[0] > 1 || values[0] < 0
}

/**
 * Construit le résultat de prédiction à partir des probabilités
 */
function buildPredictionResult(probs: number[]): PredictionResult {
  const allScores: Record<string, number> = {}
  let maxIdx = 0
  let maxProb = 0

  DEFECT_CLASSES.forEach((cls, idx) => {
    const prob = probs[idx]
    allScores[cls] = prob
    if (prob > maxProb) {
      maxProb = prob
      maxIdx = idx
    }
  })

  return {
    defectType: DEFECT_CLASSES[maxIdx],
    confidence: maxProb,
    allScores,
  }
}

/**
 * Prédiction mock pour le développement
 */
function mockPrediction(): Promise<PredictionResult> {
  return new Promise((resolve) => {
    setTimeout(() => {
      const scores: Record<string, number> = {}
      let total = 0
      
      DEFECT_CLASSES.forEach((cls) => {
        scores[cls] = Math.random()
        total += scores[cls]
      })
      
      // Normaliser
      Object.keys(scores).forEach((k) => (scores[k] /= total))
      
      const entries = Object.entries(scores).sort(([, a], [, b]) => b - a)
      
      resolve({
        defectType: entries[0][0],
        confidence: entries[0][1],
        allScores: scores,
      })
    }, 800)
  })
}

/**
 * Libère les ressources
 */
export async function disposeModel(): Promise<void> {
  if (session) {
    await session.release()
    session = null
  }
}

