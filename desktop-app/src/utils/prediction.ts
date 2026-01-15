/**
 * Module de prédiction/classification pour PCB
 */
import * as ort from 'onnxruntime-web'
import { PredictionResult } from '../types'
import { DEFECT_CLASSES } from '../constants'
import {
  OnnxSessionManager,
  loadImage,
  createCanvas,
  extractRgbData,
  softmax,
  needsSoftmax,
  generateMockScores,
  findMaxClass,
  delay,
} from './onnx-common'

// Configuration
const MODEL_CONFIG = {
  inputSize: 224,
  inputChannels: 3,
} as const

// Gestionnaire de session
const sessionManager = new OnnxSessionManager('Modèle classification')

/**
 * Charge le modèle ONNX avec warmup
 */
export async function loadOnnxModel(modelPath = '/model/pcb_model.onnx'): Promise<boolean> {
  const success = await sessionManager.load(modelPath)
  if (success) await warmupModel()
  return success
}

/**
 * Warmup du modèle avec une inférence à vide
 */
async function warmupModel(): Promise<void> {
  if (!sessionManager.isReady()) return

  const { inputSize, inputChannels } = MODEL_CONFIG
  const dummyData = new Float32Array(1 * inputSize * inputSize * inputChannels).fill(0)
  const tensor = new ort.Tensor('float32', dummyData, [1, inputSize, inputSize, inputChannels])

  await sessionManager.run({ [sessionManager.inputNames[0]]: tensor })
  console.log('✅ Warmup terminé')
}

/**
 * Vérifie si le modèle est chargé
 */
export function isModelReady(): boolean {
  return sessionManager.isReady()
}

/**
 * Prétraite une image pour l'inférence (224x224, NHWC)
 */
async function preprocessImage(imageSrc: string): Promise<Float32Array> {
  const img = await loadImage(imageSrc)
  const { inputSize } = MODEL_CONFIG
  const { ctx } = createCanvas(inputSize, inputSize)

  ctx.drawImage(img, 0, 0, inputSize, inputSize)
  const imageData = ctx.getImageData(0, 0, inputSize, inputSize)

  return extractRgbData(imageData, 'NHWC')
}

/**
 * Prédit le défaut sur une image avec le modèle ONNX
 */
export async function predictImage(imageSrc?: string): Promise<PredictionResult> {
  if (!sessionManager.isReady() || !imageSrc) {
    return mockPrediction()
  }

  try {
    const { inputSize, inputChannels } = MODEL_CONFIG
    const inputData = await preprocessImage(imageSrc)
    const inputTensor = new ort.Tensor('float32', inputData, [1, inputSize, inputSize, inputChannels])

    const results = await sessionManager.run({ [sessionManager.inputNames[0]]: inputTensor })
    const output = results[sessionManager.outputNames[0]].data as Float32Array

    const probs = needsSoftmax(output) ? softmax(output) : Array.from(output)
    return buildPredictionResult(probs)
  } catch (error) {
    console.error('Erreur prédiction:', error)
    return mockPrediction()
  }
}

/**
 * Construit le résultat de prédiction à partir des probabilités
 */
function buildPredictionResult(probs: number[]): PredictionResult {
  const allScores: Record<string, number> = {}

  DEFECT_CLASSES.forEach((cls, idx) => {
    allScores[cls] = probs[idx] ?? 0
  })

  const { className, confidence } = findMaxClass(allScores)

  return {
    defectType: className,
    confidence,
    allScores,
  }
}

/**
 * Prédiction mock pour le développement
 */
async function mockPrediction(): Promise<PredictionResult> {
  await delay(800)

  const allScores = generateMockScores()
  const { className, confidence } = findMaxClass(allScores)

  return {
    defectType: className,
    confidence,
    allScores,
  }
}

/**
 * Libère les ressources
 */
export async function disposeModel(): Promise<void> {
  await sessionManager.dispose()
}
