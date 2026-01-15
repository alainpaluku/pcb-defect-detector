/**
 * Module de détection d'objets YOLO pour PCB
 */
import * as ort from 'onnxruntime-web'
import { BoundingBox } from '../types'
import { DEFECT_CLASSES } from '../constants'
import {
  OnnxSessionManager,
  loadImage,
  createCanvas,
  extractRgbData,
  delay,
} from './onnx-common'

// Configuration du détecteur YOLO
const DETECTOR_CONFIG = {
  inputSize: 640,
  confThreshold: 0.25,
  iouThreshold: 0.45,
} as const

// Gestionnaire de session
const sessionManager = new OnnxSessionManager('Détecteur YOLO')

/**
 * Charge le modèle de détection YOLO (ONNX)
 */
export async function loadDetectorModel(modelPath = '/model/pcb_detector.onnx'): Promise<boolean> {
  return sessionManager.load(modelPath)
}

/**
 * Vérifie si le détecteur est chargé
 */
export function isDetectorReady(): boolean {
  return sessionManager.isReady()
}

/**
 * Prétraite une image pour YOLO (640x640, letterbox, NCHW)
 */
async function preprocessForYolo(imageSrc: string): Promise<{
  tensor: Float32Array
  originalWidth: number
  originalHeight: number
  scale: number
}> {
  const img = await loadImage(imageSrc)
  const { inputSize } = DETECTOR_CONFIG
  const { ctx } = createCanvas(inputSize, inputSize)

  // Calculer le scale pour garder le ratio
  const scale = Math.min(inputSize / img.width, inputSize / img.height)
  const scaledWidth = img.width * scale
  const scaledHeight = img.height * scale

  // Centrer l'image avec padding (letterbox)
  const offsetX = (inputSize - scaledWidth) / 2
  const offsetY = (inputSize - scaledHeight) / 2

  ctx.fillStyle = '#808080'
  ctx.fillRect(0, 0, inputSize, inputSize)
  ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight)

  const imageData = ctx.getImageData(0, 0, inputSize, inputSize)
  const tensor = extractRgbData(imageData, 'NCHW')

  return {
    tensor,
    originalWidth: img.width,
    originalHeight: img.height,
    scale,
  }
}

/**
 * Détecte les défauts et retourne les bounding boxes
 */
export async function detectDefects(imageSrc: string): Promise<BoundingBox[]> {
  if (!sessionManager.isReady()) {
    console.warn('Détecteur non chargé, utilisation du mode mock')
    return mockDetection(imageSrc)
  }

  try {
    const { inputSize, confThreshold, iouThreshold } = DETECTOR_CONFIG
    const { tensor, originalWidth, originalHeight, scale } = await preprocessForYolo(imageSrc)

    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, inputSize, inputSize])
    const results = await sessionManager.run({ [sessionManager.inputNames[0]]: inputTensor })

    const output = results[sessionManager.outputNames[0]]
    const outputData = output.data as Float32Array
    const outputShape = output.dims as number[]

    const boxes = parseYoloOutput(outputData, outputShape, originalWidth, originalHeight, scale, confThreshold)
    return applyNms(boxes, iouThreshold)
  } catch (error) {
    console.error('Erreur détection:', error)
    return mockDetection(imageSrc)
  }
}

/**
 * Parse la sortie du modèle YOLO
 * YOLOv8 ONNX output format: [1, 4 + num_classes, num_boxes]
 */
function parseYoloOutput(
  data: Float32Array,
  shape: number[],
  originalWidth: number,
  originalHeight: number,
  scale: number,
  confThreshold: number
): BoundingBox[] {
  const boxes: BoundingBox[] = []
  const { inputSize } = DETECTOR_CONFIG
  const numClasses = DEFECT_CLASSES.length
  const numBoxes = shape[2]

  const offsetX = (inputSize - originalWidth * scale) / 2
  const offsetY = (inputSize - originalHeight * scale) / 2

  for (let i = 0; i < numBoxes; i++) {
    const cx = data[0 * numBoxes + i]
    const cy = data[1 * numBoxes + i]
    const w = data[2 * numBoxes + i]
    const h = data[3 * numBoxes + i]

    // Trouver la classe avec la plus haute confiance
    let maxConf = 0
    let maxClassId = 0
    for (let c = 0; c < numClasses; c++) {
      const conf = data[(4 + c) * numBoxes + i]
      if (conf > maxConf) {
        maxConf = conf
        maxClassId = c
      }
    }

    if (maxConf < confThreshold) continue

    // Convertir de xywh centré à xyxy
    const box = convertToOriginalCoords(cx, cy, w, h, offsetX, offsetY, scale, originalWidth, originalHeight)
    if (box) {
      boxes.push({
        ...box,
        className: DEFECT_CLASSES[maxClassId],
        confidence: maxConf,
      })
    }
  }

  return boxes
}

/**
 * Convertit les coordonnées YOLO vers les coordonnées de l'image originale
 */
function convertToOriginalCoords(
  cx: number, cy: number, w: number, h: number,
  offsetX: number, offsetY: number, scale: number,
  maxWidth: number, maxHeight: number
): Omit<BoundingBox, 'className' | 'confidence'> | null {
  const x1 = Math.max(0, ((cx - w / 2) - offsetX) / scale)
  const y1 = Math.max(0, ((cy - h / 2) - offsetY) / scale)
  const x2 = Math.min(maxWidth, ((cx + w / 2) - offsetX) / scale)
  const y2 = Math.min(maxHeight, ((cy + h / 2) - offsetY) / scale)

  if (x2 <= x1 || y2 <= y1) return null

  return { x1, y1, x2, y2, width: x2 - x1, height: y2 - y1 }
}

/**
 * Applique Non-Maximum Suppression
 */
function applyNms(boxes: BoundingBox[], iouThreshold: number): BoundingBox[] {
  if (boxes.length === 0) return []

  const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence)
  const selected: BoundingBox[] = []
  const suppressed = new Set<number>()

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed.has(i)) continue
    selected.push(sorted[i])

    for (let j = i + 1; j < sorted.length; j++) {
      if (!suppressed.has(j) && calculateIoU(sorted[i], sorted[j]) > iouThreshold) {
        suppressed.add(j)
      }
    }
  }

  return selected
}

/**
 * Calcule l'Intersection over Union entre deux boxes
 */
function calculateIoU(box1: BoundingBox, box2: BoundingBox): number {
  const x1 = Math.max(box1.x1, box2.x1)
  const y1 = Math.max(box1.y1, box2.y1)
  const x2 = Math.min(box1.x2, box2.x2)
  const y2 = Math.min(box1.y2, box2.y2)

  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  const union = box1.width * box1.height + box2.width * box2.height - intersection

  return union > 0 ? intersection / union : 0
}

/**
 * Détection mock pour le développement
 */
async function mockDetection(imageSrc: string): Promise<BoundingBox[]> {
  const img = await loadImage(imageSrc)
  await delay(500)

  const numDefects = Math.floor(Math.random() * 3) + 1
  const boxes: BoundingBox[] = []

  for (let i = 0; i < numDefects; i++) {
    const width = 30 + Math.random() * 60
    const height = 30 + Math.random() * 60
    const x1 = Math.random() * (img.width - width)
    const y1 = Math.random() * (img.height - height)

    boxes.push({
      x1, y1,
      x2: x1 + width,
      y2: y1 + height,
      width, height,
      className: DEFECT_CLASSES[Math.floor(Math.random() * DEFECT_CLASSES.length)],
      confidence: 0.7 + Math.random() * 0.25,
    })
  }

  return boxes
}

/**
 * Libère les ressources du détecteur
 */
export async function disposeDetector(): Promise<void> {
  await sessionManager.dispose()
}
