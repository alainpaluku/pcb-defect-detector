import * as ort from 'onnxruntime-web'
import { BoundingBox, PredictionResult } from '../types'
import { DEFECT_CLASSES } from '../constants'

// Configuration du détecteur YOLO
const DETECTOR_CONFIG = {
  inputSize: 640,
  confThreshold: 0.25,
  iouThreshold: 0.45,
} as const

// Session ONNX pour le détecteur
let detectorSession: ort.InferenceSession | null = null

/**
 * Charge le modèle de détection YOLO (ONNX)
 */
export async function loadDetectorModel(modelPath = '/model/pcb_detector.onnx'): Promise<boolean> {
  try {
    const options: ort.InferenceSession.SessionOptions = {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }
    
    detectorSession = await ort.InferenceSession.create(modelPath, options)
    console.log('✅ Modèle de détection YOLO chargé')
    
    return true
  } catch (error) {
    console.error('❌ Erreur chargement détecteur:', error)
    return false
  }
}

/**
 * Vérifie si le détecteur est chargé
 */
export function isDetectorReady(): boolean {
  return detectorSession !== null
}

/**
 * Prétraite une image pour YOLO (640x640, normalisé)
 */
async function preprocessForYolo(imageSrc: string): Promise<{
  tensor: Float32Array
  originalWidth: number
  originalHeight: number
  scale: number
}> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    
    img.onload = () => {
      const { inputSize } = DETECTOR_CONFIG
      const canvas = document.createElement('canvas')
      canvas.width = inputSize
      canvas.height = inputSize
      
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        reject(new Error('Impossible de créer le contexte canvas'))
        return
      }
      
      // Calculer le scale pour garder le ratio
      const scale = Math.min(inputSize / img.width, inputSize / img.height)
      const scaledWidth = img.width * scale
      const scaledHeight = img.height * scale
      
      // Centrer l'image avec padding
      const offsetX = (inputSize - scaledWidth) / 2
      const offsetY = (inputSize - scaledHeight) / 2
      
      // Fond gris (letterbox)
      ctx.fillStyle = '#808080'
      ctx.fillRect(0, 0, inputSize, inputSize)
      
      // Dessiner l'image centrée
      ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight)
      
      const { data } = ctx.getImageData(0, 0, inputSize, inputSize)
      
      // Format NCHW pour YOLO: [1, 3, 640, 640], normalisé [0, 1]
      const floatData = new Float32Array(1 * 3 * inputSize * inputSize)
      
      for (let i = 0; i < inputSize * inputSize; i++) {
        const offset = i * 4
        // YOLO attend RGB en format CHW (channels first)
        floatData[i] = data[offset] / 255.0                           // R channel
        floatData[inputSize * inputSize + i] = data[offset + 1] / 255.0     // G channel
        floatData[2 * inputSize * inputSize + i] = data[offset + 2] / 255.0 // B channel
      }
      
      resolve({
        tensor: floatData,
        originalWidth: img.width,
        originalHeight: img.height,
        scale
      })
    }
    
    img.onerror = () => reject(new Error('Erreur chargement image'))
    img.src = imageSrc
  })
}

/**
 * Applique Non-Maximum Suppression
 */
function nms(boxes: BoundingBox[], iouThreshold: number): BoundingBox[] {
  if (boxes.length === 0) return []
  
  // Trier par confiance décroissante
  const sorted = [...boxes].sort((a, b) => b.confidence - a.confidence)
  const selected: BoundingBox[] = []
  const used = new Set<number>()
  
  for (let i = 0; i < sorted.length; i++) {
    if (used.has(i)) continue
    
    selected.push(sorted[i])
    
    for (let j = i + 1; j < sorted.length; j++) {
      if (used.has(j)) continue
      
      if (calculateIoU(sorted[i], sorted[j]) > iouThreshold) {
        used.add(j)
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
  const area1 = box1.width * box1.height
  const area2 = box2.width * box2.height
  const union = area1 + area2 - intersection
  
  return intersection / union
}

/**
 * Détecte les défauts et retourne les bounding boxes
 */
export async function detectDefects(imageSrc: string): Promise<BoundingBox[]> {
  if (!detectorSession) {
    console.warn('Détecteur non chargé, utilisation du mode mock')
    return mockDetection(imageSrc)
  }
  
  try {
    const { inputSize, confThreshold, iouThreshold } = DETECTOR_CONFIG
    const { tensor, originalWidth, originalHeight, scale } = await preprocessForYolo(imageSrc)
    
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, inputSize, inputSize])
    
    const inputName = detectorSession.inputNames[0]
    const results = await detectorSession.run({ [inputName]: inputTensor })
    
    // Parser la sortie YOLO (format: [1, num_boxes, 4 + num_classes])
    const output = results[detectorSession.outputNames[0]]
    const outputData = output.data as Float32Array
    const outputShape = output.dims as number[]
    
    const boxes = parseYoloOutput(
      outputData, 
      outputShape, 
      originalWidth, 
      originalHeight, 
      scale,
      confThreshold
    )
    
    // Appliquer NMS
    return nms(boxes, iouThreshold)
    
  } catch (error) {
    console.error('Erreur détection:', error)
    return mockDetection(imageSrc)
  }
}

/**
 * Parse la sortie du modèle YOLO
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
  
  // Format YOLOv8: [1, 4 + num_classes, num_boxes] transposé
  const numClasses = DEFECT_CLASSES.length
  const numBoxes = shape[2] || shape[1]
  
  // Offset pour le letterbox
  const offsetX = (inputSize - originalWidth * scale) / 2
  const offsetY = (inputSize - originalHeight * scale) / 2
  
  for (let i = 0; i < numBoxes; i++) {
    // Extraire les coordonnées (format xywh)
    const cx = data[i]
    const cy = data[numBoxes + i]
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
    
    // Convertir de xywh à xyxy et ajuster pour l'image originale
    const x1 = ((cx - w / 2) - offsetX) / scale
    const y1 = ((cy - h / 2) - offsetY) / scale
    const x2 = ((cx + w / 2) - offsetX) / scale
    const y2 = ((cy + h / 2) - offsetY) / scale
    
    // Vérifier que la box est dans l'image
    if (x1 >= 0 && y1 >= 0 && x2 <= originalWidth && y2 <= originalHeight) {
      boxes.push({
        x1: Math.max(0, x1),
        y1: Math.max(0, y1),
        x2: Math.min(originalWidth, x2),
        y2: Math.min(originalHeight, y2),
        width: x2 - x1,
        height: y2 - y1,
        className: DEFECT_CLASSES[maxClassId],
        confidence: maxConf
      })
    }
  }
  
  return boxes
}

/**
 * Détection mock pour le développement
 */
function mockDetection(imageSrc: string): Promise<BoundingBox[]> {
  return new Promise((resolve) => {
    // Charger l'image pour obtenir ses dimensions
    const img = new Image()
    img.onload = () => {
      const numDefects = Math.floor(Math.random() * 3) + 1
      const boxes: BoundingBox[] = []
      
      for (let i = 0; i < numDefects; i++) {
        const width = 30 + Math.random() * 60
        const height = 30 + Math.random() * 60
        const x1 = Math.random() * (img.width - width)
        const y1 = Math.random() * (img.height - height)
        
        boxes.push({
          x1,
          y1,
          x2: x1 + width,
          y2: y1 + height,
          width,
          height,
          className: DEFECT_CLASSES[Math.floor(Math.random() * DEFECT_CLASSES.length)],
          confidence: 0.7 + Math.random() * 0.25
        })
      }
      
      setTimeout(() => resolve(boxes), 500)
    }
    img.src = imageSrc
  })
}

/**
 * Libère les ressources du détecteur
 */
export async function disposeDetector(): Promise<void> {
  if (detectorSession) {
    await detectorSession.release()
    detectorSession = null
  }
}
