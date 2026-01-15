/**
 * Point d'entrée des utilitaires
 */

// ONNX commun
export {
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

// Prédiction/Classification
export {
  loadOnnxModel,
  isModelReady,
  predictImage,
  disposeModel,
} from './prediction'

// Détection YOLO
export {
  loadDetectorModel,
  isDetectorReady,
  detectDefects,
  disposeDetector,
} from './detection'
