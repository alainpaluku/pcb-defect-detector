/**
 * Configuration centralisée pour le modèle ONNX
 */

export const MODEL_CONFIG = {
  // Chemins
  modelPath: '/model/pcb_model.onnx',
  
  // Dimensions d'entrée
  inputSize: 224,
  inputChannels: 3,
  inputName: 'input_image',
  
  // Seuils de confiance
  confidenceThreshold: 0.7,
  
  // Options ONNX Runtime
  executionProviders: ['wasm'] as const,
  graphOptimizationLevel: 'all' as const,
} as const

export const DEFECT_CONFIG = {
  // Classes de défauts
  classes: [
    'missing_hole',
    'mouse_bite',
    'open_circuit',
    'short',
    'spur',
    'spurious_copper',
  ] as const,
  
  // Labels en français
  labels: {
    missing_hole: 'Trou manquant',
    mouse_bite: 'Bord rongé',
    open_circuit: 'Circuit ouvert',
    short: 'Court-circuit',
    spur: 'Excroissance',
    spurious_copper: 'Cuivre parasite',
  } as const,
  
  // Couleurs pour la visualisation
  colors: {
    missing_hole: 'bg-red-500',
    mouse_bite: 'bg-orange-500',
    open_circuit: 'bg-yellow-500',
    short: 'bg-purple-500',
    spur: 'bg-blue-500',
    spurious_copper: 'bg-pink-500',
  } as const,
} as const

export type DefectClass = typeof DEFECT_CONFIG.classes[number]
