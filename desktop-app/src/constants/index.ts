export const DEFECT_LABELS: Record<string, string> = {
  missing_hole: 'Trou manquant',
  mouse_bite: 'Bord rongé',
  open_circuit: 'Circuit ouvert',
  short: 'Court-circuit',
  spur: 'Excroissance',
  spurious_copper: 'Cuivre parasite',
}

export const DEFECT_COLORS: Record<string, string> = {
  missing_hole: 'bg-red-500',
  mouse_bite: 'bg-orange-500',
  open_circuit: 'bg-yellow-500',
  short: 'bg-purple-500',
  spur: 'bg-blue-500',
  spurious_copper: 'bg-pink-500',
}

export const DEFECT_CLASSES = [
  'missing_hole',
  'mouse_bite',
  'open_circuit',
  'short',
  'spur',
  'spurious_copper',
] as const

export const DEFAULT_SETTINGS = {
  confidenceThreshold: 70,
  historyLimit: 50,
  autoSave: true,
  showNotifications: true,
}
