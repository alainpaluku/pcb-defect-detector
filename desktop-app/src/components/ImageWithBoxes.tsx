import { useRef, useEffect, useState } from 'react'
import { BoundingBox } from '../types'
import { DEFECT_COLORS, DEFECT_LABELS } from '../constants'

interface ImageWithBoxesProps {
  imageSrc: string
  boundingBoxes?: BoundingBox[]
  showBoxes?: boolean
  darkMode?: boolean
  className?: string
}

// Couleurs pour les bounding boxes (format RGB pour canvas)
const BOX_COLORS: Record<string, string> = {
  missing_hole: '#ef4444',     // red-500
  mouse_bite: '#f97316',       // orange-500
  open_circuit: '#eab308',     // yellow-500
  short: '#a855f7',            // purple-500
  spur: '#3b82f6',             // blue-500
  spurious_copper: '#ec4899',  // pink-500
}

/**
 * Composant qui affiche une image avec des bounding boxes
 * pour encercler les zones de défauts détectées
 */
export function ImageWithBoxes({
  imageSrc,
  boundingBoxes = [],
  showBoxes = true,
  darkMode = true,
  className = ''
}: ImageWithBoxesProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [imageLoaded, setImageLoaded] = useState(false)
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 })

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.crossOrigin = 'anonymous'

    img.onload = () => {
      // Calculer les dimensions pour s'adapter au conteneur
      const containerWidth = container.clientWidth
      const containerHeight = container.clientHeight || 400
      
      const scale = Math.min(
        containerWidth / img.width,
        containerHeight / img.height,
        1 // Ne pas agrandir au-delà de la taille originale
      )
      
      const displayWidth = img.width * scale
      const displayHeight = img.height * scale
      
      // Configurer le canvas
      canvas.width = displayWidth
      canvas.height = displayHeight
      
      setImageDimensions({ width: displayWidth, height: displayHeight })
      
      // Dessiner l'image
      ctx.drawImage(img, 0, 0, displayWidth, displayHeight)
      
      // Dessiner les bounding boxes si activé
      if (showBoxes && boundingBoxes.length > 0) {
        drawBoundingBoxes(ctx, boundingBoxes, img.width, img.height, scale)
      }
      
      setImageLoaded(true)
    }

    img.onerror = () => {
      console.error('Erreur de chargement de l\'image')
      setImageLoaded(false)
    }

    img.src = imageSrc
  }, [imageSrc, boundingBoxes, showBoxes])

  return (
    <div 
      ref={containerRef} 
      className={`relative flex items-center justify-center ${className}`}
    >
      <canvas
        ref={canvasRef}
        className="rounded-lg shadow-lg max-w-full"
        style={{ 
          maxHeight: '100%',
          objectFit: 'contain'
        }}
      />
      
      {/* Légende des défauts détectés */}
      {showBoxes && boundingBoxes.length > 0 && (
        <div className={`absolute bottom-2 left-2 p-2 rounded-lg text-xs ${
          darkMode ? 'bg-gray-900/80' : 'bg-white/80'
        }`}>
          <div className={`font-medium mb-1 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
            {boundingBoxes.length} défaut{boundingBoxes.length > 1 ? 's' : ''} détecté{boundingBoxes.length > 1 ? 's' : ''}
          </div>
          {getUniqueDefects(boundingBoxes).map(({ className: cls, count }) => (
            <div key={cls} className="flex items-center gap-1.5">
              <span 
                className="w-3 h-3 rounded-sm" 
                style={{ backgroundColor: BOX_COLORS[cls] || '#888' }}
              />
              <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>
                {DEFECT_LABELS[cls] || cls} ({count})
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * Dessine les bounding boxes sur le canvas
 */
function drawBoundingBoxes(
  ctx: CanvasRenderingContext2D,
  boxes: BoundingBox[],
  originalWidth: number,
  originalHeight: number,
  scale: number
) {
  boxes.forEach((box) => {
    const color = BOX_COLORS[box.className] || '#888888'
    
    // Convertir les coordonnées à l'échelle d'affichage
    const x1 = box.x1 * scale
    const y1 = box.y1 * scale
    const width = box.width * scale
    const height = box.height * scale
    
    // Dessiner le rectangle
    ctx.strokeStyle = color
    ctx.lineWidth = 3
    ctx.strokeRect(x1, y1, width, height)
    
    // Fond semi-transparent pour le label
    const label = `${DEFECT_LABELS[box.className] || box.className} ${(box.confidence * 100).toFixed(0)}%`
    ctx.font = 'bold 12px Inter, sans-serif'
    const textMetrics = ctx.measureText(label)
    const textHeight = 16
    const padding = 4
    
    // Position du label (au-dessus de la box si possible)
    let labelY = y1 - textHeight - padding
    if (labelY < 0) labelY = y1 + padding
    
    // Fond du label
    ctx.fillStyle = color
    ctx.fillRect(
      x1, 
      labelY, 
      textMetrics.width + padding * 2, 
      textHeight + padding
    )
    
    // Texte du label
    ctx.fillStyle = '#ffffff'
    ctx.fillText(label, x1 + padding, labelY + textHeight)
    
    // Cercle au centre pour les petits défauts
    if (width < 50 || height < 50) {
      const centerX = x1 + width / 2
      const centerY = y1 + height / 2
      const radius = Math.max(width, height) / 2 + 5
      
      ctx.beginPath()
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI)
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.stroke()
      ctx.setLineDash([])
    }
  })
}

/**
 * Compte les défauts uniques par type
 */
function getUniqueDefects(boxes: BoundingBox[]): Array<{ className: string; count: number }> {
  const counts: Record<string, number> = {}
  
  boxes.forEach(box => {
    counts[box.className] = (counts[box.className] || 0) + 1
  })
  
  return Object.entries(counts)
    .map(([className, count]) => ({ className, count }))
    .sort((a, b) => b.count - a.count)
}

export default ImageWithBoxes
