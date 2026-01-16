import { useRef, useEffect, useState, useCallback } from 'react'
import { BoundingBox } from '../types'
import { DEFECT_COLORS, DEFECT_LABELS } from '../constants'
import { MagnifyingGlassPlusIcon, MagnifyingGlassMinusIcon, ArrowsPointingOutIcon } from '@heroicons/react/24/outline'

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

const MIN_ZOOM = 0.5
const MAX_ZOOM = 5
const ZOOM_STEP = 0.25

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
  const [zoom, setZoom] = useState(1)
  const [pan, setPan] = useState({ x: 0, y: 0 })
  const [isPanning, setIsPanning] = useState(false)
  const [lastPanPoint, setLastPanPoint] = useState({ x: 0, y: 0 })
  const imageRef = useRef<HTMLImageElement | null>(null)
  const baseScaleRef = useRef(1)

  // Redessiner quand zoom/pan change
  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    const img = imageRef.current
    if (!canvas || !container || !img) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Effacer le canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // Sauvegarder le contexte
    ctx.save()
    
    // Appliquer le zoom et le pan
    ctx.translate(pan.x, pan.y)
    ctx.scale(zoom, zoom)
    
    // Dessiner l'image
    const displayWidth = img.width * baseScaleRef.current
    const displayHeight = img.height * baseScaleRef.current
    ctx.drawImage(img, 0, 0, displayWidth, displayHeight)
    
    // Dessiner les bounding boxes si activé
    if (showBoxes && boundingBoxes.length > 0) {
      drawBoundingBoxes(ctx, boundingBoxes, img.width, img.height, baseScaleRef.current)
    }
    
    // Restaurer le contexte
    ctx.restore()
  }, [zoom, pan, showBoxes, boundingBoxes])

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.crossOrigin = 'anonymous'

    img.onload = () => {
      imageRef.current = img
      
      // Calculer les dimensions pour s'adapter au conteneur
      const containerWidth = container.clientWidth
      const containerHeight = container.clientHeight || 400
      
      const scale = Math.min(
        containerWidth / img.width,
        containerHeight / img.height,
        1 // Ne pas agrandir au-delà de la taille originale
      )
      
      baseScaleRef.current = scale
      const displayWidth = img.width * scale
      const displayHeight = img.height * scale
      
      // Configurer le canvas
      canvas.width = containerWidth
      canvas.height = containerHeight
      
      setImageDimensions({ width: displayWidth, height: displayHeight })
      
      // Reset zoom et pan pour nouvelle image
      setZoom(1)
      setPan({ x: (containerWidth - displayWidth) / 2, y: (containerHeight - displayHeight) / 2 })
      
      setImageLoaded(true)
    }

    img.onerror = () => {
      console.error('Erreur de chargement de l\'image')
      setImageLoaded(false)
    }

    img.src = imageSrc
  }, [imageSrc])

  // Redessiner quand les paramètres changent
  useEffect(() => {
    if (imageLoaded) {
      redraw()
    }
  }, [imageLoaded, redraw, boundingBoxes, showBoxes])

  // Gestion du zoom avec la molette
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP
    setZoom(prev => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prev + delta)))
  }, [])

  // Gestion du pan (drag)
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (zoom > 1) {
      setIsPanning(true)
      setLastPanPoint({ x: e.clientX, y: e.clientY })
    }
  }, [zoom])

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    if (isPanning) {
      const dx = e.clientX - lastPanPoint.x
      const dy = e.clientY - lastPanPoint.y
      setPan(prev => ({ x: prev.x + dx, y: prev.y + dy }))
      setLastPanPoint({ x: e.clientX, y: e.clientY })
    }
  }, [isPanning, lastPanPoint])

  const handleMouseUp = useCallback(() => {
    setIsPanning(false)
  }, [])

  // Boutons de zoom
  const zoomIn = () => setZoom(prev => Math.min(MAX_ZOOM, prev + ZOOM_STEP))
  const zoomOut = () => setZoom(prev => Math.max(MIN_ZOOM, prev - ZOOM_STEP))
  const resetZoom = () => {
    setZoom(1)
    const container = containerRef.current
    if (container) {
      setPan({ 
        x: (container.clientWidth - imageDimensions.width) / 2, 
        y: (container.clientHeight - imageDimensions.height) / 2 
      })
    }
  }

  return (
    <div 
      ref={containerRef} 
      className={`relative flex items-center justify-center ${className}`}
    >
      <canvas
        ref={canvasRef}
        className={`rounded-lg shadow-lg max-w-full ${zoom > 1 ? 'cursor-grab' : 'cursor-zoom-in'} ${isPanning ? 'cursor-grabbing' : ''}`}
        style={{ 
          maxHeight: '100%',
          objectFit: 'contain'
        }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />
      
      {/* Contrôles de zoom */}
      <div className={`absolute top-2 right-2 flex flex-col gap-1 p-1 rounded-lg ${
        darkMode ? 'bg-gray-900/80' : 'bg-white/80'
      }`}>
        <button
          onClick={zoomIn}
          disabled={zoom >= MAX_ZOOM}
          className={`p-1.5 rounded transition-colors disabled:opacity-30 ${
            darkMode ? 'hover:bg-gray-700 text-white' : 'hover:bg-gray-200 text-gray-700'
          }`}
          title="Zoom avant"
        >
          <MagnifyingGlassPlusIcon className="w-5 h-5" />
        </button>
        <button
          onClick={zoomOut}
          disabled={zoom <= MIN_ZOOM}
          className={`p-1.5 rounded transition-colors disabled:opacity-30 ${
            darkMode ? 'hover:bg-gray-700 text-white' : 'hover:bg-gray-200 text-gray-700'
          }`}
          title="Zoom arrière"
        >
          <MagnifyingGlassMinusIcon className="w-5 h-5" />
        </button>
        <button
          onClick={resetZoom}
          disabled={zoom === 1}
          className={`p-1.5 rounded transition-colors disabled:opacity-30 ${
            darkMode ? 'hover:bg-gray-700 text-white' : 'hover:bg-gray-200 text-gray-700'
          }`}
          title="Réinitialiser"
        >
          <ArrowsPointingOutIcon className="w-5 h-5" />
        </button>
        <div className={`text-xs text-center py-1 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
          {Math.round(zoom * 100)}%
        </div>
      </div>
      
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
      
      {/* Indication zoom */}
      {zoom > 1 && (
        <div className={`absolute bottom-2 right-2 px-2 py-1 rounded text-xs ${
          darkMode ? 'bg-gray-900/80 text-gray-400' : 'bg-white/80 text-gray-500'
        }`}>
          Glissez pour déplacer
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
