/**
 * ImageUpload Component
 * =====================
 *
 * Componente de upload de imagens com drag-and-drop.
 * Suporta visualização de preview após seleção.
 */

import React, { useCallback, useRef } from 'react'
import { Upload, Image as ImageIcon, X } from 'lucide-react'

function ImageUpload({ onImageUpload, preview }) {
  const fileInputRef = useRef(null)

  /**
   * Handler para drag over
   */
  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    e.currentTarget.classList.add('border-lab-primary', 'bg-lab-primary/10')
  }, [])

  /**
   * Handler para drag leave
   */
  const handleDragLeave = useCallback((e) => {
    e.preventDefault()
    e.currentTarget.classList.remove('border-lab-primary', 'bg-lab-primary/10')
  }, [])

  /**
   * Handler para drop
   */
  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.currentTarget.classList.remove('border-lab-primary', 'bg-lab-primary/10')

    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      onImageUpload(file)
    }
  }, [onImageUpload])

  /**
   * Handler para seleção via input
   */
  const handleFileSelect = useCallback((e) => {
    const file = e.target.files[0]
    if (file) {
      onImageUpload(file)
    }
  }, [onImageUpload])

  /**
   * Abre o seletor de arquivo
   */
  const openFileSelector = () => {
    fileInputRef.current?.click()
  }

  /**
   * Remove a imagem selecionada
   */
  const clearImage = (e) => {
    e.stopPropagation()
    onImageUpload(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-gray-300">
        Imagem de Entrada
      </label>

      <div
        onClick={openFileSelector}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          relative border-2 border-dashed rounded-lg cursor-pointer
          transition-all duration-200 overflow-hidden
          ${preview
            ? 'border-gray-600'
            : 'border-gray-600 hover:border-lab-primary hover:bg-lab-primary/5'
          }
        `}
      >
        {preview ? (
          /* Preview da imagem */
          <div className="relative">
            <img
              src={preview}
              alt="Preview"
              className="w-full h-48 object-contain bg-gray-900"
            />
            <button
              onClick={clearImage}
              className="absolute top-2 right-2 p-1 bg-red-600 rounded-full
                         hover:bg-red-500 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
            <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/80 to-transparent">
              <p className="text-xs text-gray-300 flex items-center gap-1">
                <ImageIcon className="w-3 h-3" />
                Clique para trocar a imagem
              </p>
            </div>
          </div>
        ) : (
          /* Área de upload vazia */
          <div className="p-8 text-center">
            <Upload className="w-10 h-10 text-gray-500 mx-auto mb-3" />
            <p className="text-sm text-gray-400 mb-1">
              Arraste uma imagem ou clique para selecionar
            </p>
            <p className="text-xs text-gray-500">
              PNG, JPEG, WebP (máx. 10MB)
            </p>
          </div>
        )}
      </div>

      {/* Input escondido */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
      />
    </div>
  )
}

export default ImageUpload
