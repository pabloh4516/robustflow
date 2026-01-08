/**
 * ImageComparison Component
 * =========================
 *
 * Exibe comparação visual lado a lado entre:
 * - Imagem original
 * - Imagem adversária (perturbada)
 * - Heatmap de perturbação
 *
 * O heatmap mostra a magnitude da perturbação amplificada
 * para visualização, já que perturbações típicas são imperceptíveis.
 */

import React, { useState } from 'react'
import axios from 'axios'
import { Image as ImageIcon, Layers, Flame, Eye, EyeOff, Download, Loader2 } from 'lucide-react'
import { API_URL } from '../config'

function ImageComparison({ originalImage, adversarialImage, heatmap, attackParams, originalFile }) {
  const [showDifference, setShowDifference] = useState(false)
  const [amplification, setAmplification] = useState(10)
  const [downloading, setDownloading] = useState(false)

  // Função para baixar imagem base64 (baixa resolução)
  const downloadImage = (base64Data, filename) => {
    const link = document.createElement('a')
    link.href = `data:image/png;base64,${base64Data}`
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Função para baixar em ALTA RESOLUÇÃO via API
  const downloadHighRes = async () => {
    if (!originalFile || !attackParams) {
      // Fallback para download normal
      downloadImage(adversarialImage, `adversarial_${Date.now()}.png`)
      return
    }

    setDownloading(true)
    try {
      const formData = new FormData()
      formData.append('file', originalFile)
      formData.append('epsilon', attackParams.epsilon)
      formData.append('alpha', attackParams.alpha)
      formData.append('num_iterations', attackParams.num_iterations)
      formData.append('model_id', attackParams.model_id)
      formData.append('output_format', 'PNG')

      const response = await axios.post(`${API_URL}/attack/download`, formData, {
        responseType: 'blob',
        timeout: 120000
      })

      // Cria link de download
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.download = `adversarial_HD_${Date.now()}.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Erro no download HD:', error)
      // Fallback para download normal
      downloadImage(adversarialImage, `adversarial_${Date.now()}.png`)
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Layers className="w-5 h-5 text-cyan-400" />
          Comparação Visual
        </h2>

        <button
          onClick={() => setShowDifference(!showDifference)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm
                      transition-colors ${
                        showDifference
                          ? 'bg-lab-primary text-white'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
        >
          {showDifference ? (
            <Eye className="w-4 h-4" />
          ) : (
            <EyeOff className="w-4 h-4" />
          )}
          Destacar Diferença
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Imagem Original */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <ImageIcon className="w-4 h-4 text-green-400" />
            <span>Imagem Original</span>
          </div>
          <div className="relative aspect-square bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
            <img
              src={`data:image/png;base64,${originalImage}`}
              alt="Original"
              className="w-full h-full object-contain"
            />
            <div className="absolute bottom-2 left-2 px-2 py-1 bg-green-600/80 rounded text-xs text-white">
              Original
            </div>
          </div>
        </div>

        {/* Imagem Adversária */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm text-gray-400">
            <div className="flex items-center gap-2">
              <ImageIcon className="w-4 h-4 text-red-400" />
              <span>Imagem Adversária</span>
            </div>
            <button
              onClick={downloadHighRes}
              disabled={downloading}
              className="flex items-center gap-1 px-2 py-1 bg-red-600 hover:bg-red-500
                         rounded text-xs text-white transition-colors disabled:opacity-50"
              title="Baixar imagem em alta resolução"
            >
              {downloading ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <Download className="w-3 h-3" />
              )}
              {downloading ? 'Baixando...' : 'Baixar HD'}
            </button>
          </div>
          <div className="relative aspect-square bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
            <img
              src={`data:image/png;base64,${adversarialImage}`}
              alt="Adversarial"
              className={`w-full h-full object-contain transition-all duration-300
                         ${showDifference ? 'brightness-110 contrast-110' : ''}`}
            />
            <div className="absolute bottom-2 left-2 px-2 py-1 bg-red-600/80 rounded text-xs text-white">
              Adversária
            </div>
          </div>
        </div>

        {/* Heatmap de Perturbação */}
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <Flame className="w-4 h-4 text-orange-400" />
            <span>Perturbação (Amplificada)</span>
          </div>
          <div className="relative aspect-square bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
            <img
              src={`data:image/png;base64,${heatmap}`}
              alt="Perturbation Heatmap"
              className="w-full h-full object-contain"
              style={{ filter: `brightness(${amplification / 10})` }}
            />
            <div className="absolute bottom-2 left-2 px-2 py-1 bg-orange-600/80 rounded text-xs text-white">
              δ × {amplification}
            </div>
          </div>
        </div>
      </div>

      {/* Controle de Amplificação */}
      <div className="mt-4 p-3 bg-gray-800/50 rounded-lg">
        <div className="flex items-center gap-4">
          <label className="text-sm text-gray-400 whitespace-nowrap">
            Amplificação do Heatmap:
          </label>
          <input
            type="range"
            min="1"
            max="50"
            value={amplification}
            onChange={(e) => setAmplification(parseInt(e.target.value))}
            className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
                       accent-orange-500"
          />
          <span className="text-sm text-orange-400 mono w-12 text-right">
            ×{amplification}
          </span>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          A perturbação real é imperceptível ao olho humano.
          O heatmap mostra a diferença absoluta amplificada para visualização.
        </p>
      </div>

      {/* Legenda */}
      <div className="mt-4 flex items-center justify-center gap-6 text-xs text-gray-500">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-black rounded" />
          <span>Sem perturbação</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-red-600 rounded" />
          <span>Perturbação baixa</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-yellow-400 rounded" />
          <span>Perturbação alta</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-white rounded" />
          <span>Perturbação máxima</span>
        </div>
      </div>
    </div>
  )
}

export default ImageComparison
