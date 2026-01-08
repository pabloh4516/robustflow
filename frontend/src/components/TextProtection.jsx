/**
 * TextProtection Component
 * ========================
 *
 * Interface completa para proteção de texto contra OCR.
 * Permite proteger imagens existentes ou criar texto protegido.
 * Inclui download em alta qualidade.
 */

import React, { useState, useCallback } from 'react'
import axios from 'axios'
import {
  Shield,
  Eye,
  EyeOff,
  Type,
  Upload,
  Settings,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Zap,
  FileText,
  Lock,
  Download,
  Image,
  FileImage,
  Maximize2
} from 'lucide-react'
import { API_URL } from '../config'

// Níveis de proteção
const PROTECTION_LEVELS = [
  {
    id: 'readable',
    name: 'Legível (Recomendado)',
    color: 'emerald',
    description: 'Texto perfeito para humanos, ataques invisíveis devastam OCR (70-90%)',
    icon: '✨',
    recommended: true
  },
  {
    id: 'stealth',
    name: 'Stealth',
    color: 'cyan',
    description: 'Perturbações 100% invisíveis, imagem idêntica ao original',
    icon: '👁️‍🗨️'
  },
  { id: 'low', name: 'Baixo', color: 'green', description: 'Sutil, pode não bloquear todos OCRs' },
  { id: 'medium', name: 'Médio', color: 'yellow', description: 'Balanço entre invisibilidade e eficácia' },
  { id: 'high', name: 'Alto', color: 'orange', description: 'Bloqueia maioria dos OCRs, artefatos leves' },
  { id: 'maximum', name: 'Máximo', color: 'red', description: 'Proteção máxima, TERÁ artefatos visíveis' }
]

// Técnicas disponíveis
const TECHNIQUES = [
  { id: 'adversarial_noise', name: 'Ruído Adversarial', icon: '🎲' },
  { id: 'structured_pattern', name: 'Padrão Estruturado', icon: '📊' },
  { id: 'frequency_perturbation', name: 'Perturbação FFT', icon: '📡' },
  { id: 'geometric_distortion', name: 'Distorção Geométrica', icon: '🌀' },
  { id: 'edge_disruption', name: 'Disrupção de Bordas', icon: '✂️' },
  { id: 'color_channel_shift', name: 'Shift de Canais', icon: '🎨' },
  { id: 'adversarial_texture', name: 'Textura Adversarial', icon: '🔲' },
  { id: 'micro_patterns', name: 'Micro Padrões', icon: '🔬' },
  { id: 'gradient_masking', name: 'Mascaramento', icon: '🎭' },
  { id: 'dithering_noise', name: 'Ruído Dithering', icon: '📺' }
]

// Formatos de imagem
const IMAGE_FORMATS = [
  { id: 'PNG', name: 'PNG', description: 'Sem perda de qualidade (recomendado)' },
  { id: 'JPEG', name: 'JPEG', description: 'Menor tamanho, leve perda' },
  { id: 'WEBP', name: 'WebP', description: 'Moderno, boa compressão' }
]

// Escalas de resolução
const RESOLUTION_SCALES = [
  { id: 1, name: '1x (Original)', description: 'Mantém resolução original' },
  { id: 1.5, name: '1.5x', description: '50% maior' },
  { id: 2, name: '2x (HD)', description: 'Dobro da resolução' },
  { id: 3, name: '3x', description: 'Triplo da resolução' },
  { id: 4, name: '4x (Ultra HD)', description: 'Máxima qualidade' }
]

function TextProtection() {
  // Estado do modo (upload ou criar)
  const [mode, setMode] = useState('upload') // 'upload' ou 'create'

  // Estado para upload
  const [selectedFile, setSelectedFile] = useState(null)
  const [filePreview, setFilePreview] = useState(null)

  // Estado para criação
  const [inputText, setInputText] = useState('')
  const [fontSize, setFontSize] = useState(60)
  const [imageWidth, setImageWidth] = useState(1920)
  const [imageHeight, setImageHeight] = useState(400)
  const [bgColor, setBgColor] = useState('#FFFFFF')
  const [textColor, setTextColor] = useState('#000000')

  // Configurações de proteção
  const [protectionLevel, setProtectionLevel] = useState('readable')
  const [selectedTechniques, setSelectedTechniques] = useState([])
  const [useAllTechniques, setUseAllTechniques] = useState(true)

  // Configurações de download
  const [downloadFormat, setDownloadFormat] = useState('PNG')
  const [downloadQuality, setDownloadQuality] = useState(100)
  const [downloadScale, setDownloadScale] = useState(1)

  // Estado de execução
  const [isProcessing, setIsProcessing] = useState(false)
  const [isDownloading, setIsDownloading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  /**
   * Handler para upload de arquivo
   */
  const handleFileUpload = useCallback((e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setResults(null)
      setError(null)

      const reader = new FileReader()
      reader.onload = (e) => setFilePreview(e.target.result)
      reader.readAsDataURL(file)
    }
  }, [])

  /**
   * Toggle técnica individual
   */
  const toggleTechnique = (techId) => {
    setSelectedTechniques(prev => {
      if (prev.includes(techId)) {
        return prev.filter(t => t !== techId)
      }
      return [...prev, techId]
    })
  }

  /**
   * Proteger imagem existente
   */
  const protectImage = async () => {
    if (!selectedFile) {
      setError('Selecione uma imagem primeiro')
      return
    }

    setIsProcessing(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      formData.append('protection_level', protectionLevel)
      formData.append('preserve_colors', 'true')

      if (useAllTechniques) {
        formData.append('techniques', 'all')
      } else if (selectedTechniques.length > 0) {
        formData.append('techniques', selectedTechniques.join(','))
      } else {
        formData.append('techniques', 'all')
      }

      const response = await axios.post(
        `${API_URL}/text-protection/protect`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 120000 }
      )

      setResults(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Erro ao proteger imagem')
    } finally {
      setIsProcessing(false)
    }
  }

  /**
   * Criar texto protegido
   */
  const createProtectedText = async () => {
    if (!inputText.trim()) {
      setError('Digite um texto primeiro')
      return
    }

    setIsProcessing(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('text', inputText)
      formData.append('font_size', fontSize)
      formData.append('width', imageWidth)
      formData.append('height', imageHeight)
      formData.append('protection_level', protectionLevel)
      formData.append('background_color', bgColor)
      formData.append('text_color', textColor)

      const response = await axios.post(
        `${API_URL}/text-protection/create`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 120000 }
      )

      setResults(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Erro ao criar texto protegido')
    } finally {
      setIsProcessing(false)
    }
  }

  /**
   * Download da imagem protegida em alta qualidade
   */
  const downloadProtectedImage = async () => {
    setIsDownloading(true)

    try {
      const formData = new FormData()

      if (mode === 'upload' && selectedFile) {
        formData.append('file', selectedFile)
        formData.append('protection_level', protectionLevel)
        formData.append('quality', downloadQuality)
        formData.append('format', downloadFormat)
        formData.append('scale', downloadScale)

        const response = await axios.post(
          `${API_URL}/text-protection/download`,
          formData,
          {
            headers: { 'Content-Type': 'multipart/form-data' },
            responseType: 'blob',
            timeout: 180000
          }
        )

        // Criar link de download
        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        link.setAttribute('download', `protected_image_${protectionLevel}_${downloadScale}x.${downloadFormat.toLowerCase()}`)
        document.body.appendChild(link)
        link.click()
        link.remove()
        window.URL.revokeObjectURL(url)

      } else if (mode === 'create' && inputText.trim()) {
        formData.append('text', inputText)
        formData.append('font_size', Math.round(fontSize * downloadScale))
        formData.append('width', Math.round(imageWidth * downloadScale))
        formData.append('height', Math.round(imageHeight * downloadScale))
        formData.append('protection_level', protectionLevel)
        formData.append('background_color', bgColor)
        formData.append('text_color', textColor)
        formData.append('quality', downloadQuality)
        formData.append('format', downloadFormat)

        const response = await axios.post(
          `${API_URL}/text-protection/create-download`,
          formData,
          {
            headers: { 'Content-Type': 'multipart/form-data' },
            responseType: 'blob',
            timeout: 180000
          }
        )

        const url = window.URL.createObjectURL(new Blob([response.data]))
        const link = document.createElement('a')
        link.href = url
        const safeText = inputText.substring(0, 20).replace(/[^a-zA-Z0-9]/g, '_')
        link.setAttribute('download', `protected_text_${safeText}.${downloadFormat.toLowerCase()}`)
        document.body.appendChild(link)
        link.click()
        link.remove()
        window.URL.revokeObjectURL(url)
      }
    } catch (err) {
      setError('Erro ao baixar imagem: ' + (err.response?.data?.detail || err.message))
    } finally {
      setIsDownloading(false)
    }
  }

  /**
   * Download rápido do preview (base64)
   */
  const downloadPreview = () => {
    if (!results?.protected_image) return

    const link = document.createElement('a')
    link.href = `data:image/png;base64,${results.protected_image}`
    link.download = `protected_preview.png`
    link.click()
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900/50 to-blue-900/50 rounded-lg p-6 border border-purple-700/30">
        <div className="flex items-center gap-3 mb-3">
          <Lock className="w-8 h-8 text-purple-400" />
          <div>
            <h2 className="text-2xl font-bold text-white">Proteção Anti-OCR</h2>
            <p className="text-gray-300 text-sm">
              Torne texto legível apenas para humanos, não para máquinas
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mt-4">
          <div className="bg-black/30 rounded p-3 text-center">
            <div className="text-2xl font-bold text-green-400">15</div>
            <div className="text-xs text-gray-400">Técnicas</div>
          </div>
          <div className="bg-black/30 rounded p-3 text-center">
            <div className="text-2xl font-bold text-blue-400">6</div>
            <div className="text-xs text-gray-400">Níveis</div>
          </div>
          <div className="bg-black/30 rounded p-3 text-center border border-emerald-500/50">
            <div className="text-2xl font-bold text-emerald-400">✨</div>
            <div className="text-xs text-emerald-400">100% Legível</div>
          </div>
          <div className="bg-black/30 rounded p-3 text-center">
            <div className="text-2xl font-bold text-yellow-400">4K</div>
            <div className="text-xs text-gray-400">Até Ultra HD</div>
          </div>
          <div className="bg-black/30 rounded p-3 text-center">
            <div className="text-2xl font-bold text-purple-400">PNG</div>
            <div className="text-xs text-gray-400">Sem Perda</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Painel de Configuração */}
        <div className="lg:col-span-1 space-y-4">
          {/* Seletor de Modo */}
          <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-3">Modo de Operação</h3>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => setMode('upload')}
                className={`p-3 rounded-lg border transition-all flex flex-col items-center gap-2
                  ${mode === 'upload'
                    ? 'bg-lab-primary/20 border-lab-primary text-white'
                    : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600'
                  }`}
              >
                <Upload className="w-5 h-5" />
                <span className="text-xs">Upload Imagem</span>
              </button>
              <button
                onClick={() => setMode('create')}
                className={`p-3 rounded-lg border transition-all flex flex-col items-center gap-2
                  ${mode === 'create'
                    ? 'bg-lab-primary/20 border-lab-primary text-white'
                    : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600'
                  }`}
              >
                <Type className="w-5 h-5" />
                <span className="text-xs">Criar Texto</span>
              </button>
            </div>
          </div>

          {/* Input baseado no modo */}
          {mode === 'upload' ? (
            <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
              <h3 className="text-sm font-medium text-gray-300 mb-3">Imagem com Texto</h3>
              <label className="block cursor-pointer">
                <div className={`border-2 border-dashed rounded-lg p-4 text-center transition-all
                  ${filePreview ? 'border-gray-600' : 'border-gray-600 hover:border-lab-primary'}`}>
                  {filePreview ? (
                    <img src={filePreview} alt="Preview" className="max-h-32 mx-auto rounded" />
                  ) : (
                    <>
                      <Upload className="w-8 h-8 text-gray-500 mx-auto mb-2" />
                      <p className="text-sm text-gray-400">Clique ou arraste uma imagem</p>
                    </>
                  )}
                </div>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </label>
            </div>
          ) : (
            <div className="bg-lab-dark rounded-lg border border-gray-700 p-4 space-y-4">
              <h3 className="text-sm font-medium text-gray-300 mb-3">Criar Texto Protegido</h3>

              <div>
                <label className="block text-xs text-gray-400 mb-1">Texto</label>
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Digite o texto a proteger..."
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded
                           text-white text-sm resize-none h-20
                           focus:outline-none focus:ring-2 focus:ring-lab-primary"
                />
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Fonte</label>
                  <input
                    type="number"
                    value={fontSize}
                    onChange={(e) => setFontSize(parseInt(e.target.value) || 20)}
                    min="12"
                    max="200"
                    className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded
                             text-white text-sm focus:outline-none focus:ring-2 focus:ring-lab-primary"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Largura</label>
                  <input
                    type="number"
                    value={imageWidth}
                    onChange={(e) => setImageWidth(parseInt(e.target.value) || 400)}
                    min="200"
                    max="4000"
                    className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded
                             text-white text-sm focus:outline-none focus:ring-2 focus:ring-lab-primary"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Altura</label>
                  <input
                    type="number"
                    value={imageHeight}
                    onChange={(e) => setImageHeight(parseInt(e.target.value) || 100)}
                    min="50"
                    max="2000"
                    className="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded
                             text-white text-sm focus:outline-none focus:ring-2 focus:ring-lab-primary"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Cor Fundo</label>
                  <input
                    type="color"
                    value={bgColor}
                    onChange={(e) => setBgColor(e.target.value)}
                    className="w-full h-8 rounded cursor-pointer"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Cor Texto</label>
                  <input
                    type="color"
                    value={textColor}
                    onChange={(e) => setTextColor(e.target.value)}
                    className="w-full h-8 rounded cursor-pointer"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Nível de Proteção */}
          <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
              <Shield className="w-4 h-4 text-purple-400" />
              Nível de Proteção
            </h3>
            <div className="space-y-2">
              {PROTECTION_LEVELS.map(level => (
                <button
                  key={level.id}
                  onClick={() => setProtectionLevel(level.id)}
                  className={`w-full p-2 rounded-lg border text-left transition-all
                    ${protectionLevel === level.id
                      ? level.id === 'readable'
                        ? 'bg-emerald-900/30 border-emerald-500/70 ring-1 ring-emerald-500/30'
                        : level.id === 'stealth'
                          ? 'bg-cyan-900/30 border-cyan-500/70 ring-1 ring-cyan-500/30'
                          : `bg-${level.color}-900/30 border-${level.color}-700/50`
                      : level.id === 'readable'
                        ? 'bg-gray-800 border-emerald-700/30 hover:border-emerald-600/50'
                        : level.id === 'stealth'
                          ? 'bg-gray-800 border-cyan-700/30 hover:border-cyan-600/50'
                          : 'bg-gray-800 border-gray-700 hover:border-gray-600'
                    }`}
                >
                  <div className="flex items-center justify-between">
                    <span className={`font-medium ${
                      protectionLevel === level.id
                        ? level.id === 'readable' ? 'text-emerald-400'
                          : level.id === 'stealth' ? 'text-cyan-400' : `text-${level.color}-400`
                        : level.id === 'readable' ? 'text-emerald-300'
                          : level.id === 'stealth' ? 'text-cyan-300' : 'text-gray-300'
                    }`}>
                      {level.icon && <span className="mr-2">{level.icon}</span>}
                      {level.name}
                    </span>
                    <div className="flex items-center gap-2">
                      {level.recommended && (
                        <span className={`px-1.5 py-0.5 text-xs rounded ${
                          level.id === 'readable' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-cyan-500/20 text-cyan-400'
                        }`}>
                          Recomendado
                        </span>
                      )}
                      {protectionLevel === level.id && (
                        <CheckCircle className={`w-4 h-4 ${
                          level.id === 'readable' ? 'text-emerald-400'
                            : level.id === 'stealth' ? 'text-cyan-400' : `text-${level.color}-400`
                        }`} />
                      )}
                    </div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">{level.description}</p>
                </button>
              ))}
            </div>
          </div>

          {/* Configurações de Download */}
          <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
              <Download className="w-4 h-4 text-green-400" />
              Configurações de Download
            </h3>

            <div className="space-y-3">
              {/* Formato */}
              <div>
                <label className="block text-xs text-gray-400 mb-2">Formato</label>
                <div className="grid grid-cols-3 gap-2">
                  {IMAGE_FORMATS.map(fmt => (
                    <button
                      key={fmt.id}
                      onClick={() => setDownloadFormat(fmt.id)}
                      className={`p-2 rounded border text-xs transition-all
                        ${downloadFormat === fmt.id
                          ? 'bg-green-900/30 border-green-700/50 text-green-400'
                          : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600'
                        }`}
                    >
                      {fmt.name}
                    </button>
                  ))}
                </div>
              </div>

              {/* Escala/Resolução */}
              <div>
                <label className="block text-xs text-gray-400 mb-2">Resolução</label>
                <select
                  value={downloadScale}
                  onChange={(e) => setDownloadScale(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-600 rounded
                           text-white text-sm focus:outline-none focus:ring-2 focus:ring-green-500"
                >
                  {RESOLUTION_SCALES.map(scale => (
                    <option key={scale.id} value={scale.id}>
                      {scale.name} - {scale.description}
                    </option>
                  ))}
                </select>
              </div>

              {/* Qualidade (apenas para JPEG/WEBP) */}
              {downloadFormat !== 'PNG' && (
                <div>
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Qualidade</span>
                    <span>{downloadQuality}%</span>
                  </div>
                  <input
                    type="range"
                    min="50"
                    max="100"
                    value={downloadQuality}
                    onChange={(e) => setDownloadQuality(parseInt(e.target.value))}
                    className="w-full accent-green-500"
                  />
                </div>
              )}

              {/* Info de resolução final */}
              <div className="p-2 bg-gray-800/50 rounded text-xs text-gray-400">
                <div className="flex items-center gap-2">
                  <Maximize2 className="w-3 h-3" />
                  <span>
                    Resolução final: {Math.round((mode === 'upload' ? 1920 : imageWidth) * downloadScale)} x {Math.round((mode === 'upload' ? 1080 : imageHeight) * downloadScale)}px
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Botões de Ação */}
          <div className="space-y-2">
            {/* Botão de Preview */}
            <button
              onClick={mode === 'upload' ? protectImage : createProtectedText}
              disabled={isProcessing || (mode === 'upload' ? !selectedFile : !inputText.trim())}
              className={`w-full py-3 px-4 rounded-lg font-semibold text-white
                transition-all duration-200 flex items-center justify-center gap-2
                ${isProcessing || (mode === 'upload' ? !selectedFile : !inputText.trim())
                  ? 'bg-gray-600 cursor-not-allowed'
                  : 'bg-gradient-to-r from-purple-600 to-blue-600 hover:opacity-90'
                }`}
            >
              {isProcessing ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Processando...
                </>
              ) : (
                <>
                  <Eye className="w-5 h-5" />
                  Visualizar Proteção
                </>
              )}
            </button>

            {/* Botão de Download HD */}
            <button
              onClick={downloadProtectedImage}
              disabled={isDownloading || (mode === 'upload' ? !selectedFile : !inputText.trim())}
              className={`w-full py-3 px-4 rounded-lg font-semibold text-white
                transition-all duration-200 flex items-center justify-center gap-2
                ${isDownloading || (mode === 'upload' ? !selectedFile : !inputText.trim())
                  ? 'bg-gray-600 cursor-not-allowed'
                  : 'bg-gradient-to-r from-green-600 to-emerald-600 hover:opacity-90'
                }`}
            >
              {isDownloading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Preparando Download...
                </>
              ) : (
                <>
                  <Download className="w-5 h-5" />
                  Download Alta Qualidade ({downloadScale}x {downloadFormat})
                </>
              )}
            </button>
          </div>

          {/* Mensagem de Erro */}
          {error && (
            <div className="p-3 bg-red-900/30 border border-red-700/50 rounded-lg">
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          )}
        </div>

        {/* Painel de Resultados */}
        <div className="lg:col-span-2">
          {results ? (
            <TextProtectionResults results={results} onDownloadPreview={downloadPreview} />
          ) : (
            <div className="bg-lab-dark rounded-lg border border-gray-700 p-12 text-center h-full flex flex-col items-center justify-center">
              <Lock className="w-16 h-16 text-gray-600 mb-4" />
              <h3 className="text-xl font-semibold text-gray-400 mb-2">
                Proteção Anti-OCR
              </h3>
              <p className="text-gray-500 max-w-md">
                Selecione uma imagem ou crie um texto para proteger contra
                reconhecimento automático por sistemas OCR.
              </p>
              <div className="mt-6 p-4 bg-gray-800/50 rounded-lg">
                <p className="text-xs text-gray-500 mb-2">Formatos de saída disponíveis:</p>
                <div className="flex gap-2 justify-center">
                  <span className="px-2 py-1 bg-green-900/30 text-green-400 rounded text-xs">PNG (Sem perda)</span>
                  <span className="px-2 py-1 bg-blue-900/30 text-blue-400 rounded text-xs">JPEG</span>
                  <span className="px-2 py-1 bg-purple-900/30 text-purple-400 rounded text-xs">WebP</span>
                </div>
                <p className="text-xs text-gray-500 mt-2">Até 4x de resolução (Ultra HD)</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/**
 * Componente de Resultados
 */
function TextProtectionResults({ results, onDownloadPreview }) {
  const [showOriginal, setShowOriginal] = useState(true)

  const ocrComparison = results.ocr_comparison || {}
  const engines = ocrComparison.engines || {}
  const summary = ocrComparison.summary || {}

  return (
    <div className="space-y-4">
      {/* Comparação Visual */}
      <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Eye className="w-5 h-5 text-green-400" />
            Comparação Visual
          </h3>
          <div className="flex gap-2">
            <button
              onClick={() => setShowOriginal(!showOriginal)}
              className="flex items-center gap-2 px-3 py-1.5 bg-gray-700 rounded-lg text-sm
                       text-gray-300 hover:bg-gray-600 transition-colors"
            >
              {showOriginal ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
              {showOriginal ? 'Original' : 'Protegida'}
            </button>
            <button
              onClick={onDownloadPreview}
              className="flex items-center gap-2 px-3 py-1.5 bg-green-700 rounded-lg text-sm
                       text-white hover:bg-green-600 transition-colors"
            >
              <Download className="w-4 h-4" />
              Preview
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Imagem Original */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <div className="w-3 h-3 bg-green-500 rounded-full" />
              Original (Legível por OCR)
            </div>
            <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
              <img
                src={`data:image/png;base64,${results.original_image}`}
                alt="Original"
                className="w-full h-auto"
              />
            </div>
          </div>

          {/* Imagem Protegida */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <div className="w-3 h-3 bg-purple-500 rounded-full" />
              Protegida (Ilegível para OCR)
            </div>
            <div className="bg-gray-900 rounded-lg overflow-hidden border border-purple-700/50">
              <img
                src={`data:image/png;base64,${results.protected_image}`}
                alt="Protected"
                className="w-full h-auto"
              />
            </div>
          </div>
        </div>

        {/* Heatmap */}
        {results.difference_heatmap && (
          <div className="mt-4">
            <div className="flex items-center gap-2 text-sm text-gray-400 mb-2">
              <div className="w-3 h-3 bg-orange-500 rounded-full" />
              Mapa de Perturbação (Amplificado)
            </div>
            <div className="bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
              <img
                src={`data:image/png;base64,${results.difference_heatmap}`}
                alt="Difference"
                className="w-full h-auto max-h-32 object-contain"
              />
            </div>
          </div>
        )}
      </div>

      {/* Métricas de Sucesso */}
      <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-400" />
          Eficácia da Proteção
        </h3>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <div className={`p-4 rounded-lg ${
            summary.protection_success_rate >= 0.7 ? 'bg-green-900/30' : 'bg-yellow-900/30'
          }`}>
            <div className={`text-2xl font-bold ${
              summary.protection_success_rate >= 0.7 ? 'text-green-400' : 'text-yellow-400'
            }`}>
              {((summary.protection_success_rate || 0) * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-gray-400">Taxa de Sucesso</div>
          </div>

          <div className="p-4 rounded-lg bg-blue-900/30">
            <div className="text-2xl font-bold text-blue-400">
              {(results.protection_metrics?.psnr || 0).toFixed(1)}dB
            </div>
            <div className="text-xs text-gray-400">PSNR</div>
          </div>

          <div className="p-4 rounded-lg bg-purple-900/30">
            <div className="text-2xl font-bold text-purple-400">
              {results.protection_metrics?.techniques_applied?.length || 0}
            </div>
            <div className="text-xs text-gray-400">Técnicas Usadas</div>
          </div>

          <div className="p-4 rounded-lg bg-cyan-900/30">
            <div className="text-2xl font-bold text-cyan-400">
              {results.protection_metrics?.estimated_ocr_accuracy_drop || 0}%
            </div>
            <div className="text-xs text-gray-400">Queda Estimada OCR</div>
          </div>
        </div>

        {/* Comparação por Engine */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-300">Resultado por Engine OCR</h4>
          {Object.entries(engines).map(([engineName, data]) => (
            <div key={engineName} className="p-3 bg-gray-800 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-gray-200 capitalize">{engineName}</span>
                {data.protection_success ? (
                  <span className="flex items-center gap-1 text-green-400 text-sm">
                    <CheckCircle className="w-4 h-4" /> Protegido
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-yellow-400 text-sm">
                    <AlertTriangle className="w-4 h-4" /> Parcial
                  </span>
                )}
              </div>

              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <span className="text-gray-500">Texto Original:</span>
                  <p className="text-gray-300 truncate" title={data.original_text}>
                    "{data.original_text || 'N/A'}"
                  </p>
                </div>
                <div>
                  <span className="text-gray-500">Texto Após Proteção:</span>
                  <p className="text-gray-300 truncate" title={data.protected_text}>
                    "{data.protected_text || 'N/A'}"
                  </p>
                </div>
              </div>

              <div className="mt-2 flex items-center gap-4 text-xs">
                <span className="text-gray-500">
                  Confiança: {((data.original_confidence || 0) * 100).toFixed(0)}% →{' '}
                  <span className={(data.protected_confidence || 0) < (data.original_confidence || 0) ? 'text-green-400' : 'text-red-400'}>
                    {((data.protected_confidence || 0) * 100).toFixed(0)}%
                  </span>
                </span>
                <span className="text-gray-500">
                  Similaridade: {((data.text_similarity || 0) * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))}

          {Object.keys(engines).length === 0 && (
            <p className="text-gray-500 text-sm text-center py-4">
              Nenhum engine OCR disponível para teste.
              Instale pytesseract ou easyocr para comparação.
            </p>
          )}
        </div>
      </div>

      {/* Técnicas Aplicadas */}
      <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
        <h3 className="text-sm font-medium text-gray-300 mb-3">Técnicas Aplicadas</h3>
        <div className="flex flex-wrap gap-2">
          {(results.protection_metrics?.techniques_applied || []).map(tech => (
            <span
              key={tech}
              className="px-2 py-1 bg-purple-900/30 text-purple-300 rounded text-xs"
            >
              {tech.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

export default TextProtection
