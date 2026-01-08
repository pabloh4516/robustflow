/**
 * TextCamouflage Component
 * ========================
 *
 * Interface para camuflagem de texto - transforma texto normal
 * em versões que apenas humanos conseguem ler.
 */

import React, { useState, useCallback } from 'react'
import axios from 'axios'
import {
  EyeOff,
  Eye,
  Copy,
  Check,
  Wand2,
  FileText,
  Image,
  Settings,
  Sparkles,
  AlertCircle,
  RefreshCw,
  Download,
  Code
} from 'lucide-react'
import { API_URL } from '../config'

// Modos de camuflagem
const CAMOUFLAGE_MODES = [
  {
    id: 'homoglyph',
    name: 'Homoglyphs',
    icon: '🔤',
    description: 'Substitui por caracteres Unicode idênticos (a→а)',
    effectiveness: 'Alta'
  },
  {
    id: 'zero_width',
    name: 'Zero-Width',
    icon: '👻',
    description: 'Injeta caracteres invisíveis entre letras',
    effectiveness: 'Muito Alta'
  },
  {
    id: 'leetspeak',
    name: 'Leetspeak',
    icon: '🎮',
    description: 'Estilo hacker (A→4, E→3, O→0)',
    effectiveness: 'Média'
  },
  {
    id: 'mixed_scripts',
    name: 'Scripts Mistos',
    icon: '🌍',
    description: 'Mistura Latin/Cirílico/Grego',
    effectiveness: 'Muito Alta'
  },
  {
    id: 'combining_marks',
    name: 'Marcas Unicode',
    icon: '✨',
    description: 'Adiciona diacríticos invisíveis',
    effectiveness: 'Alta'
  },
  {
    id: 'full_camouflage',
    name: 'Camuflagem Total',
    icon: '🛡️',
    description: 'Combina TODAS as técnicas',
    effectiveness: 'Máxima'
  }
]

function TextCamouflage() {
  // Estado do input
  const [inputText, setInputText] = useState('')
  const [selectedMode, setSelectedMode] = useState('full_camouflage')
  const [intensity, setIntensity] = useState(0.7)

  // Estado de output
  const [outputMode, setOutputMode] = useState('text') // 'text' ou 'image'
  const [results, setResults] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState(null)
  const [copied, setCopied] = useState(false)

  // Configurações de imagem
  const [fontSize, setFontSize] = useState(40)
  const [imageWidth, setImageWidth] = useState(800)
  const [imageHeight, setImageHeight] = useState(150)

  /**
   * Processa camuflagem de texto
   */
  const processText = async () => {
    if (!inputText.trim()) {
      setError('Digite um texto para camuflar')
      return
    }

    setIsProcessing(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('text', inputText)
      formData.append('mode', selectedMode)
      formData.append('intensity', intensity)

      const response = await axios.post(`${API_URL}/camouflage/text`, formData)
      setResults(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Erro ao processar texto')
    } finally {
      setIsProcessing(false)
    }
  }

  /**
   * Processa camuflagem como imagem
   */
  const processImage = async () => {
    if (!inputText.trim()) {
      setError('Digite um texto para camuflar')
      return
    }

    setIsProcessing(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('text', inputText)
      formData.append('mode', selectedMode)
      formData.append('font_size', fontSize)
      formData.append('width', imageWidth)
      formData.append('height', imageHeight)
      formData.append('add_visual_noise', 'true')

      const response = await axios.post(`${API_URL}/camouflage/image`, formData)
      setResults({ ...response.data, isImage: true })
    } catch (err) {
      setError(err.response?.data?.detail || 'Erro ao criar imagem')
    } finally {
      setIsProcessing(false)
    }
  }

  /**
   * Copia texto para clipboard
   */
  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Erro ao copiar:', err)
    }
  }

  /**
   * Download da imagem
   */
  const downloadImage = () => {
    if (!results?.image) return

    const link = document.createElement('a')
    link.href = `data:image/png;base64,${results.image}`
    link.download = 'texto_camuflado.png'
    link.click()
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-900/50 to-purple-900/50 rounded-lg p-6 border border-indigo-700/30">
        <div className="flex items-center gap-3 mb-3">
          <EyeOff className="w-8 h-8 text-indigo-400" />
          <div>
            <h2 className="text-2xl font-bold text-white">Camuflagem de Texto</h2>
            <p className="text-gray-300 text-sm">
              Texto que apenas humanos conseguem ler - invisível para máquinas
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
          <div className="bg-black/30 rounded p-3 text-center">
            <div className="text-2xl font-bold text-indigo-400">6</div>
            <div className="text-xs text-gray-400">Modos</div>
          </div>
          <div className="bg-black/30 rounded p-3 text-center">
            <div className="text-2xl font-bold text-purple-400">∞</div>
            <div className="text-xs text-gray-400">Variações</div>
          </div>
          <div className="bg-black/30 rounded p-3 text-center">
            <div className="text-2xl font-bold text-pink-400">100%</div>
            <div className="text-xs text-gray-400">Legível</div>
          </div>
          <div className="bg-black/30 rounded p-3 text-center">
            <div className="text-2xl font-bold text-cyan-400">0%</div>
            <div className="text-xs text-gray-400">Para OCR</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Painel de Input */}
        <div className="space-y-4">
          {/* Textarea de Input */}
          <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                <FileText className="w-4 h-4 text-blue-400" />
                Texto Original
              </h3>
              <span className="text-xs text-gray-500">
                {inputText.length} caracteres
              </span>
            </div>

            <textarea
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Digite o texto que deseja camuflar...

Exemplo: Senha: MinhaSenha123
         Email: usuario@email.com
         Código: ABC-XYZ-123"
              className="w-full h-40 px-4 py-3 bg-gray-900 border border-gray-700 rounded-lg
                       text-white text-sm resize-none
                       focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            />
          </div>

          {/* Seletor de Modo */}
          <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
              <Wand2 className="w-4 h-4 text-purple-400" />
              Modo de Camuflagem
            </h3>

            <div className="grid grid-cols-2 gap-2">
              {CAMOUFLAGE_MODES.map(mode => (
                <button
                  key={mode.id}
                  onClick={() => setSelectedMode(mode.id)}
                  className={`p-3 rounded-lg border text-left transition-all
                    ${selectedMode === mode.id
                      ? 'bg-indigo-900/40 border-indigo-500 text-white'
                      : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-600'
                    }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span>{mode.icon}</span>
                    <span className="font-medium text-sm">{mode.name}</span>
                  </div>
                  <p className="text-xs text-gray-500">{mode.description}</p>
                  <span className={`text-xs mt-1 inline-block px-2 py-0.5 rounded
                    ${mode.effectiveness === 'Máxima' ? 'bg-green-900/50 text-green-400' :
                      mode.effectiveness === 'Muito Alta' ? 'bg-blue-900/50 text-blue-400' :
                      mode.effectiveness === 'Alta' ? 'bg-purple-900/50 text-purple-400' :
                      'bg-yellow-900/50 text-yellow-400'
                    }`}>
                    {mode.effectiveness}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Controles */}
          <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
              <Settings className="w-4 h-4 text-gray-400" />
              Configurações
            </h3>

            <div className="space-y-4">
              {/* Intensidade */}
              <div>
                <div className="flex justify-between text-xs text-gray-400 mb-1">
                  <span>Intensidade</span>
                  <span>{Math.round(intensity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.1"
                  value={intensity}
                  onChange={(e) => setIntensity(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500"
                />
              </div>

              {/* Tipo de Output */}
              <div>
                <label className="text-xs text-gray-400 block mb-2">Tipo de Saída</label>
                <div className="flex gap-2">
                  <button
                    onClick={() => setOutputMode('text')}
                    className={`flex-1 py-2 px-3 rounded-lg text-sm flex items-center justify-center gap-2
                      ${outputMode === 'text'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                      }`}
                  >
                    <Code className="w-4 h-4" />
                    Texto
                  </button>
                  <button
                    onClick={() => setOutputMode('image')}
                    className={`flex-1 py-2 px-3 rounded-lg text-sm flex items-center justify-center gap-2
                      ${outputMode === 'image'
                        ? 'bg-indigo-600 text-white'
                        : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                      }`}
                  >
                    <Image className="w-4 h-4" />
                    Imagem
                  </button>
                </div>
              </div>

              {/* Config de imagem */}
              {outputMode === 'image' && (
                <div className="grid grid-cols-3 gap-2">
                  <div>
                    <label className="text-xs text-gray-500">Fonte</label>
                    <input
                      type="number"
                      value={fontSize}
                      onChange={(e) => setFontSize(parseInt(e.target.value) || 20)}
                      className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-white"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">Largura</label>
                    <input
                      type="number"
                      value={imageWidth}
                      onChange={(e) => setImageWidth(parseInt(e.target.value) || 400)}
                      className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-white"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-500">Altura</label>
                    <input
                      type="number"
                      value={imageHeight}
                      onChange={(e) => setImageHeight(parseInt(e.target.value) || 100)}
                      className="w-full px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-white"
                    />
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Botão de Processar */}
          <button
            onClick={outputMode === 'text' ? processText : processImage}
            disabled={!inputText.trim() || isProcessing}
            className={`w-full py-3 px-4 rounded-lg font-semibold text-white
              transition-all duration-200 flex items-center justify-center gap-2
              ${!inputText.trim() || isProcessing
                ? 'bg-gray-600 cursor-not-allowed'
                : 'bg-gradient-to-r from-indigo-600 to-purple-600 hover:opacity-90'
              }`}
          >
            {isProcessing ? (
              <>
                <RefreshCw className="w-5 h-5 animate-spin" />
                Processando...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                Camuflar Texto
              </>
            )}
          </button>

          {error && (
            <div className="p-3 bg-red-900/30 border border-red-700/50 rounded-lg flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-400" />
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          )}
        </div>

        {/* Painel de Output */}
        <div className="space-y-4">
          {results ? (
            <>
              {/* Resultado */}
              <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                    <EyeOff className="w-4 h-4 text-green-400" />
                    Texto Camuflado
                  </h3>
                  {!results.isImage && (
                    <button
                      onClick={() => copyToClipboard(results.camouflaged?.text || results.camouflaged_text)}
                      className="flex items-center gap-1 px-3 py-1 bg-gray-700 rounded text-sm
                               text-gray-300 hover:bg-gray-600 transition-colors"
                    >
                      {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                      {copied ? 'Copiado!' : 'Copiar'}
                    </button>
                  )}
                  {results.isImage && (
                    <button
                      onClick={downloadImage}
                      className="flex items-center gap-1 px-3 py-1 bg-gray-700 rounded text-sm
                               text-gray-300 hover:bg-gray-600 transition-colors"
                    >
                      <Download className="w-4 h-4" />
                      Download
                    </button>
                  )}
                </div>

                {results.isImage ? (
                  <div className="bg-white rounded-lg p-4">
                    <img
                      src={`data:image/png;base64,${results.image}`}
                      alt="Texto Camuflado"
                      className="max-w-full h-auto mx-auto"
                    />
                  </div>
                ) : (
                  <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm text-green-300 break-all">
                    {results.camouflaged?.text || results.camouflaged_text}
                  </div>
                )}
              </div>

              {/* Comparação */}
              {!results.isImage && (
                <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
                  <h3 className="text-sm font-medium text-gray-300 mb-3">Comparação</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">Original:</span>
                      <p className="text-gray-300">{results.original?.length || 0} chars</p>
                      <p className="text-gray-500">{results.original?.bytes || 0} bytes</p>
                    </div>
                    <div>
                      <span className="text-gray-500">Camuflado:</span>
                      <p className="text-green-300">{results.camouflaged?.length || 0} chars</p>
                      <p className="text-green-500">{results.camouflaged?.bytes || 0} bytes</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Análise Unicode */}
              {results.unicode_analysis && (
                <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
                  <h3 className="text-sm font-medium text-gray-300 mb-3">Análise Unicode</h3>
                  <div className="space-y-2 text-xs">
                    {results.unicode_analysis.has_invisible_chars && (
                      <div className="flex items-center gap-2 text-purple-400">
                        <Check className="w-3 h-3" />
                        Contém caracteres invisíveis
                      </div>
                    )}
                    {results.unicode_analysis.has_mixed_scripts && (
                      <div className="flex items-center gap-2 text-blue-400">
                        <Check className="w-3 h-3" />
                        Scripts misturados detectados
                      </div>
                    )}
                    {results.unicode_analysis.scripts_detected?.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {results.unicode_analysis.scripts_detected.map(script => (
                          <span key={script} className="px-2 py-1 bg-gray-800 rounded text-gray-400">
                            {script}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Dicas de Uso */}
              {results.usage_tips && (
                <div className="bg-indigo-900/20 border border-indigo-700/30 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-indigo-300 mb-2">Como Usar</h3>
                  <ul className="text-xs text-gray-400 space-y-1">
                    {results.usage_tips.map((tip, i) => (
                      <li key={i} className="flex items-start gap-2">
                        <span className="text-indigo-400">•</span>
                        {tip}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          ) : (
            /* Placeholder */
            <div className="bg-lab-dark rounded-lg border border-gray-700 p-12 text-center h-full flex flex-col items-center justify-center">
              <EyeOff className="w-16 h-16 text-gray-600 mb-4" />
              <h3 className="text-xl font-semibold text-gray-400 mb-2">
                Camuflagem de Texto
              </h3>
              <p className="text-gray-500 max-w-sm">
                Digite um texto, escolha o modo de camuflagem
                e clique em "Camuflar Texto" para gerar uma
                versão que só humanos conseguem ler.
              </p>

              <div className="mt-6 p-4 bg-gray-800/50 rounded-lg text-left max-w-sm">
                <p className="text-xs text-gray-500 mb-2">Exemplo de resultado:</p>
                <p className="text-sm text-gray-300">
                  <span className="text-gray-500">Original:</span> Hello World
                </p>
                <p className="text-sm text-green-400">
                  <span className="text-gray-500">Camuflado:</span> Неllо Wоrld
                </p>
                <p className="text-xs text-gray-600 mt-2">
                  (Visualmente igual, Unicode diferente)
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default TextCamouflage
