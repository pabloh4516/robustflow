/**
 * MediaProtection Component
 * =========================
 *
 * Interface para proteção de vídeo e áudio contra indexação automática.
 * Processamento 100% no navegador usando FFmpeg.wasm e Web Audio API.
 */

import React, { useState, useCallback, useRef, useEffect } from 'react'
import {
  Video,
  Music,
  Upload,
  Shield,
  Download,
  Play,
  Pause,
  Settings,
  AlertCircle,
  CheckCircle,
  Loader2,
  Eye,
  EyeOff,
  Volume2,
  VolumeX,
  FileVideo,
  FileAudio,
  Trash2,
  Info
} from 'lucide-react'
import { videoProcessor } from '../lib/videoProcessor'
import { audioProcessor } from '../lib/audioProcessor'

// Tipos de mídia suportados
const SUPPORTED_VIDEO = ['.mp4', '.webm', '.mov', '.avi']
const SUPPORTED_AUDIO = ['.mp3', '.wav', '.ogg', '.m4a']

function MediaProtection() {
  // Estado do arquivo
  const [file, setFile] = useState(null)
  const [fileType, setFileType] = useState(null) // 'video' ou 'audio'
  const [fileInfo, setFileInfo] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)

  // Estado de processamento
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState({ type: '', percent: 0, message: '' })
  const [error, setError] = useState(null)

  // Resultado
  const [processedBlob, setProcessedBlob] = useState(null)
  const [processedUrl, setProcessedUrl] = useState(null)

  // Configurações
  const [visualCamouflage, setVisualCamouflage] = useState(5)
  const [audioProtection, setAudioProtection] = useState(true)
  const [stripMetadata, setStripMetadata] = useState(true)
  const [temporalJitter, setTemporalJitter] = useState(true)

  // Player
  const [isPlaying, setIsPlaying] = useState(false)
  const [showOriginal, setShowOriginal] = useState(true)
  const videoRef = useRef(null)
  const audioRef = useRef(null)
  const fileInputRef = useRef(null)

  // FFmpeg loading state
  const [ffmpegLoaded, setFfmpegLoaded] = useState(false)
  const [ffmpegLoading, setFfmpegLoading] = useState(false)

  // Carrega FFmpeg ao montar
  useEffect(() => {
    return () => {
      // Cleanup
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      if (processedUrl) URL.revokeObjectURL(processedUrl)
    }
  }, [])

  /**
   * Handler para upload de arquivo
   */
  const handleFileSelect = useCallback(async (e) => {
    const selectedFile = e.target.files?.[0]
    if (!selectedFile) return

    setError(null)
    setProcessedBlob(null)
    setProcessedUrl(null)

    const ext = '.' + selectedFile.name.split('.').pop().toLowerCase()

    if (SUPPORTED_VIDEO.includes(ext)) {
      setFileType('video')
      setFile(selectedFile)

      // Cria preview
      const url = URL.createObjectURL(selectedFile)
      setPreviewUrl(url)

      // Obtém info do vídeo
      try {
        const info = await videoProcessor.getVideoInfo(selectedFile)
        setFileInfo({
          ...info,
          name: selectedFile.name,
          type: 'video'
        })
      } catch (err) {
        console.error('Erro ao obter info:', err)
      }

    } else if (SUPPORTED_AUDIO.includes(ext)) {
      setFileType('audio')
      setFile(selectedFile)

      // Cria preview
      const url = URL.createObjectURL(selectedFile)
      setPreviewUrl(url)

      // Obtém info do áudio
      try {
        const info = await audioProcessor.getAudioInfo(selectedFile)
        setFileInfo({
          ...info,
          name: selectedFile.name,
          type: 'audio'
        })
      } catch (err) {
        console.error('Erro ao obter info:', err)
      }

    } else {
      setError(`Formato não suportado. Use: ${[...SUPPORTED_VIDEO, ...SUPPORTED_AUDIO].join(', ')}`)
      return
    }
  }, [])

  /**
   * Processa o arquivo
   */
  const handleProcess = useCallback(async () => {
    if (!file) return

    setIsProcessing(true)
    setError(null)
    setProgress({ type: 'starting', percent: 0, message: 'Iniciando...' })

    try {
      let result

      if (fileType === 'video') {
        // Carrega FFmpeg se necessário
        if (!ffmpegLoaded) {
          setFfmpegLoading(true)
          await videoProcessor.load((p) => setProgress(p))
          setFfmpegLoaded(true)
          setFfmpegLoading(false)
        }

        // Processa vídeo
        videoProcessor.onProgress = setProgress
        result = await videoProcessor.processVideo(file, {
          noiseLevel: visualCamouflage,
          stripMetadata,
          temporalJitter,
          audioProtection
        })

      } else if (fileType === 'audio') {
        // Processa áudio
        audioProcessor.onProgress = setProgress
        result = await audioProcessor.processAudio(file, {
          protectionLevel: visualCamouflage,
          addPsychoacousticNoise: audioProtection,
          saturateFrequencies: true,
          addMicroJitter: true
        })
      }

      // Cria URL para preview do resultado
      const resultUrl = URL.createObjectURL(result)
      setProcessedBlob(result)
      setProcessedUrl(resultUrl)
      setShowOriginal(false)

      setProgress({ type: 'complete', percent: 100, message: 'Processamento concluído!' })

    } catch (err) {
      console.error('Erro no processamento:', err)
      let errorMessage = 'Erro ao processar arquivo'

      if (err.message?.includes('Timeout')) {
        errorMessage = 'O processamento demorou demais. Tente um vídeo menor ou reduza a qualidade.'
      } else if (err.message?.includes('SharedArrayBuffer')) {
        errorMessage = 'Seu navegador não suporta processamento de vídeo. Tente usar Chrome ou Firefox.'
      } else if (err.message?.includes('load')) {
        errorMessage = 'Erro ao carregar o motor de processamento. Verifique sua conexão com a internet.'
      } else if (err.message) {
        errorMessage = err.message
      }

      setError(errorMessage)
      setProgress({ type: 'error', percent: 0, message: 'Erro!' })
    } finally {
      setIsProcessing(false)
    }
  }, [file, fileType, visualCamouflage, stripMetadata, temporalJitter, audioProtection, ffmpegLoaded])

  /**
   * Download do arquivo processado
   */
  const handleDownload = useCallback(() => {
    if (!processedBlob) return

    const ext = fileType === 'video' ? 'mp4' : 'wav'
    const filename = `protected_${Date.now()}.${ext}`

    const link = document.createElement('a')
    link.href = processedUrl
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }, [processedBlob, processedUrl, fileType])

  /**
   * Limpa o arquivo atual
   */
  const handleClear = useCallback(() => {
    setFile(null)
    setFileType(null)
    setFileInfo(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    if (processedUrl) URL.revokeObjectURL(processedUrl)
    setPreviewUrl(null)
    setProcessedBlob(null)
    setProcessedUrl(null)
    setError(null)
    setProgress({ type: '', percent: 0, message: '' })
    if (fileInputRef.current) fileInputRef.current.value = ''
  }, [previewUrl, processedUrl])

  /**
   * Toggle play/pause
   */
  const togglePlay = useCallback(() => {
    const media = fileType === 'video' ? videoRef.current : audioRef.current
    if (!media) return

    if (isPlaying) {
      media.pause()
    } else {
      media.play()
    }
    setIsPlaying(!isPlaying)
  }, [isPlaying, fileType])

  /**
   * Formata bytes para exibição
   */
  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  /**
   * Formata duração
   */
  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 rounded-lg p-6 border border-purple-700/50">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-2 bg-purple-600 rounded-lg">
            <Shield className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-xl font-bold text-white">Proteção de Mídia</h2>
        </div>
        <p className="text-gray-300 text-sm">
          Proteja seus vídeos e áudios contra indexação automática por bots e IA.
          Processamento 100% no seu navegador - seus arquivos nunca saem do seu dispositivo.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Coluna Esquerda - Upload e Configurações */}
        <div className="space-y-4">
          {/* Upload */}
          <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
            <h3 className="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
              <Upload className="w-4 h-4" />
              Upload de Mídia
            </h3>

            <div
              className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer
                         transition-colors ${file
                  ? 'border-purple-500 bg-purple-900/20'
                  : 'border-gray-600 hover:border-purple-500 hover:bg-gray-800/50'
                }`}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept={[...SUPPORTED_VIDEO, ...SUPPORTED_AUDIO].join(',')}
                onChange={handleFileSelect}
                className="hidden"
              />

              {file ? (
                <div className="space-y-2">
                  {fileType === 'video' ? (
                    <FileVideo className="w-10 h-10 mx-auto text-purple-400" />
                  ) : (
                    <FileAudio className="w-10 h-10 mx-auto text-pink-400" />
                  )}
                  <p className="text-sm text-white font-medium truncate">{file.name}</p>
                  <p className="text-xs text-gray-400">
                    {fileInfo && `${formatBytes(fileInfo.size)} • ${formatDuration(fileInfo.duration)}`}
                  </p>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleClear() }}
                    className="text-xs text-red-400 hover:text-red-300 flex items-center gap-1 mx-auto"
                  >
                    <Trash2 className="w-3 h-3" /> Remover
                  </button>
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="flex justify-center gap-2">
                    <Video className="w-8 h-8 text-gray-500" />
                    <Music className="w-8 h-8 text-gray-500" />
                  </div>
                  <p className="text-sm text-gray-400">
                    Clique para selecionar ou arraste
                  </p>
                  <p className="text-xs text-gray-500">
                    MP4, WebM, MOV, MP3, WAV
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Configurações */}
          <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
            <h3 className="text-sm font-semibold text-gray-300 mb-4 flex items-center gap-2">
              <Settings className="w-4 h-4" />
              Configurações
            </h3>

            <div className="space-y-4">
              {/* Nível de Camuflagem Visual */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <label className="text-sm text-gray-300 flex items-center gap-2">
                    <Eye className="w-4 h-4 text-yellow-400" />
                    Camuflagem Visual
                  </label>
                  <span className="text-xs px-2 py-0.5 bg-yellow-900/30 text-yellow-400 rounded">
                    Nível {visualCamouflage}
                  </span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={visualCamouflage}
                  onChange={(e) => setVisualCamouflage(parseInt(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg cursor-pointer accent-yellow-500"
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>Sutil</span>
                  <span>Forte</span>
                </div>
                <p className="text-xs text-gray-500">
                  Intensidade do ruído visual que altera o hash do arquivo.
                </p>
              </div>

              {/* Toggle - Proteção de Áudio */}
              <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                <div className="flex items-center gap-2">
                  {audioProtection ? (
                    <Volume2 className="w-4 h-4 text-green-400" />
                  ) : (
                    <VolumeX className="w-4 h-4 text-gray-500" />
                  )}
                  <span className="text-sm text-gray-300">Proteger Áudio</span>
                </div>
                <button
                  onClick={() => setAudioProtection(!audioProtection)}
                  className={`w-12 h-6 rounded-full transition-colors ${audioProtection ? 'bg-green-600' : 'bg-gray-600'
                    }`}
                >
                  <div className={`w-5 h-5 bg-white rounded-full transition-transform ${audioProtection ? 'translate-x-6' : 'translate-x-0.5'
                    }`} />
                </button>
              </div>

              {/* Toggle - Remover Metadados */}
              <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                <div className="flex items-center gap-2">
                  <Shield className="w-4 h-4 text-purple-400" />
                  <span className="text-sm text-gray-300">Remover Metadados</span>
                </div>
                <button
                  onClick={() => setStripMetadata(!stripMetadata)}
                  className={`w-12 h-6 rounded-full transition-colors ${stripMetadata ? 'bg-purple-600' : 'bg-gray-600'
                    }`}
                >
                  <div className={`w-5 h-5 bg-white rounded-full transition-transform ${stripMetadata ? 'translate-x-6' : 'translate-x-0.5'
                    }`} />
                </button>
              </div>

              {/* Toggle - Jitter Temporal */}
              {fileType === 'video' && (
                <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                  <div className="flex items-center gap-2">
                    <Play className="w-4 h-4 text-cyan-400" />
                    <span className="text-sm text-gray-300">Jitter Temporal</span>
                  </div>
                  <button
                    onClick={() => setTemporalJitter(!temporalJitter)}
                    className={`w-12 h-6 rounded-full transition-colors ${temporalJitter ? 'bg-cyan-600' : 'bg-gray-600'
                      }`}
                  >
                    <div className={`w-5 h-5 bg-white rounded-full transition-transform ${temporalJitter ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                  </button>
                </div>
              )}
            </div>

            {/* Botão de Processar */}
            <button
              onClick={handleProcess}
              disabled={!file || isProcessing}
              className={`w-full mt-4 py-3 rounded-lg font-semibold flex items-center justify-center gap-2
                         transition-all ${!file || isProcessing
                  ? 'bg-gray-700 text-gray-500 cursor-not-allowed'
                  : 'bg-gradient-to-r from-purple-600 to-pink-600 text-white hover:from-purple-500 hover:to-pink-500'
                }`}
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Processando...
                </>
              ) : (
                <>
                  <Shield className="w-5 h-5" />
                  Proteger Mídia
                </>
              )}
            </button>
          </div>

          {/* Info Box */}
          <div className="bg-blue-900/20 rounded-lg border border-blue-700/50 p-4">
            <div className="flex items-start gap-2">
              <Info className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
              <div className="text-xs text-gray-400">
                <p className="text-blue-300 font-medium mb-1">Como funciona:</p>
                <ul className="space-y-1 list-disc list-inside">
                  <li>Ruído dinâmico altera o hash do arquivo</li>
                  <li>Metadados EXIF são removidos</li>
                  <li>Jitter temporal confunde detecção de movimento</li>
                  <li>Ruído psicoacústico interfere com transcrição</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Coluna Central e Direita - Preview */}
        <div className="lg:col-span-2 space-y-4">
          {/* Barra de Progresso */}
          {isProcessing && (
            <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-300">{progress.message}</span>
                <span className="text-sm text-purple-400">{progress.percent}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-purple-600 to-pink-600 transition-all duration-300"
                  style={{ width: `${progress.percent}%` }}
                />
              </div>
              {ffmpegLoading && (
                <p className="text-xs text-gray-500 mt-2">
                  Carregando motor de processamento (primeira vez pode demorar)...
                </p>
              )}
            </div>
          )}

          {/* Erro */}
          {error && (
            <div className="bg-red-900/20 rounded-lg border border-red-700/50 p-4 flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
              <p className="text-sm text-red-300">{error}</p>
            </div>
          )}

          {/* Sucesso */}
          {processedBlob && !isProcessing && (
            <div className="bg-green-900/20 rounded-lg border border-green-700/50 p-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <p className="text-sm text-green-300">
                  Mídia protegida com sucesso! ({formatBytes(processedBlob.size)})
                </p>
              </div>
              <button
                onClick={handleDownload}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-500
                           rounded-lg text-white text-sm font-medium transition-colors"
              >
                <Download className="w-4 h-4" />
                Baixar
              </button>
            </div>
          )}

          {/* Player de Preview */}
          {file && (
            <div className="bg-lab-dark rounded-lg border border-gray-700 overflow-hidden">
              {/* Toggle Original/Processado */}
              {processedUrl && (
                <div className="flex border-b border-gray-700">
                  <button
                    onClick={() => setShowOriginal(true)}
                    className={`flex-1 py-2 text-sm font-medium transition-colors ${showOriginal
                        ? 'bg-gray-700 text-white'
                        : 'text-gray-400 hover:text-white'
                      }`}
                  >
                    Original
                  </button>
                  <button
                    onClick={() => setShowOriginal(false)}
                    className={`flex-1 py-2 text-sm font-medium transition-colors ${!showOriginal
                        ? 'bg-purple-600 text-white'
                        : 'text-gray-400 hover:text-white'
                      }`}
                  >
                    Protegido
                  </button>
                </div>
              )}

              {/* Media Player */}
              <div className="aspect-video bg-black flex items-center justify-center">
                {fileType === 'video' ? (
                  <video
                    ref={videoRef}
                    src={showOriginal ? previewUrl : (processedUrl || previewUrl)}
                    className="max-w-full max-h-full"
                    controls
                    onPlay={() => setIsPlaying(true)}
                    onPause={() => setIsPlaying(false)}
                  />
                ) : (
                  <div className="text-center p-8">
                    <div className="w-32 h-32 mx-auto mb-4 bg-gradient-to-br from-purple-600 to-pink-600 rounded-full flex items-center justify-center">
                      <Music className="w-16 h-16 text-white" />
                    </div>
                    <audio
                      ref={audioRef}
                      src={showOriginal ? previewUrl : (processedUrl || previewUrl)}
                      className="w-full max-w-md mx-auto"
                      controls
                      onPlay={() => setIsPlaying(true)}
                      onPause={() => setIsPlaying(false)}
                    />
                  </div>
                )}
              </div>

              {/* Info do arquivo */}
              {fileInfo && (
                <div className="p-3 bg-gray-800/50 border-t border-gray-700">
                  <div className="grid grid-cols-3 gap-4 text-center text-xs">
                    <div>
                      <div className="text-gray-400">Duração</div>
                      <div className="text-white font-medium">{formatDuration(fileInfo.duration)}</div>
                    </div>
                    {fileInfo.type === 'video' && (
                      <div>
                        <div className="text-gray-400">Resolução</div>
                        <div className="text-white font-medium">{fileInfo.width}x{fileInfo.height}</div>
                      </div>
                    )}
                    {fileInfo.type === 'audio' && (
                      <div>
                        <div className="text-gray-400">Sample Rate</div>
                        <div className="text-white font-medium">{fileInfo.sampleRate} Hz</div>
                      </div>
                    )}
                    <div>
                      <div className="text-gray-400">Tamanho</div>
                      <div className="text-white font-medium">{formatBytes(fileInfo.size)}</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Placeholder quando não há arquivo */}
          {!file && (
            <div className="bg-lab-dark rounded-lg border border-gray-700 aspect-video flex items-center justify-center">
              <div className="text-center text-gray-500">
                <div className="flex justify-center gap-4 mb-4">
                  <Video className="w-12 h-12" />
                  <Music className="w-12 h-12" />
                </div>
                <p className="text-sm">Selecione um arquivo para visualizar</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default MediaProtection
