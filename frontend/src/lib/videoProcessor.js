/**
 * Video Processor v2.1 - Motor de Proteção Anti-OCR Otimizado
 * ============================================================
 *
 * OTIMIZAÇÕES v2.1:
 * - Preset 'veryfast' + CRF 26 = arquivos 5-10MB para 30s
 * - Camuflagem estocástica (scale+crop com jitter)
 * - Ruído calibrado (15) para qualidade visual
 * - Sem gblur pesado = renderização mais rápida
 * - Redução de ruído cromático para economia de bits
 */

import { FFmpeg } from '@ffmpeg/ffmpeg'
import { fetchFile, toBlobURL } from '@ffmpeg/util'

class VideoProcessor {
  constructor() {
    this.ffmpeg = null
    this.loaded = false
    this.loadProgress = 0
    this.onProgress = null
    this.onLog = null
  }

  /**
   * Carrega o FFmpeg.wasm com suporte a múltiplas threads
   */
  async load(onProgress = null) {
    if (this.loaded) return true

    this.onProgress = onProgress
    this.ffmpeg = new FFmpeg()

    this.ffmpeg.on('log', ({ message }) => {
      if (this.onLog) this.onLog(message)
      console.log('[FFmpeg]', message)
    })

    this.ffmpeg.on('progress', ({ progress, time }) => {
      const percent = Math.round(progress * 100)
      if (this.onProgress) {
        this.onProgress({
          type: 'encoding',
          percent,
          time: time / 1000000
        })
      }
    })

    try {
      const baseURL = 'https://unpkg.com/@ffmpeg/core-mt@0.12.6/dist/esm'

      if (onProgress) onProgress({ type: 'loading', percent: 10, message: 'Carregando FFmpeg...' })

      await this.ffmpeg.load({
        coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
        wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
        workerURL: await toBlobURL(`${baseURL}/ffmpeg-core.worker.js`, 'text/javascript'),
      })

      if (onProgress) onProgress({ type: 'loading', percent: 100, message: 'FFmpeg pronto!' })

      this.loaded = true
      return true
    } catch (error) {
      console.error('Erro ao carregar FFmpeg:', error)

      try {
        const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm'
        await this.ffmpeg.load({
          coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
          wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
        })
        this.loaded = true
        return true
      } catch (fallbackError) {
        console.error('Erro no fallback:', fallbackError)
        throw new Error('Não foi possível carregar o processador de vídeo')
      }
    }
  }

  /**
   * Processa vídeo com proteção anti-OCR otimizada
   */
  async processVideo(videoFile, options = {}) {
    if (!this.loaded) {
      await this.load(this.onProgress)
    }

    const {
      noiseLevel = 5,
      stripMetadata = true,
      temporalJitter = true,
      audioProtection = false,
      stochasticCrop = true,      // NOVO: Camuflagem estocástica
      outputFormat = 'mp4'
    } = options

    const inputName = 'input.mp4'
    const outputName = `output.${outputFormat}`

    try {
      if (this.onProgress) this.onProgress({ type: 'preparing', percent: 0, message: 'Preparando vídeo...' })

      await this.ffmpeg.writeFile(inputName, await fetchFile(videoFile))

      // Constrói os filtros de vídeo OTIMIZADOS
      const videoFilters = this._buildOptimizedVideoFilters(noiseLevel, stochasticCrop)

      // Constrói os filtros de áudio
      const audioFilters = audioProtection ? this._buildAudioFilters() : null

      // Constrói o comando FFmpeg OTIMIZADO
      const args = this._buildOptimizedFFmpegArgs({
        inputName,
        outputName,
        videoFilters,
        audioFilters,
        stripMetadata
      })

      if (this.onProgress) this.onProgress({ type: 'encoding', percent: 0, message: 'Otimizando proteção...' })

      const execPromise = this.ffmpeg.exec(args)
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Timeout: processamento demorou mais de 5 minutos')), 300000)
      })
      await Promise.race([execPromise, timeoutPromise])

      const data = await this.ffmpeg.readFile(outputName)

      await this.ffmpeg.deleteFile(inputName)
      await this.ffmpeg.deleteFile(outputName)

      if (this.onProgress) this.onProgress({ type: 'complete', percent: 100, message: 'Proteção aplicada!' })

      return new Blob([data.buffer], { type: `video/${outputFormat}` })
    } catch (error) {
      console.error('Erro ao processar vídeo:', error)
      throw error
    }
  }

  /**
   * Constrói filtros de vídeo OTIMIZADOS v2.1
   *
   * MUDANÇAS:
   * - Ruído reduzido para 15 (qualidade visual)
   * - Sem gblur (economia de processamento)
   * - Camuflagem estocástica via scale+crop
   * - Redução de ruído cromático (economia de bits)
   */
  _buildOptimizedVideoFilters(noiseLevel, stochasticCrop) {
    const filters = []

    // ===========================================
    // 1. CAMUFLAGEM ESTOCÁSTICA (Jitter de Posição)
    // ===========================================
    // Scale up 5% e crop aleatório - impede template matching
    if (stochasticCrop) {
      // Valores pseudo-aleatórios fixos por encode (baseado em timestamp)
      const seed1 = Math.floor(Math.random() * 8) + 2  // 2-10 pixels
      const seed2 = Math.floor(Math.random() * 8) + 2  // 2-10 pixels
      // Scale up 5%, depois crop para tamanho original com offset aleatório
      filters.push(`scale=iw*1.05:-2`)
      filters.push(`crop=iw/1.05:ih/1.05:${seed1}:${seed2}`)
    }

    // ===========================================
    // 2. SCALE PARA 720p (otimização de tamanho)
    // ===========================================
    filters.push(`scale=720:-2`)

    // ===========================================
    // 3. RUÍDO TEMPORAL CALIBRADO (intensidade 15)
    // ===========================================
    // Ruído reduzido para não degradar visualmente
    // Mantém eficácia anti-OCR sem parecer erro
    const noiseStrength = Math.max(10, Math.min(20, 15 + (noiseLevel - 5)))
    filters.push(`noise=alls=${noiseStrength}:allf=t+u`)

    // ===========================================
    // 4. REDUÇÃO DE RUÍDO CROMÁTICO (economia de bits)
    // ===========================================
    // Substitui gblur por hqdn3d leve (denoise)
    // Economiza bits no encoding sem perder proteção
    if (noiseLevel > 3) {
      // hqdn3d: luma_spatial:chroma_spatial:luma_tmp:chroma_tmp
      filters.push(`hqdn3d=2:2:3:3`)
    }

    // ===========================================
    // 5. VARIAÇÃO DE BRILHO/CONTRASTE SUTIL
    // ===========================================
    const brightness = (0.005 + (noiseLevel * 0.002) * (Math.random() * 2 - 1)).toFixed(3)
    const contrast = (1.0 + (noiseLevel * 0.005)).toFixed(3)
    filters.push(`eq=brightness=${brightness}:contrast=${contrast}`)

    // ===========================================
    // 6. UNSHARP LEVE (quebra padrões de texto)
    // ===========================================
    if (noiseLevel > 4) {
      filters.push(`unsharp=3:3:0.5:3:3:0.3`)
    }

    // ===========================================
    // 7. RUÍDO TEMPORAL SECUNDÁRIO (opcional)
    // ===========================================
    // Apenas para níveis altos de proteção
    if (noiseLevel > 7) {
      filters.push(`noise=alls=10:allf=t`)
    }

    return filters.join(',')
  }

  /**
   * Constrói filtros de áudio OTIMIZADOS
   * Mais suave para não degradar música
   */
  _buildAudioFilters() {
    const filters = []

    // 1. Pitch shift muito sutil (2%)
    filters.push(`asetrate=44100*1.02,aresample=44100`)

    // 2. Equalização focada apenas em fricativas
    filters.push(`equalizer=f=2500:t=q:w=3:g=-3`)
    filters.push(`equalizer=f=3500:t=q:w=3:g=-3`)

    // 3. Compressor SUAVE (ratio 2:1 para preservar música)
    filters.push(`acompressor=threshold=-20dB:ratio=2:attack=20:release=200`)

    // 4. Tremolo muito imperceptível
    filters.push(`tremolo=f=5:d=0.05`)

    return filters.join(',')
  }

  /**
   * Constrói argumentos FFmpeg OTIMIZADOS para tamanho
   *
   * Target: 5-10MB para vídeos de 30s
   * - Preset: veryfast (bom balanço velocidade/compressão)
   * - CRF: 26 (qualidade suficiente para anúncios)
   * - Audio: 96kbps AAC
   */
  _buildOptimizedFFmpegArgs({ inputName, outputName, videoFilters, audioFilters, stripMetadata }) {
    const args = ['-i', inputName]

    // Remove metadados
    if (stripMetadata) {
      args.push('-map_metadata', '-1')
    }

    // Aplica filtros de vídeo
    if (videoFilters) {
      args.push('-vf', videoFilters)
    }

    // Aplica filtros de áudio
    if (audioFilters) {
      args.push('-af', audioFilters)
    }

    // =========================================
    // ENCODING OTIMIZADO PARA TAMANHO
    // =========================================
    args.push('-c:v', 'libx264')
    args.push('-preset', 'veryfast')  // Bom balanço velocidade/compressão
    args.push('-crf', '26')           // Qualidade otimizada para tamanho
    args.push('-profile:v', 'main')   // Compatibilidade ampla
    args.push('-level', '4.0')

    // Áudio otimizado
    args.push('-c:a', 'aac')
    args.push('-b:a', '96k')          // 96kbps é suficiente para fala
    args.push('-ar', '44100')         // Sample rate padrão

    // Otimizações adicionais
    args.push('-movflags', '+faststart')  // Streaming otimizado
    args.push('-threads', '2')

    args.push('-y', outputName)

    return args
  }

  /**
   * Extrai informações do vídeo
   */
  async getVideoInfo(videoFile) {
    const video = document.createElement('video')
    video.preload = 'metadata'

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        URL.revokeObjectURL(video.src)
        reject(new Error('Timeout ao carregar metadados do vídeo'))
      }, 10000)

      video.onloadedmetadata = () => {
        clearTimeout(timeout)
        const info = {
          duration: video.duration,
          width: video.videoWidth,
          height: video.videoHeight,
          size: videoFile.size
        }
        URL.revokeObjectURL(video.src)
        resolve(info)
      }

      video.onerror = (e) => {
        clearTimeout(timeout)
        URL.revokeObjectURL(video.src)
        reject(new Error('Erro ao carregar metadados do vídeo'))
      }

      video.src = URL.createObjectURL(videoFile)
    })
  }

  /**
   * Libera recursos
   */
  terminate() {
    if (this.ffmpeg) {
      this.ffmpeg.terminate()
      this.ffmpeg = null
      this.loaded = false
    }
  }
}

// Exporta instância singleton
export const videoProcessor = new VideoProcessor()
export default VideoProcessor
