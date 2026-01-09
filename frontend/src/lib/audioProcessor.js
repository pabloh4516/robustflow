/**
 * Audio Processor v2.1 - Motor de Proteção de Áudio OTIMIZADO
 * ==========================================================
 *
 * Utiliza Web Audio API para processamento 100% no navegador.
 *
 * OTIMIZAÇÕES v2.1 (Preservação de Música):
 * - High-Shelf BOOST acima de 4kHz para preservar brilho da música
 * - Ruído rosa focado APENAS em 500Hz-3000Hz (faixa de fala)
 * - Compressão reduzida (ratio 2:1) para evitar dropouts
 * - Threshold de compressor ajustado para dinâmica natural
 * - Preservação de transientes musicais
 */

class AudioProcessor {
  constructor() {
    this.audioContext = null
    this.onProgress = null
  }

  /**
   * Inicializa o contexto de áudio
   */
  async init() {
    if (!this.audioContext) {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)()
    }
    return this.audioContext
  }

  /**
   * Processa arquivo de áudio com proteção anti-transcrição OTIMIZADA
   *
   * @param {File} audioFile - Arquivo de áudio (MP3, WAV, etc.)
   * @param {Object} options - Opções de processamento
   * @returns {Blob} - Áudio processado
   */
  async processAudio(audioFile, options = {}) {
    const {
      protectionLevel = 5,           // 1-10: Intensidade da proteção
      addPsychoacousticNoise = true, // Ruído rosa calibrado
      pulsatingNoise = true,         // Ruído pulsante (novo)
      surgicalNotch = true,          // Notch cirúrgico em fricativas
      preserveVowels = true,         // Preservar clareza de vogais
      preserveMusic = true,          // NOVO v2.1: Preservar brilho musical
      outputFormat = 'wav'
    } = options

    await this.init()

    if (this.onProgress) this.onProgress({ type: 'loading', percent: 10, message: 'Carregando áudio...' })

    // Decodifica o arquivo de áudio
    const arrayBuffer = await audioFile.arrayBuffer()
    const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer)

    if (this.onProgress) this.onProgress({ type: 'processing', percent: 30, message: 'Aplicando proteção otimizada...' })

    // Cria buffer offline para processamento
    const offlineContext = new OfflineAudioContext(
      audioBuffer.numberOfChannels,
      audioBuffer.length,
      audioBuffer.sampleRate
    )

    // Cria source node
    const source = offlineContext.createBufferSource()
    source.buffer = audioBuffer

    // Cadeia de processamento OTIMIZADA v2.1
    let currentNode = source

    // 1. Ruído Psicoacústico FOCADO (500Hz-3000Hz apenas)
    if (addPsychoacousticNoise) {
      currentNode = this._addFocusedPinkNoise(offlineContext, currentNode, protectionLevel, pulsatingNoise)
    }

    // 2. Notch Cirúrgico em Fricativas (2500Hz e 3500Hz)
    if (surgicalNotch) {
      currentNode = this._addSurgicalNotchFilters(offlineContext, currentNode, protectionLevel)
    }

    // 3. Preservação de Vogais (boost sutil em 300-800Hz)
    if (preserveVowels) {
      currentNode = this._addVowelPreservation(offlineContext, currentNode)
    }

    // 4. NOVO v2.1: Preservação de Brilho Musical (High-Shelf acima de 4kHz)
    if (preserveMusic) {
      currentNode = this._addMusicBrightnessPreservation(offlineContext, currentNode)
    }

    // 5. Micro-modulações sutis (anti-STT) com compressão SUAVE
    currentNode = this._addSubtleMicroModulation(offlineContext, currentNode, protectionLevel)

    // Conecta ao destino
    currentNode.connect(offlineContext.destination)

    if (this.onProgress) this.onProgress({ type: 'rendering', percent: 60, message: 'Renderizando...' })

    // Inicia a source e renderiza
    source.start(0)
    const renderedBuffer = await offlineContext.startRendering()

    if (this.onProgress) this.onProgress({ type: 'encoding', percent: 80, message: 'Codificando...' })

    // Converte para o formato de saída
    const outputBlob = await this._bufferToBlob(renderedBuffer, outputFormat)

    if (this.onProgress) this.onProgress({ type: 'complete', percent: 100, message: 'Concluído!' })

    return outputBlob
  }

  /**
   * Adiciona ruído rosa FOCADO na faixa de fala (500Hz-3000Hz)
   *
   * OTIMIZAÇÕES v2.1:
   * - Faixa de frequência mais estreita: 500Hz-3000Hz
   * - NÃO afeta frequências acima de 3kHz (preserva música)
   * - Modulação LFO menos agressiva
   */
  _addFocusedPinkNoise(context, inputNode, level, pulsating = true) {
    // ===========================================
    // CÁLCULO DE GANHO CALIBRADO
    // ===========================================
    const minDB = -45  // Nível mínimo (level 1)
    const maxDB = -35  // Nível máximo (level 10)
    const targetDB = minDB + ((maxDB - minDB) * (level / 10))
    const linearGain = Math.pow(10, targetDB / 20)

    console.log(`[AudioProcessor v2.1] Ruído rosa focado: ${targetDB.toFixed(1)}dB (500-3000Hz)`)

    // ===========================================
    // GERAÇÃO DE RUÍDO ROSA (Paul Kellet Algorithm)
    // ===========================================
    const bufferSize = context.sampleRate * 4
    const noiseBuffer = context.createBuffer(1, bufferSize, context.sampleRate)
    const noiseData = noiseBuffer.getChannelData(0)

    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0
    for (let i = 0; i < bufferSize; i++) {
      const white = Math.random() * 2 - 1
      b0 = 0.99886 * b0 + white * 0.0555179
      b1 = 0.99332 * b1 + white * 0.0750759
      b2 = 0.96900 * b2 + white * 0.1538520
      b3 = 0.86650 * b3 + white * 0.3104856
      b4 = 0.55000 * b4 + white * 0.5329522
      b5 = -0.7616 * b5 - white * 0.0168980
      noiseData[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.11
      b6 = white * 0.115926
    }

    const noiseSource = context.createBufferSource()
    noiseSource.buffer = noiseBuffer
    noiseSource.loop = true
    noiseSource.start(0)

    // ===========================================
    // FILTRO BANDPASS ESTREITO (500Hz-3000Hz)
    // ===========================================
    // Centro em 1225Hz (média geométrica de 500 e 3000)
    // Q calculado para cobrir apenas essa faixa
    const bandpass = context.createBiquadFilter()
    bandpass.type = 'bandpass'
    bandpass.frequency.value = 1225  // sqrt(500 * 3000) ≈ 1225Hz
    bandpass.Q.value = 0.5  // Q baixo = banda mais larga mas controlada

    // Lowpass para cortar acima de 3kHz (protege música)
    const lowpass = context.createBiquadFilter()
    lowpass.type = 'lowpass'
    lowpass.frequency.value = 3000
    lowpass.Q.value = 0.7

    // Highpass para cortar abaixo de 500Hz (protege graves)
    const highpass = context.createBiquadFilter()
    highpass.type = 'highpass'
    highpass.frequency.value = 500
    highpass.Q.value = 0.7

    noiseSource.connect(bandpass)
    bandpass.connect(lowpass)
    lowpass.connect(highpass)

    // ===========================================
    // RUÍDO PULSANTE SUAVE
    // ===========================================
    const noiseGain = context.createGain()
    noiseGain.gain.value = linearGain

    if (pulsating) {
      // LFO mais suave para não causar dropouts
      const pulseLFO = context.createOscillator()
      const pulseDepth = context.createGain()

      pulseLFO.type = 'sine'
      pulseLFO.frequency.value = 3 + (level * 0.3)  // 3-6 Hz (mais lento)
      pulseDepth.gain.value = linearGain * 0.3  // Modulação reduzida para 30%

      pulseLFO.connect(pulseDepth)
      pulseDepth.connect(noiseGain.gain)
      pulseLFO.start(0)
    }

    highpass.connect(noiseGain)

    // ===========================================
    // MIXER FINAL (sem segundo canal de ruído)
    // ===========================================
    const merger = context.createGain()
    merger.gain.value = 1.0

    inputNode.connect(merger)
    noiseGain.connect(merger)

    return merger
  }

  /**
   * Filtros Notch CIRÚRGICOS para fricativas
   * (Mantido de v2.0 - funciona bem)
   */
  _addSurgicalNotchFilters(context, inputNode, level) {
    // NOTCH 1: 2500Hz (Fricativas como 's', 'f')
    const notch2500 = context.createBiquadFilter()
    notch2500.type = 'notch'
    notch2500.frequency.value = 2500
    notch2500.Q.value = 15 + (level * 2)

    // NOTCH 2: 3500Hz (Fricativas como 'sh', 'ch')
    const notch3500 = context.createBiquadFilter()
    notch3500.type = 'notch'
    notch3500.frequency.value = 3500
    notch3500.Q.value = 15 + (level * 2)

    // Atenuação MUITO SUAVE de sibilantes (apenas -1 a -3dB)
    const sibilantShelf = context.createBiquadFilter()
    sibilantShelf.type = 'highshelf'
    sibilantShelf.frequency.value = 5000
    sibilantShelf.gain.value = -1 - (level * 0.2)  // -1dB a -3dB (muito mais suave)

    inputNode.connect(notch2500)
    notch2500.connect(notch3500)
    notch3500.connect(sibilantShelf)

    return sibilantShelf
  }

  /**
   * Preservação de Vogais (mantido de v2.0)
   */
  _addVowelPreservation(context, inputNode) {
    const vowelBoost = context.createBiquadFilter()
    vowelBoost.type = 'peaking'
    vowelBoost.frequency.value = 500
    vowelBoost.Q.value = 1.0
    vowelBoost.gain.value = 2

    const presenceBoost = context.createBiquadFilter()
    presenceBoost.type = 'peaking'
    presenceBoost.frequency.value = 1200
    presenceBoost.Q.value = 1.5
    presenceBoost.gain.value = 1.5

    inputNode.connect(vowelBoost)
    vowelBoost.connect(presenceBoost)

    return presenceBoost
  }

  /**
   * NOVO v2.1: Preservação de Brilho Musical
   *
   * High-Shelf BOOST acima de 4kHz para:
   * - Manter clareza de instrumentos (pratos, cordas)
   * - Preservar brilho de vocais femininos
   * - Compensar qualquer perda nas altas frequências
   */
  _addMusicBrightnessPreservation(context, inputNode) {
    // ===========================================
    // HIGH-SHELF BOOST (4kHz+)
    // ===========================================
    // Compensa a atenuação do filtro de sibilantes
    const brightnessBoost = context.createBiquadFilter()
    brightnessBoost.type = 'highshelf'
    brightnessBoost.frequency.value = 4000
    brightnessBoost.gain.value = 2  // +2dB boost acima de 4kHz

    // ===========================================
    // AIR BAND (10kHz-16kHz)
    // ===========================================
    // Preserva "ar" e espacialidade da música
    const airBand = context.createBiquadFilter()
    airBand.type = 'highshelf'
    airBand.frequency.value = 10000
    airBand.gain.value = 1.5  // +1.5dB de "ar"

    // ===========================================
    // PROTEÇÃO DE SUB-GRAVES (20-80Hz)
    // ===========================================
    // Garante que graves permaneçam intactos
    const subBassBoost = context.createBiquadFilter()
    subBassBoost.type = 'lowshelf'
    subBassBoost.frequency.value = 80
    subBassBoost.gain.value = 1  // +1dB sutil nos graves

    inputNode.connect(brightnessBoost)
    brightnessBoost.connect(airBand)
    airBand.connect(subBassBoost)

    return subBassBoost
  }

  /**
   * Micro-modulações SUTIS para anti-STT
   *
   * OTIMIZAÇÕES v2.1:
   * - Ratio reduzido para 2:1 (era 3:1) - evita dropouts
   * - Threshold mais alto (-15dB) - preserva dinâmica
   * - Attack mais lento (50ms) - preserva transientes
   * - Release mais rápido (150ms) - recuperação natural
   */
  _addSubtleMicroModulation(context, inputNode, level) {
    // ===========================================
    // TREMOLO MUITO SUTIL
    // ===========================================
    const tremoloGain = context.createGain()
    tremoloGain.gain.value = 1.0

    const tremoloLFO = context.createOscillator()
    const tremoloDepth = context.createGain()

    tremoloLFO.type = 'sine'
    tremoloLFO.frequency.value = 5 + (level * 0.3)
    tremoloDepth.gain.value = 0.01 + (level * 0.001)  // Ainda mais sutil

    tremoloLFO.connect(tremoloDepth)
    tremoloDepth.connect(tremoloGain.gain)
    tremoloLFO.start(0)

    // ===========================================
    // COMPRESSOR SUAVE v2.1 (Ratio 2:1)
    // ===========================================
    const compressor = context.createDynamicsCompressor()
    compressor.threshold.value = -15  // Mais alto (era -20)
    compressor.knee.value = 30        // Knee muito suave
    compressor.ratio.value = 2        // RATIO 2:1 (era 3:1)
    compressor.attack.value = 0.05    // 50ms (preserva transientes)
    compressor.release.value = 0.15   // 150ms (recuperação rápida)

    // ===========================================
    // LIMITER DE SEGURANÇA (apenas para picos)
    // ===========================================
    const limiter = context.createDynamicsCompressor()
    limiter.threshold.value = -1      // Apenas picos extremos
    limiter.knee.value = 0
    limiter.ratio.value = 20
    limiter.attack.value = 0.001
    limiter.release.value = 0.05      // Release mais rápido

    // ===========================================
    // MAKEUP GAIN (compensa compressão)
    // ===========================================
    const makeupGain = context.createGain()
    makeupGain.gain.value = 1.1  // +0.8dB de makeup

    inputNode.connect(tremoloGain)
    tremoloGain.connect(compressor)
    compressor.connect(makeupGain)
    makeupGain.connect(limiter)

    return limiter
  }

  /**
   * Converte AudioBuffer para Blob
   */
  async _bufferToBlob(audioBuffer, format = 'wav') {
    if (format === 'wav') {
      const wavBuffer = this._encodeWAV(audioBuffer)
      return new Blob([wavBuffer], { type: 'audio/wav' })
    } else {
      const wavBuffer = this._encodeWAV(audioBuffer)
      return new Blob([wavBuffer], { type: 'audio/wav' })
    }
  }

  /**
   * Codifica AudioBuffer como WAV
   */
  _encodeWAV(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels
    const sampleRate = audioBuffer.sampleRate
    const format = 1 // PCM
    const bitDepth = 16

    const bytesPerSample = bitDepth / 8
    const blockAlign = numChannels * bytesPerSample

    const length = audioBuffer.length
    const data = new Float32Array(length * numChannels)

    for (let channel = 0; channel < numChannels; channel++) {
      const channelData = audioBuffer.getChannelData(channel)
      for (let i = 0; i < length; i++) {
        data[i * numChannels + channel] = channelData[i]
      }
    }

    const buffer = new ArrayBuffer(44 + data.length * bytesPerSample)
    const view = new DataView(buffer)

    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i))
      }
    }

    writeString(0, 'RIFF')
    view.setUint32(4, 36 + data.length * bytesPerSample, true)
    writeString(8, 'WAVE')
    writeString(12, 'fmt ')
    view.setUint32(16, 16, true)
    view.setUint16(20, format, true)
    view.setUint16(22, numChannels, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * blockAlign, true)
    view.setUint16(32, blockAlign, true)
    view.setUint16(34, bitDepth, true)
    writeString(36, 'data')
    view.setUint32(40, data.length * bytesPerSample, true)

    let offset = 44
    for (let i = 0; i < data.length; i++) {
      const sample = Math.max(-1, Math.min(1, data[i]))
      const intSample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF
      view.setInt16(offset, intSample, true)
      offset += 2
    }

    return buffer
  }

  /**
   * Obtém informações do arquivo de áudio
   */
  async getAudioInfo(audioFile) {
    await this.init()

    const arrayBuffer = await audioFile.arrayBuffer()
    const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer)

    return {
      duration: audioBuffer.duration,
      sampleRate: audioBuffer.sampleRate,
      numberOfChannels: audioBuffer.numberOfChannels,
      size: audioFile.size
    }
  }

  /**
   * Libera recursos
   */
  terminate() {
    if (this.audioContext) {
      this.audioContext.close()
      this.audioContext = null
    }
  }
}

// Exporta instância singleton
export const audioProcessor = new AudioProcessor()
export default AudioProcessor
