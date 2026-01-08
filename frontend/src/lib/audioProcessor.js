/**
 * Audio Processor - Motor de Proteção de Áudio
 * =============================================
 *
 * Utiliza Web Audio API para processamento 100% no navegador.
 *
 * Técnicas implementadas:
 * 1. Ruído Psicoacústico - Interfere com Speech-to-Text
 * 2. Saturação de Frequências - Confunde reconhecimento de fonemas
 * 3. Micro-variações Temporais - Altera timing de sílabas
 * 4. Mascaramento Espectral - Oculta padrões de fala
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
   * Processa arquivo de áudio com proteção anti-transcrição
   *
   * @param {File} audioFile - Arquivo de áudio (MP3, WAV, etc.)
   * @param {Object} options - Opções de processamento
   * @returns {Blob} - Áudio processado
   */
  async processAudio(audioFile, options = {}) {
    const {
      protectionLevel = 5,      // 1-10: Intensidade da proteção
      addPsychoacousticNoise = true,
      saturateFrequencies = true,
      addMicroJitter = true,
      outputFormat = 'wav'
    } = options

    await this.init()

    if (this.onProgress) this.onProgress({ type: 'loading', percent: 10, message: 'Carregando áudio...' })

    // Decodifica o arquivo de áudio
    const arrayBuffer = await audioFile.arrayBuffer()
    const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer)

    if (this.onProgress) this.onProgress({ type: 'processing', percent: 30, message: 'Aplicando proteção...' })

    // Cria buffer offline para processamento
    const offlineContext = new OfflineAudioContext(
      audioBuffer.numberOfChannels,
      audioBuffer.length,
      audioBuffer.sampleRate
    )

    // Cria source node
    const source = offlineContext.createBufferSource()
    source.buffer = audioBuffer

    // Cadeia de processamento
    let currentNode = source

    // 1. Ruído Psicoacústico
    if (addPsychoacousticNoise) {
      currentNode = this._addPsychoacousticNoise(offlineContext, currentNode, protectionLevel)
    }

    // 2. Saturação de Frequências de Fala
    if (saturateFrequencies) {
      currentNode = this._addFrequencySaturation(offlineContext, currentNode, protectionLevel)
    }

    // 3. Micro-variações (simuladas via modulação)
    if (addMicroJitter) {
      currentNode = this._addMicroModulation(offlineContext, currentNode, protectionLevel)
    }

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
   * Adiciona ruído psicoacústico que interfere com Speech-to-Text
   * Frequências escolhidas para serem imperceptíveis mas efetivas
   */
  _addPsychoacousticNoise(context, inputNode, level) {
    // Cria gerador de ruído
    const noiseGain = context.createGain()
    const noiseIntensity = 0.005 + (level * 0.003) // 0.008 a 0.035

    // Gera ruído rosa (mais natural que ruído branco)
    const bufferSize = context.sampleRate * 2
    const noiseBuffer = context.createBuffer(1, bufferSize, context.sampleRate)
    const noiseData = noiseBuffer.getChannelData(0)

    // Geração de ruído rosa usando algoritmo Paul Kellet
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

    // Filtra o ruído para frequências que afetam STT (300-3400 Hz)
    const bandpass = context.createBiquadFilter()
    bandpass.type = 'bandpass'
    bandpass.frequency.value = 1500
    bandpass.Q.value = 0.5

    noiseSource.connect(bandpass)
    bandpass.connect(noiseGain)
    noiseGain.gain.value = noiseIntensity

    // Mixer para combinar sinal original com ruído
    const merger = context.createGain()
    merger.gain.value = 1.0

    inputNode.connect(merger)
    noiseGain.connect(merger)

    return merger
  }

  /**
   * Satura frequências críticas para reconhecimento de fala
   * Faixa de frequência de fala: 300-3400 Hz
   */
  _addFrequencySaturation(context, inputNode, level) {
    // Equalização que confunde reconhecimento de fonemas

    // Filtro 1: Atenua frequências de fricativas (4-8 kHz)
    const highShelf = context.createBiquadFilter()
    highShelf.type = 'highshelf'
    highShelf.frequency.value = 5000
    highShelf.gain.value = -2 - (level * 0.5)

    // Filtro 2: Boost sutil em frequências de mascaramento (1-2 kHz)
    const peaking1 = context.createBiquadFilter()
    peaking1.type = 'peaking'
    peaking1.frequency.value = 1500
    peaking1.Q.value = 1.0
    peaking1.gain.value = 1 + (level * 0.3)

    // Filtro 3: Atenua frequências de vogais (500-1000 Hz) levemente
    const peaking2 = context.createBiquadFilter()
    peaking2.type = 'peaking'
    peaking2.frequency.value = 700
    peaking2.Q.value = 2.0
    peaking2.gain.value = -1 - (level * 0.2)

    // Filtro 4: Notch em frequência crítica de plosivas
    const notch = context.createBiquadFilter()
    notch.type = 'notch'
    notch.frequency.value = 2500
    notch.Q.value = 10

    // Cadeia de filtros
    inputNode.connect(highShelf)
    highShelf.connect(peaking1)
    peaking1.connect(peaking2)
    peaking2.connect(notch)

    return notch
  }

  /**
   * Adiciona micro-modulações que confundem timing de sílabas
   */
  _addMicroModulation(context, inputNode, level) {
    // Tremolo muito sutil (variação de amplitude)
    const tremoloGain = context.createGain()
    const tremoloOsc = context.createOscillator()
    const tremoloDepth = context.createGain()

    tremoloOsc.type = 'sine'
    tremoloOsc.frequency.value = 0.5 + (level * 0.3) // 0.5-3.5 Hz
    tremoloDepth.gain.value = 0.02 + (level * 0.01) // Profundidade muito sutil

    tremoloOsc.connect(tremoloDepth)
    tremoloDepth.connect(tremoloGain.gain)
    tremoloOsc.start(0)

    // Compressor para normalizar variações
    const compressor = context.createDynamicsCompressor()
    compressor.threshold.value = -24
    compressor.knee.value = 30
    compressor.ratio.value = 4
    compressor.attack.value = 0.003
    compressor.release.value = 0.25

    inputNode.connect(tremoloGain)
    tremoloGain.connect(compressor)

    return compressor
  }

  /**
   * Converte AudioBuffer para Blob
   */
  async _bufferToBlob(audioBuffer, format = 'wav') {
    const numChannels = audioBuffer.numberOfChannels
    const sampleRate = audioBuffer.sampleRate
    const length = audioBuffer.length

    if (format === 'wav') {
      // Encode como WAV
      const wavBuffer = this._encodeWAV(audioBuffer)
      return new Blob([wavBuffer], { type: 'audio/wav' })
    } else {
      // Para MP3, retornamos WAV (conversão MP3 requer biblioteca adicional)
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

    // Interleave channels
    const length = audioBuffer.length
    const data = new Float32Array(length * numChannels)

    for (let channel = 0; channel < numChannels; channel++) {
      const channelData = audioBuffer.getChannelData(channel)
      for (let i = 0; i < length; i++) {
        data[i * numChannels + channel] = channelData[i]
      }
    }

    // Cria buffer WAV
    const buffer = new ArrayBuffer(44 + data.length * bytesPerSample)
    const view = new DataView(buffer)

    // WAV Header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i))
      }
    }

    writeString(0, 'RIFF')
    view.setUint32(4, 36 + data.length * bytesPerSample, true)
    writeString(8, 'WAVE')
    writeString(12, 'fmt ')
    view.setUint32(16, 16, true) // Subchunk1Size
    view.setUint16(20, format, true)
    view.setUint16(22, numChannels, true)
    view.setUint32(24, sampleRate, true)
    view.setUint32(28, sampleRate * blockAlign, true)
    view.setUint16(32, blockAlign, true)
    view.setUint16(34, bitDepth, true)
    writeString(36, 'data')
    view.setUint32(40, data.length * bytesPerSample, true)

    // Write audio data
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
