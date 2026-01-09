/**
 * BotEye Processor v2.0 - Simulador de Visão de Bot com Grad-CAM Real
 * =====================================================================
 *
 * MELHORIAS v2.0:
 * 1. Grad-CAM Real - Extrai ativações reais das camadas convolucionais
 * 2. Amostragem de Vídeo Completa - Analisa 1 frame/segundo
 * 3. Agregação Temporal - Combina resultados de múltiplos frames
 * 4. Memory Management - Cleanup adequado de tensores
 */

import Tesseract from 'tesseract.js';
import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';

// Palavras-chave que disparam alertas de vulnerabilidade
export const PROHIBITED_KEYWORDS = [
  'dinheiro', 'lucro', 'emagrecimento', 'aprovado', 'ganhe', 'renda',
  'money', 'profit', 'weight loss', 'approved', 'earn', 'income',
  'grátis', 'free', 'garantido', 'guaranteed', 'milionário', 'millionaire',
  'bitcoin', 'crypto', 'investimento', 'investment', 'rápido', 'fast',
  'promoção', 'desconto', 'oferta', 'clique', 'compre', 'agora'
];

// Singleton para modelos
let mobilenetModel = null;
let gradCamModel = null;
let isModelLoading = false;

// Worker do Tesseract reutilizável
let tesseractWorker = null;

/**
 * Inicializa o worker do Tesseract para reutilização
 */
async function initTesseractWorker() {
  if (tesseractWorker) return tesseractWorker;

  tesseractWorker = await Tesseract.createWorker('por+eng', 1, {
    logger: () => {} // Silencia logs
  });

  return tesseractWorker;
}

/**
 * Carrega o modelo MobileNet e prepara para Grad-CAM
 */
export async function loadMobileNet(onProgress) {
  if (mobilenetModel && gradCamModel) return { mobilenetModel, gradCamModel };

  if (isModelLoading) {
    while (isModelLoading) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    return { mobilenetModel, gradCamModel };
  }

  isModelLoading = true;

  try {
    onProgress?.({ type: 'model', percent: 10, message: 'Inicializando TensorFlow.js...' });
    await tf.ready();

    // Define backend preferido
    if (tf.getBackend() !== 'webgl') {
      await tf.setBackend('webgl');
    }

    onProgress?.({ type: 'model', percent: 30, message: 'Carregando MobileNet...' });

    // Carrega MobileNet v2
    mobilenetModel = await mobilenet.load({
      version: 2,
      alpha: 1.0
    });

    onProgress?.({ type: 'model', percent: 60, message: 'Preparando Grad-CAM...' });

    // Cria modelo para Grad-CAM (acessa camadas intermediárias)
    // MobileNetV2 usa a camada 'conv_pw_13_relu' como última convolucional
    const baseModel = mobilenetModel.model;
    const lastConvLayer = baseModel.getLayer('conv_pw_13_relu') || baseModel.layers[baseModel.layers.length - 4];

    gradCamModel = tf.model({
      inputs: baseModel.inputs,
      outputs: [lastConvLayer.output, baseModel.outputs[0]]
    });

    onProgress?.({ type: 'model', percent: 100, message: 'Modelos carregados!' });

    return { mobilenetModel, gradCamModel };
  } catch (error) {
    console.error('Erro ao carregar modelos:', error);
    // Fallback: usa apenas MobileNet sem Grad-CAM
    if (mobilenetModel) {
      gradCamModel = null;
      return { mobilenetModel, gradCamModel: null };
    }
    throw error;
  } finally {
    isModelLoading = false;
  }
}

/**
 * Executa OCR usando worker reutilizável
 */
export async function performOCR(imageSource, onProgress) {
  onProgress?.({ type: 'ocr', percent: 0, message: 'Iniciando análise OCR...' });

  try {
    // Usa API simples para compatibilidade
    const result = await Tesseract.recognize(imageSource, 'por+eng', {
      logger: (m) => {
        if (m.status === 'recognizing text') {
          onProgress?.({
            type: 'ocr',
            percent: Math.round(m.progress * 100),
            message: `Analisando texto: ${Math.round(m.progress * 100)}%`
          });
        }
      }
    });

    const text = result.data.text.toLowerCase();
    const words = result.data.words || [];
    const confidence = result.data.confidence || 0;

    // Detecta palavras proibidas
    const detectedKeywords = PROHIBITED_KEYWORDS.filter(keyword =>
      text.includes(keyword.toLowerCase())
    );

    // Calcula bounding boxes das palavras detectadas
    const wordBoundingBoxes = words.map(word => ({
      text: word.text,
      bbox: word.bbox,
      confidence: word.confidence
    }));

    return {
      fullText: result.data.text,
      confidence,
      detectedKeywords,
      hasVulnerability: detectedKeywords.length > 0,
      wordBoundingBoxes,
      wordCount: words.length
    };
  } catch (error) {
    console.error('Erro no OCR:', error);
    return {
      fullText: '',
      confidence: 0,
      detectedKeywords: [],
      hasVulnerability: false,
      wordBoundingBoxes: [],
      wordCount: 0
    };
  }
}

/**
 * Gera Grad-CAM REAL usando ativações do modelo
 *
 * Grad-CAM = ReLU(sum(alpha_k * A_k))
 * onde alpha_k = global_average_pooling(gradients)
 */
export async function generateGradCAM(imageElement, onProgress) {
  onProgress?.({ type: 'gradcam', percent: 0, message: 'Gerando Grad-CAM...' });

  try {
    const { mobilenetModel: model, gradCamModel: gcModel } = await loadMobileNet(onProgress);

    // Prepara imagem
    const inputTensor = tf.tidy(() => {
      return tf.browser.fromPixels(imageElement)
        .resizeBilinear([224, 224])
        .expandDims(0)
        .toFloat()
        .div(127.5)
        .sub(1); // Normalização MobileNet
    });

    onProgress?.({ type: 'gradcam', percent: 30, message: 'Calculando ativações...' });

    let saliencyMap;
    let predictions;

    if (gcModel) {
      // =============================================
      // GRAD-CAM REAL
      // =============================================
      const [convOutputs, modelOutput] = gcModel.predict(inputTensor);

      // Obtém a classe com maior probabilidade
      const predictionData = await modelOutput.data();
      const classIndex = predictionData.indexOf(Math.max(...predictionData));

      // Calcula gradientes em relação à classe predita
      const gradients = tf.tidy(() => {
        return tf.grad(x => {
          const [convOut, output] = gcModel.predict(x);
          return output.gather([classIndex], 1);
        })(inputTensor);
      });

      onProgress?.({ type: 'gradcam', percent: 50, message: 'Calculando pesos...' });

      // Calcula pesos (global average pooling dos gradientes)
      const weights = tf.tidy(() => {
        return gradients.mean([1, 2]); // GAP sobre dimensões espaciais
      });

      onProgress?.({ type: 'gradcam', percent: 70, message: 'Gerando heatmap...' });

      // Gera Grad-CAM: soma ponderada das ativações
      const gradCam = tf.tidy(() => {
        const convData = convOutputs.squeeze();
        const weightsData = weights.squeeze();

        // Multiplica cada canal pelo seu peso
        let cam = tf.zeros([convData.shape[0], convData.shape[1]]);
        for (let i = 0; i < convData.shape[2]; i++) {
          const channel = convData.slice([0, 0, i], [convData.shape[0], convData.shape[1], 1]).squeeze();
          const weight = weightsData.slice([i], [1]);
          cam = cam.add(channel.mul(weight));
        }

        // ReLU e normalização
        cam = tf.relu(cam);
        const max = cam.max();
        const min = cam.min();
        return cam.sub(min).div(max.sub(min).add(1e-8));
      });

      // Redimensiona para tamanho da imagem
      const resizedCam = tf.image.resizeBilinear(
        gradCam.expandDims(-1).expandDims(0),
        [imageElement.height || 224, imageElement.width || 224]
      ).squeeze();

      saliencyMap = await tensorToSaliencyMap(resizedCam, imageElement.width || 224, imageElement.height || 224);

      // Obtém predictions via MobileNet original
      predictions = await model.classify(imageElement, 10);

      // Cleanup
      tf.dispose([inputTensor, convOutputs, modelOutput, gradients, weights, gradCam, resizedCam]);

    } else {
      // =============================================
      // FALLBACK: Gradient Saliency (se Grad-CAM falhar)
      // =============================================
      saliencyMap = await computeGradientSaliency(inputTensor, imageElement);
      predictions = await model.classify(imageElement, 10);
      tf.dispose(inputTensor);
    }

    onProgress?.({ type: 'gradcam', percent: 90, message: 'Processando resultados...' });

    // Calcula dispersão e hotspots
    const dispersion = calculateDispersion(saliencyMap);
    const hotspots = findHotspots(saliencyMap, imageElement.width || 224, imageElement.height || 224);

    onProgress?.({ type: 'gradcam', percent: 100, message: 'Grad-CAM gerado!' });

    return {
      saliencyData: saliencyMap,
      predictions,
      dispersion,
      hotspots,
      isFocused: dispersion < 0.5,
      method: gcModel ? 'grad-cam' : 'gradient-saliency'
    };
  } catch (error) {
    console.error('Erro ao gerar Grad-CAM:', error);
    // Fallback para método simples
    return generateSimpleSaliency(imageElement, onProgress);
  }
}

/**
 * Converte tensor para formato de saliency map
 */
async function tensorToSaliencyMap(tensor, width, height) {
  const data = await tensor.data();
  const saliencyMap = new Float32Array(data.length);

  // Normaliza para 0-1
  let max = -Infinity, min = Infinity;
  for (let i = 0; i < data.length; i++) {
    max = Math.max(max, data[i]);
    min = Math.min(min, data[i]);
  }

  const range = max - min || 1;
  for (let i = 0; i < data.length; i++) {
    saliencyMap[i] = (data[i] - min) / range;
  }

  return {
    data: saliencyMap,
    width: Math.round(Math.sqrt(data.length)),
    height: Math.round(Math.sqrt(data.length))
  };
}

/**
 * Fallback: Saliency baseado em gradiente de intensidade
 */
async function computeGradientSaliency(tensor, imageElement) {
  const width = 224;
  const height = 224;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(imageElement, 0, 0, width, height);
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;

  const saliencyMap = new Float32Array(width * height);

  // Sobel operator
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = (y * width + x) * 4;

      const leftIdx = (y * width + (x - 1)) * 4;
      const rightIdx = (y * width + (x + 1)) * 4;
      const gx = Math.abs(
        (data[rightIdx] + data[rightIdx + 1] + data[rightIdx + 2]) / 3 -
        (data[leftIdx] + data[leftIdx + 1] + data[leftIdx + 2]) / 3
      );

      const topIdx = ((y - 1) * width + x) * 4;
      const bottomIdx = ((y + 1) * width + x) * 4;
      const gy = Math.abs(
        (data[bottomIdx] + data[bottomIdx + 1] + data[bottomIdx + 2]) / 3 -
        (data[topIdx] + data[topIdx + 1] + data[topIdx + 2]) / 3
      );

      saliencyMap[y * width + x] = Math.sqrt(gx * gx + gy * gy);
    }
  }

  // Normaliza
  const max = Math.max(...saliencyMap);
  if (max > 0) {
    for (let i = 0; i < saliencyMap.length; i++) {
      saliencyMap[i] /= max;
    }
  }

  return { data: saliencyMap, width, height };
}

/**
 * Fallback simples para saliency
 */
async function generateSimpleSaliency(imageElement, onProgress) {
  const saliencyMap = await computeGradientSaliency(null, imageElement);
  const dispersion = calculateDispersion(saliencyMap);
  const hotspots = findHotspots(saliencyMap, imageElement.width || 224, imageElement.height || 224);

  return {
    saliencyData: saliencyMap,
    predictions: [],
    dispersion,
    hotspots,
    isFocused: dispersion < 0.5,
    method: 'edge-detection'
  };
}

/**
 * Calcula dispersão do mapa de saliência
 */
function calculateDispersion(saliencyMap) {
  const { data, width, height } = saliencyMap;

  const gridSize = 4;
  const cellWidth = Math.floor(width / gridSize);
  const cellHeight = Math.floor(height / gridSize);
  const cellSums = [];

  for (let gy = 0; gy < gridSize; gy++) {
    for (let gx = 0; gx < gridSize; gx++) {
      let sum = 0;
      let count = 0;

      for (let y = gy * cellHeight; y < (gy + 1) * cellHeight && y < height; y++) {
        for (let x = gx * cellWidth; x < (gx + 1) * cellWidth && x < width; x++) {
          sum += data[y * width + x];
          count++;
        }
      }

      cellSums.push(count > 0 ? sum / count : 0);
    }
  }

  const mean = cellSums.reduce((a, b) => a + b, 0) / cellSums.length;
  const variance = cellSums.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / cellSums.length;
  const stdDev = Math.sqrt(variance);

  return Math.min(1, stdDev * 2);
}

/**
 * Encontra hotspots no mapa de saliência
 */
function findHotspots(saliencyMap, originalWidth, originalHeight) {
  const { data, width, height } = saliencyMap;
  const threshold = 0.7;
  const hotspots = [];

  const scaleX = originalWidth / width;
  const scaleY = originalHeight / height;

  const visited = new Set();

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      if (data[idx] >= threshold && !visited.has(idx)) {
        const region = floodFill(data, width, height, x, y, threshold, visited);
        if (region.pixels.length > 10) {
          hotspots.push({
            x: Math.round(region.centerX * scaleX),
            y: Math.round(region.centerY * scaleY),
            width: Math.round(region.width * scaleX),
            height: Math.round(region.height * scaleY),
            intensity: region.maxIntensity,
            area: region.pixels.length
          });
        }
      }
    }
  }

  return hotspots.sort((a, b) => b.intensity - a.intensity).slice(0, 5);
}

/**
 * Flood fill para encontrar regiões conectadas
 */
function floodFill(data, width, height, startX, startY, threshold, visited) {
  const pixels = [];
  const queue = [[startX, startY]];
  let minX = startX, maxX = startX, minY = startY, maxY = startY;
  let sumX = 0, sumY = 0, maxIntensity = 0;

  while (queue.length > 0) {
    const [x, y] = queue.shift();
    const idx = y * width + x;

    if (x < 0 || x >= width || y < 0 || y >= height) continue;
    if (visited.has(idx)) continue;
    if (data[idx] < threshold) continue;

    visited.add(idx);
    pixels.push({ x, y, value: data[idx] });

    sumX += x;
    sumY += y;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
    maxIntensity = Math.max(maxIntensity, data[idx]);

    queue.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]);
  }

  return {
    pixels,
    centerX: pixels.length > 0 ? sumX / pixels.length : startX,
    centerY: pixels.length > 0 ? sumY / pixels.length : startY,
    width: maxX - minX + 1,
    height: maxY - minY + 1,
    maxIntensity
  };
}

/**
 * Calcula o Índice de Ofuscação
 */
export function calculateObfuscationIndex(ocrResult, saliencyResult) {
  let score = 100;
  const factors = [];

  if (ocrResult) {
    const ocrPenalty = Math.min(40, ocrResult.confidence * 0.4);
    score -= ocrPenalty;
    factors.push({
      name: 'Confiança OCR',
      impact: -ocrPenalty,
      detail: `OCR leu com ${ocrResult.confidence.toFixed(1)}% de confiança`
    });

    if (ocrResult.hasVulnerability) {
      const keywordPenalty = Math.min(30, ocrResult.detectedKeywords.length * 10);
      score -= keywordPenalty;
      factors.push({
        name: 'Palavras Proibidas',
        impact: -keywordPenalty,
        detail: `Detectadas: ${ocrResult.detectedKeywords.join(', ')}`
      });
    }
  }

  if (saliencyResult) {
    if (saliencyResult.isFocused) {
      const focusPenalty = (1 - saliencyResult.dispersion) * 30;
      score -= focusPenalty;
      factors.push({
        name: 'Atenção Concentrada',
        impact: -focusPenalty,
        detail: `IA focada em área específica (dispersão: ${(saliencyResult.dispersion * 100).toFixed(1)}%)`
      });
    } else {
      factors.push({
        name: 'Atenção Dispersa',
        impact: 0,
        detail: 'Mapa de calor bem distribuído'
      });
    }

    if (saliencyResult.hotspots.length > 0) {
      const maxHotspot = saliencyResult.hotspots[0];
      if (maxHotspot.intensity > 0.9) {
        const hotspotPenalty = 10;
        score -= hotspotPenalty;
        factors.push({
          name: 'Hotspot Intenso',
          impact: -hotspotPenalty,
          detail: `Área de alta atenção detectada (${(maxHotspot.intensity * 100).toFixed(0)}%)`
        });
      }
    }
  }

  score = Math.max(0, Math.min(100, score));

  return {
    score: Math.round(score),
    level: getSecurityLevel(score),
    factors
  };
}

/**
 * Determina o nível de segurança
 */
function getSecurityLevel(score) {
  if (score >= 80) return { name: 'Seguro', color: 'green', icon: 'shield-check' };
  if (score >= 60) return { name: 'Moderado', color: 'yellow', icon: 'shield-alert' };
  if (score >= 40) return { name: 'Baixo', color: 'orange', icon: 'shield-x' };
  return { name: 'Crítico', color: 'red', icon: 'alert-triangle' };
}

/**
 * Gera visualização estilo "Matrix"
 */
export function generateMatrixView(imageElement, saliencyMap) {
  const canvas = document.createElement('canvas');
  canvas.width = imageElement.width || imageElement.naturalWidth || 640;
  canvas.height = imageElement.height || imageElement.naturalHeight || 480;
  const ctx = canvas.getContext('2d');

  ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;

  for (let i = 0; i < data.length; i += 4) {
    const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
    data[i] = gray * 0.2;
    data[i + 1] = gray * 0.8;
    data[i + 2] = gray * 0.2;
  }

  ctx.putImageData(imageData, 0, 0);
  applyEdgeDetection(ctx, canvas.width, canvas.height);

  return canvas.toDataURL('image/png');
}

/**
 * Aplica detecção de bordas Sobel
 */
function applyEdgeDetection(ctx, width, height) {
  const imageData = ctx.getImageData(0, 0, width, height);
  const data = imageData.data;
  const output = new Uint8ClampedArray(data.length);

  const sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
  const sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0, gy = 0;

      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const idx = ((y + ky) * width + (x + kx)) * 4;
          const intensity = data[idx + 1];
          gx += intensity * sobelX[ky + 1][kx + 1];
          gy += intensity * sobelY[ky + 1][kx + 1];
        }
      }

      const magnitude = Math.min(255, Math.sqrt(gx * gx + gy * gy));
      const idx = (y * width + x) * 4;

      output[idx] = 0;
      output[idx + 1] = magnitude > 30 ? 255 : magnitude * 2;
      output[idx + 2] = magnitude > 50 ? 100 : 0;
      output[idx + 3] = 255;
    }
  }

  for (let i = 0; i < data.length; i++) {
    imageData.data[i] = output[i] || data[i];
  }

  ctx.putImageData(imageData, 0, 0);
}

/**
 * Gera heatmap overlay
 */
export function generateHeatmapOverlay(imageElement, saliencyMap) {
  const canvas = document.createElement('canvas');
  canvas.width = imageElement.width || imageElement.naturalWidth || 640;
  canvas.height = imageElement.height || imageElement.naturalHeight || 480;
  const ctx = canvas.getContext('2d');

  ctx.globalAlpha = 0.6;
  ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
  ctx.globalAlpha = 1.0;

  if (!saliencyMap || !saliencyMap.data) {
    return canvas.toDataURL('image/png');
  }

  const { data, width: smWidth, height: smHeight } = saliencyMap;

  const scaleX = canvas.width / smWidth;
  const scaleY = canvas.height / smHeight;

  for (let y = 0; y < smHeight; y++) {
    for (let x = 0; x < smWidth; x++) {
      const value = data[y * smWidth + x];
      if (value > 0.3) {
        const color = getHeatmapColor(value);
        ctx.fillStyle = `rgba(${color.r}, ${color.g}, ${color.b}, ${value * 0.6})`;
        ctx.fillRect(
          Math.round(x * scaleX),
          Math.round(y * scaleY),
          Math.ceil(scaleX),
          Math.ceil(scaleY)
        );
      }
    }
  }

  return canvas.toDataURL('image/png');
}

/**
 * Converte valor de saliência para cor
 */
function getHeatmapColor(value) {
  if (value < 0.25) {
    return { r: 0, g: Math.round(255 * value * 4), b: 255 };
  } else if (value < 0.5) {
    return { r: 0, g: 255, b: Math.round(255 * (1 - (value - 0.25) * 4)) };
  } else if (value < 0.75) {
    return { r: Math.round(255 * (value - 0.5) * 4), g: 255, b: 0 };
  } else {
    return { r: 255, g: Math.round(255 * (1 - (value - 0.75) * 4)), b: 0 };
  }
}

/**
 * NOVO v2.0: Analisa vídeo completo com amostragem temporal
 *
 * @param {HTMLVideoElement} videoElement - Elemento de vídeo
 * @param {Object} options - Opções de análise
 * @param {Function} onProgress - Callback de progresso
 * @returns {Object} - Resultados agregados
 */
export async function analyzeFullVideo(videoElement, options = {}, onProgress) {
  const {
    sampleRate = 1,           // 1 frame por segundo
    maxFrames = 30,           // Máximo de frames a analisar
    skipStart = 0,            // Segundos para pular no início
    skipEnd = 0               // Segundos para pular no final
  } = options;

  const duration = videoElement.duration;
  const effectiveDuration = duration - skipStart - skipEnd;
  const totalFrames = Math.min(maxFrames, Math.floor(effectiveDuration / sampleRate));

  onProgress?.({ type: 'video', percent: 0, message: `Analisando ${totalFrames} frames...` });

  const frameResults = [];
  const allKeywords = new Set();
  let maxConfidence = 0;
  let avgDispersion = 0;
  let worstScore = 100;

  // Salva posição original
  const originalTime = videoElement.currentTime;

  for (let i = 0; i < totalFrames; i++) {
    const targetTime = skipStart + (i * sampleRate);

    // Seek para o frame
    videoElement.currentTime = targetTime;
    await new Promise(resolve => {
      videoElement.onseeked = resolve;
    });

    // Pequena pausa para garantir que o frame está renderizado
    await new Promise(resolve => setTimeout(resolve, 50));

    onProgress?.({
      type: 'video',
      percent: Math.round((i / totalFrames) * 100),
      message: `Analisando frame ${i + 1}/${totalFrames} (${targetTime.toFixed(1)}s)`
    });

    // Analisa o frame atual
    const frameAnalysis = await processVideoFrame(videoElement, () => {});
    frameResults.push({
      time: targetTime,
      ...frameAnalysis
    });

    // Agrega resultados
    if (frameAnalysis.ocrResult) {
      maxConfidence = Math.max(maxConfidence, frameAnalysis.ocrResult.confidence);
      frameAnalysis.ocrResult.detectedKeywords.forEach(kw => allKeywords.add(kw));
    }

    if (frameAnalysis.saliencyResult) {
      avgDispersion += frameAnalysis.saliencyResult.dispersion;
    }

    worstScore = Math.min(worstScore, frameAnalysis.obfuscationIndex.score);
  }

  // Restaura posição original
  videoElement.currentTime = originalTime;

  // Calcula médias
  avgDispersion /= totalFrames;

  onProgress?.({ type: 'video', percent: 100, message: 'Análise de vídeo completa!' });

  // Resultado agregado
  const aggregatedOCR = {
    fullText: frameResults.map(f => f.ocrResult?.fullText || '').join(' '),
    confidence: maxConfidence,
    detectedKeywords: Array.from(allKeywords),
    hasVulnerability: allKeywords.size > 0,
    wordCount: frameResults.reduce((sum, f) => sum + (f.ocrResult?.wordCount || 0), 0)
  };

  const aggregatedSaliency = {
    dispersion: avgDispersion,
    isFocused: avgDispersion < 0.5,
    hotspots: frameResults.flatMap(f => f.saliencyResult?.hotspots || []).slice(0, 10)
  };

  const aggregatedIndex = calculateObfuscationIndex(aggregatedOCR, aggregatedSaliency);

  return {
    frameResults,
    totalFrames,
    duration,
    aggregated: {
      ocrResult: aggregatedOCR,
      saliencyResult: aggregatedSaliency,
      obfuscationIndex: aggregatedIndex,
      worstFrameScore: worstScore
    }
  };
}

/**
 * Processa frame único de vídeo
 */
export async function processVideoFrame(videoElement, onProgress) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0);

  const imageDataUrl = canvas.toDataURL('image/png');
  const img = new Image();

  return new Promise((resolve) => {
    img.onload = async () => {
      const ocrResult = await performOCR(canvas, onProgress);
      const saliencyResult = await generateGradCAM(img, onProgress);
      const matrixView = generateMatrixView(img, saliencyResult.saliencyData);
      const heatmapView = generateHeatmapOverlay(img, saliencyResult.saliencyData);
      const obfuscationIndex = calculateObfuscationIndex(ocrResult, saliencyResult);

      resolve({
        ocrResult,
        saliencyResult,
        matrixView,
        heatmapView,
        obfuscationIndex
      });
    };
    img.src = imageDataUrl;
  });
}

/**
 * Análise completa de imagem (usa Grad-CAM)
 */
export async function analyzeImage(imageSource, onProgress) {
  onProgress?.({ type: 'init', percent: 0, message: 'Iniciando análise...' });

  await loadMobileNet(onProgress);

  let imageElement = imageSource;
  if (typeof imageSource === 'string') {
    imageElement = await loadImage(imageSource);
  }

  onProgress?.({ type: 'analysis', percent: 20, message: 'Executando OCR...' });
  const ocrResult = await performOCR(imageElement, onProgress);

  onProgress?.({ type: 'analysis', percent: 50, message: 'Gerando Grad-CAM...' });
  const saliencyResult = await generateGradCAM(imageElement, onProgress);

  onProgress?.({ type: 'analysis', percent: 70, message: 'Gerando visualizações...' });
  const matrixView = generateMatrixView(imageElement, saliencyResult.saliencyData);
  const heatmapView = generateHeatmapOverlay(imageElement, saliencyResult.saliencyData);

  onProgress?.({ type: 'analysis', percent: 90, message: 'Calculando índice de ofuscação...' });
  const obfuscationIndex = calculateObfuscationIndex(ocrResult, saliencyResult);

  onProgress?.({ type: 'complete', percent: 100, message: 'Análise completa!' });

  return {
    ocrResult,
    saliencyResult,
    matrixView,
    heatmapView,
    obfuscationIndex,
    originalImage: imageElement.src || imageElement
  };
}

/**
 * Carrega imagem de URL ou base64
 */
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

/**
 * Libera recursos (cleanup)
 */
export function cleanup() {
  if (tesseractWorker) {
    tesseractWorker.terminate();
    tesseractWorker = null;
  }

  // Limpa cache de tensores do TensorFlow
  tf.disposeVariables();
}

// Exporta também como generateSaliencyMap para compatibilidade
export const generateSaliencyMap = generateGradCAM;

export default {
  loadMobileNet,
  performOCR,
  generateGradCAM,
  generateSaliencyMap: generateGradCAM,
  calculateObfuscationIndex,
  generateMatrixView,
  generateHeatmapOverlay,
  analyzeImage,
  processVideoFrame,
  analyzeFullVideo,
  cleanup,
  PROHIBITED_KEYWORDS
};
