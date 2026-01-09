import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Eye, Upload, Play, AlertTriangle, Shield, ShieldCheck, ShieldX, ShieldAlert,
  Camera, Video, Loader2, X, ZoomIn, Crosshair, FileText, Activity, Target, Info
} from 'lucide-react';
import {
  analyzeImage,
  processVideoFrame,
  loadMobileNet,
  PROHIBITED_KEYWORDS
} from '../lib/botEyeProcessor';

const BotEyeSimulator = () => {
  // Estados
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState(null); // 'image' ou 'video'
  const [preview, setPreview] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [progress, setProgress] = useState({ type: '', percent: 0, message: '' });
  const [results, setResults] = useState(null);
  const [viewMode, setViewMode] = useState('original'); // 'original', 'matrix', 'heatmap'
  const [error, setError] = useState(null);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);

  // Pré-carrega o modelo ao montar o componente
  useEffect(() => {
    const preloadModel = async () => {
      setIsModelLoading(true);
      try {
        await loadMobileNet((p) => setProgress(p));
      } catch (err) {
        console.error('Erro ao pré-carregar modelo:', err);
      } finally {
        setIsModelLoading(false);
      }
    };
    preloadModel();
  }, []);

  // Handler de upload de arquivo
  const handleFileUpload = useCallback((e) => {
    const uploadedFile = e.target.files?.[0];
    if (!uploadedFile) return;

    setError(null);
    setResults(null);

    const isImage = uploadedFile.type.startsWith('image/');
    const isVideo = uploadedFile.type.startsWith('video/');

    if (!isImage && !isVideo) {
      setError('Por favor, selecione uma imagem ou vídeo.');
      return;
    }

    setFile(uploadedFile);
    setFileType(isImage ? 'image' : 'video');
    setPreview(URL.createObjectURL(uploadedFile));
  }, []);

  // Handler de simulação
  const handleSimulate = useCallback(async () => {
    if (!file || !preview) return;

    setIsProcessing(true);
    setError(null);
    setProgress({ type: 'init', percent: 0, message: 'Iniciando simulação...' });

    try {
      if (fileType === 'image') {
        const analysisResults = await analyzeImage(preview, setProgress);
        setResults(analysisResults);
      } else if (fileType === 'video' && videoRef.current) {
        // Para vídeo, analisa o frame atual
        const frameResults = await processVideoFrame(videoRef.current, setProgress);
        setResults(frameResults);
      }
    } catch (err) {
      console.error('Erro na simulação:', err);
      setError(`Erro durante a análise: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  }, [file, preview, fileType]);

  // Reset
  const handleReset = useCallback(() => {
    setFile(null);
    setFileType(null);
    setPreview(null);
    setResults(null);
    setError(null);
    setViewMode('original');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  // Renderiza o indicador de segurança
  const renderSecurityBadge = (level) => {
    const icons = {
      'shield-check': <ShieldCheck className="w-6 h-6" />,
      'shield-alert': <ShieldAlert className="w-6 h-6" />,
      'shield-x': <ShieldX className="w-6 h-6" />,
      'alert-triangle': <AlertTriangle className="w-6 h-6" />
    };

    const colors = {
      green: 'bg-green-500/20 text-green-400 border-green-500/50',
      yellow: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
      orange: 'bg-orange-500/20 text-orange-400 border-orange-500/50',
      red: 'bg-red-500/20 text-red-400 border-red-500/50'
    };

    return (
      <div className={`flex items-center gap-2 px-4 py-2 rounded-lg border ${colors[level.color]}`}>
        {icons[level.icon]}
        <span className="font-semibold">{level.name}</span>
      </div>
    );
  };

  // Renderiza a imagem/vídeo atual baseado no modo de visualização
  const getCurrentView = () => {
    if (!results) return preview;
    switch (viewMode) {
      case 'matrix': return results.matrixView;
      case 'heatmap': return results.heatmapView;
      default: return preview;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-gradient-to-br from-cyan-600 to-teal-600 rounded-xl">
            <Eye className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">BotEye Simulator</h2>
            <p className="text-gray-400 text-sm">Veja através dos olhos de um bot de moderação</p>
          </div>
        </div>
        {isModelLoading && (
          <div className="flex items-center gap-2 text-cyan-400">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span className="text-sm">Carregando modelo IA...</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Coluna 1: Upload e Controles */}
        <div className="space-y-4">
          {/* Upload Area */}
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
              <Upload className="w-4 h-4" />
              Upload de Mídia
            </h3>

            {!file ? (
              <label
                className="flex flex-col items-center justify-center w-full h-40 border-2 border-dashed border-gray-600 rounded-lg cursor-pointer hover:border-cyan-500 hover:bg-gray-700/30 transition-all"
              >
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Camera className="w-10 h-10 text-gray-500 mb-2" />
                  <p className="mb-2 text-sm text-gray-400">
                    <span className="font-semibold text-cyan-400">Clique para upload</span>
                  </p>
                  <p className="text-xs text-gray-500">Imagem ou Vídeo</p>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  accept="image/*,video/*"
                  onChange={handleFileUpload}
                />
              </label>
            ) : (
              <div className="space-y-3">
                <div className="flex items-center justify-between bg-gray-700/50 rounded-lg p-3">
                  <div className="flex items-center gap-2">
                    {fileType === 'image' ? (
                      <Camera className="w-4 h-4 text-cyan-400" />
                    ) : (
                      <Video className="w-4 h-4 text-purple-400" />
                    )}
                    <span className="text-sm text-gray-300 truncate max-w-[150px]">
                      {file.name}
                    </span>
                  </div>
                  <button
                    onClick={handleReset}
                    className="p-1 hover:bg-gray-600 rounded transition-colors"
                  >
                    <X className="w-4 h-4 text-gray-400" />
                  </button>
                </div>
                <p className="text-xs text-gray-500">
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            )}
          </div>

          {/* Botão de Simulação */}
          <button
            onClick={handleSimulate}
            disabled={!file || isProcessing || isModelLoading}
            className="w-full py-3 px-4 bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-500 hover:to-teal-500 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed text-white font-semibold rounded-lg transition-all flex items-center justify-center gap-2"
          >
            {isProcessing ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Analisando...
              </>
            ) : (
              <>
                <Eye className="w-5 h-5" />
                Simular Visão do Bot
              </>
            )}
          </button>

          {/* Progress Bar */}
          {isProcessing && (
            <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">{progress.message}</span>
                <span className="text-cyan-400">{progress.percent}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-cyan-500 to-teal-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress.percent}%` }}
                />
              </div>
            </div>
          )}

          {/* Modo de Visualização */}
          {results && (
            <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
              <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
                <ZoomIn className="w-4 h-4" />
                Modo de Visualização
              </h3>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { id: 'original', label: 'Original', icon: Camera },
                  { id: 'matrix', label: 'Matrix', icon: Crosshair },
                  { id: 'heatmap', label: 'Heatmap', icon: Target }
                ].map(({ id, label, icon: Icon }) => (
                  <button
                    key={id}
                    onClick={() => setViewMode(id)}
                    className={`p-2 rounded-lg text-xs font-medium flex flex-col items-center gap-1 transition-all ${
                      viewMode === id
                        ? 'bg-cyan-600 text-white'
                        : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Info Box */}
          <div className="bg-cyan-900/20 rounded-xl p-4 border border-cyan-700/30">
            <h4 className="text-cyan-400 font-semibold mb-2 flex items-center gap-2">
              <Info className="w-4 h-4" />
              Palavras Monitoradas
            </h4>
            <div className="flex flex-wrap gap-1">
              {PROHIBITED_KEYWORDS.slice(0, 10).map((word, i) => (
                <span
                  key={i}
                  className="px-2 py-0.5 bg-gray-700/50 text-gray-400 text-xs rounded"
                >
                  {word}
                </span>
              ))}
              <span className="px-2 py-0.5 text-gray-500 text-xs">
                +{PROHIBITED_KEYWORDS.length - 10} mais
              </span>
            </div>
          </div>
        </div>

        {/* Coluna 2: Preview */}
        <div className="space-y-4">
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
              <Eye className="w-4 h-4" />
              Visão do Bot
              {viewMode !== 'original' && (
                <span className="px-2 py-0.5 bg-cyan-600/20 text-cyan-400 text-xs rounded-full ml-auto">
                  {viewMode === 'matrix' ? 'Modo Matrix' : 'Mapa de Calor'}
                </span>
              )}
            </h3>

            <div className="relative aspect-video bg-black rounded-lg overflow-hidden flex items-center justify-center">
              {preview ? (
                fileType === 'image' ? (
                  <img
                    src={getCurrentView()}
                    alt="Preview"
                    className="max-w-full max-h-full object-contain"
                  />
                ) : (
                  <video
                    ref={videoRef}
                    src={preview}
                    className="max-w-full max-h-full object-contain"
                    controls
                  />
                )
              ) : (
                <div className="text-gray-500 flex flex-col items-center gap-2">
                  <Eye className="w-12 h-12 opacity-30" />
                  <span className="text-sm">Nenhuma mídia carregada</span>
                </div>
              )}

              {/* Overlay de hotspots */}
              {results?.saliencyResult?.hotspots && viewMode === 'heatmap' && (
                <div className="absolute inset-0 pointer-events-none">
                  {results.saliencyResult.hotspots.map((hotspot, i) => (
                    <div
                      key={i}
                      className="absolute border-2 border-red-500 rounded"
                      style={{
                        left: `${(hotspot.x / 640) * 100}%`,
                        top: `${(hotspot.y / 480) * 100}%`,
                        width: `${(hotspot.width / 640) * 100}%`,
                        height: `${(hotspot.height / 480) * 100}%`
                      }}
                    >
                      <span className="absolute -top-5 left-0 text-xs bg-red-500 text-white px-1 rounded">
                        {(hotspot.intensity * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Erro */}
          {error && (
            <div className="bg-red-900/20 border border-red-700/50 rounded-xl p-4">
              <div className="flex items-center gap-2 text-red-400">
                <AlertTriangle className="w-5 h-5" />
                <span>{error}</span>
              </div>
            </div>
          )}
        </div>

        {/* Coluna 3: Resultados */}
        <div className="space-y-4">
          {/* Score de Segurança */}
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
              <Shield className="w-4 h-4" />
              Índice de Ofuscação
            </h3>

            {results?.obfuscationIndex ? (
              <div className="space-y-4">
                {/* Score circular */}
                <div className="flex items-center justify-center">
                  <div className="relative w-32 h-32">
                    <svg className="w-full h-full transform -rotate-90">
                      <circle
                        cx="64"
                        cy="64"
                        r="56"
                        stroke="currentColor"
                        strokeWidth="8"
                        fill="none"
                        className="text-gray-700"
                      />
                      <circle
                        cx="64"
                        cy="64"
                        r="56"
                        stroke="currentColor"
                        strokeWidth="8"
                        fill="none"
                        strokeDasharray={`${results.obfuscationIndex.score * 3.52} 352`}
                        className={
                          results.obfuscationIndex.score >= 80
                            ? 'text-green-500'
                            : results.obfuscationIndex.score >= 60
                            ? 'text-yellow-500'
                            : results.obfuscationIndex.score >= 40
                            ? 'text-orange-500'
                            : 'text-red-500'
                        }
                      />
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-3xl font-bold text-white">
                        {results.obfuscationIndex.score}%
                      </span>
                    </div>
                  </div>
                </div>

                {/* Badge de segurança */}
                <div className="flex justify-center">
                  {renderSecurityBadge(results.obfuscationIndex.level)}
                </div>

                {/* Fatores */}
                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-gray-400">Fatores de Análise:</h4>
                  {results.obfuscationIndex.factors.map((factor, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between bg-gray-700/30 rounded-lg p-2"
                    >
                      <div className="flex-1">
                        <p className="text-sm text-gray-300">{factor.name}</p>
                        <p className="text-xs text-gray-500">{factor.detail}</p>
                      </div>
                      <span
                        className={`text-sm font-mono ${
                          factor.impact < 0 ? 'text-red-400' : 'text-green-400'
                        }`}
                      >
                        {factor.impact > 0 ? '+' : ''}{factor.impact.toFixed(0)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Shield className="w-12 h-12 mx-auto mb-2 opacity-30" />
                <p className="text-sm">Execute a simulação para ver o índice</p>
              </div>
            )}
          </div>

          {/* Resultados OCR */}
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Análise OCR
            </h3>

            {results?.ocrResult ? (
              <div className="space-y-3">
                {/* Alerta de vulnerabilidade */}
                {results.ocrResult.hasVulnerability && (
                  <div className="bg-red-900/30 border border-red-700/50 rounded-lg p-3">
                    <div className="flex items-center gap-2 text-red-400 font-semibold mb-2">
                      <AlertTriangle className="w-4 h-4" />
                      Alerta de Vulnerabilidade!
                    </div>
                    <p className="text-sm text-gray-300">
                      Palavras detectadas:{' '}
                      <span className="text-red-300">
                        {results.ocrResult.detectedKeywords.join(', ')}
                      </span>
                    </p>
                  </div>
                )}

                {/* Métricas */}
                <div className="grid grid-cols-2 gap-2">
                  <div className="bg-gray-700/30 rounded-lg p-2">
                    <p className="text-xs text-gray-500">Confiança</p>
                    <p className="text-lg font-semibold text-white">
                      {results.ocrResult.confidence.toFixed(1)}%
                    </p>
                  </div>
                  <div className="bg-gray-700/30 rounded-lg p-2">
                    <p className="text-xs text-gray-500">Palavras</p>
                    <p className="text-lg font-semibold text-white">
                      {results.ocrResult.wordCount}
                    </p>
                  </div>
                </div>

                {/* Texto detectado */}
                {results.ocrResult.fullText && (
                  <div>
                    <p className="text-xs text-gray-500 mb-1">Texto Detectado:</p>
                    <div className="bg-gray-900/50 rounded-lg p-2 max-h-24 overflow-y-auto">
                      <p className="text-xs text-gray-400 whitespace-pre-wrap font-mono">
                        {results.ocrResult.fullText || '(nenhum texto detectado)'}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-4 text-gray-500">
                <FileText className="w-8 h-8 mx-auto mb-2 opacity-30" />
                <p className="text-sm">Aguardando análise...</p>
              </div>
            )}
          </div>

          {/* Resultados de Atenção */}
          <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700">
            <h3 className="text-white font-semibold mb-3 flex items-center gap-2">
              <Activity className="w-4 h-4" />
              Mapa de Atenção
            </h3>

            {results?.saliencyResult ? (
              <div className="space-y-3">
                {/* Status de foco */}
                <div
                  className={`rounded-lg p-3 ${
                    results.saliencyResult.isFocused
                      ? 'bg-orange-900/30 border border-orange-700/50'
                      : 'bg-green-900/30 border border-green-700/50'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <Target
                      className={`w-4 h-4 ${
                        results.saliencyResult.isFocused
                          ? 'text-orange-400'
                          : 'text-green-400'
                      }`}
                    />
                    <span
                      className={`text-sm font-medium ${
                        results.saliencyResult.isFocused
                          ? 'text-orange-300'
                          : 'text-green-300'
                      }`}
                    >
                      {results.saliencyResult.isFocused
                        ? 'IA focada em área específica'
                        : 'Atenção bem distribuída'}
                    </span>
                  </div>
                </div>

                {/* Dispersão */}
                <div className="bg-gray-700/30 rounded-lg p-2">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Dispersão:</span>
                    <span className="text-white">
                      {(results.saliencyResult.dispersion * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-1.5">
                    <div
                      className="bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-1.5 rounded-full"
                      style={{ width: `${results.saliencyResult.dispersion * 100}%` }}
                    />
                  </div>
                </div>

                {/* Hotspots */}
                {results.saliencyResult.hotspots.length > 0 && (
                  <div>
                    <p className="text-xs text-gray-500 mb-2">
                      Áreas de Alta Atenção ({results.saliencyResult.hotspots.length}):
                    </p>
                    <div className="space-y-1">
                      {results.saliencyResult.hotspots.slice(0, 3).map((hotspot, i) => (
                        <div
                          key={i}
                          className="flex items-center justify-between bg-gray-700/30 rounded px-2 py-1"
                        >
                          <span className="text-xs text-gray-400">
                            Região {i + 1}
                          </span>
                          <span className="text-xs font-mono text-red-400">
                            {(hotspot.intensity * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Top Predictions */}
                {results.saliencyResult.predictions?.length > 0 && (
                  <div>
                    <p className="text-xs text-gray-500 mb-2">Objetos Detectados:</p>
                    <div className="space-y-1">
                      {results.saliencyResult.predictions.slice(0, 3).map((pred, i) => (
                        <div
                          key={i}
                          className="flex items-center justify-between bg-gray-700/30 rounded px-2 py-1"
                        >
                          <span className="text-xs text-gray-300 truncate max-w-[140px]">
                            {pred.className}
                          </span>
                          <span className="text-xs font-mono text-cyan-400">
                            {(pred.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-4 text-gray-500">
                <Activity className="w-8 h-8 mx-auto mb-2 opacity-30" />
                <p className="text-sm">Aguardando análise...</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BotEyeSimulator;
