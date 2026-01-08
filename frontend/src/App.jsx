/**
 * RobustnessLab - Main Application Component
 * ==========================================
 *
 * Dashboard principal para análise de robustez adversária.
 * Inclui:
 * - Ataques PGD contra classificadores de imagem
 * - Proteção de texto contra OCR (Anti-OCR)
 */

import React, { useState, useCallback } from 'react'
import axios from 'axios'
import ImageUpload from './components/ImageUpload'
import ModelSelector from './components/ModelSelector'
import HyperparameterControls from './components/HyperparameterControls'
import ImageComparison from './components/ImageComparison'
import ConfidenceChart from './components/ConfidenceChart'
import MetricsPanel from './components/MetricsPanel'
import TextProtection from './components/TextProtection'
import TextCamouflage from './components/TextCamouflage'
import MediaProtection from './components/MediaProtection'
import { Shield, AlertTriangle, Zap, Info, Lock, Image, FileText, EyeOff, Video } from 'lucide-react'
import { API_URL } from './config'

function App() {
  // Estado de navegação
  const [activeTab, setActiveTab] = useState('classifier') // 'classifier' ou 'text-protection'

  // Estado da imagem selecionada
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)

  // Estado do modelo
  const [selectedModel, setSelectedModel] = useState('resnet50')

  // Hiperparâmetros do ataque
  const [epsilon, setEpsilon] = useState(0.031)  // 8/255
  const [alpha, setAlpha] = useState(0.008)      // 2/255
  const [iterations, setIterations] = useState(10)

  // Estado do ataque
  const [isAttacking, setIsAttacking] = useState(false)
  const [attackResults, setAttackResults] = useState(null)
  const [error, setError] = useState(null)

  /**
   * Handler para upload de imagem
   */
  const handleImageUpload = useCallback((file) => {
    setSelectedImage(file)
    setAttackResults(null)
    setError(null)

    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => setImagePreview(e.target.result)
      reader.readAsDataURL(file)
    } else {
      setImagePreview(null)
    }
  }, [])

  /**
   * Executa o ataque PGD via API
   */
  const executeAttack = async () => {
    if (!selectedImage) {
      setError('Por favor, selecione uma imagem primeiro.')
      return
    }

    setIsAttacking(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedImage)
      formData.append('epsilon', epsilon)
      formData.append('alpha', alpha)
      formData.append('num_iterations', iterations)
      formData.append('model_id', selectedModel)

      const response = await axios.post(`${API_URL}/attack`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000  // 2 minutos timeout
      })

      setAttackResults(response.data)
    } catch (err) {
      console.error('Erro no ataque:', err)
      setError(
        err.response?.data?.detail ||
        'Erro ao executar ataque. Verifique se o backend está rodando.'
      )
    } finally {
      setIsAttacking(false)
    }
  }

  return (
    <div className="min-h-screen bg-lab-darker text-gray-100">
      {/* Header */}
      <header className="bg-lab-dark border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="w-8 h-8 text-lab-primary" />
              <div>
                <h1 className="text-xl font-bold">RobustnessLab</h1>
                <p className="text-sm text-gray-400">
                  Análise de Robustez Adversária para DNNs
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <Info className="w-4 h-4" />
              <span>Ferramenta de Diagnóstico para Pesquisadores de IA</span>
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex gap-2 mt-4 flex-wrap">
            <button
              onClick={() => setActiveTab('classifier')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all
                ${activeTab === 'classifier'
                  ? 'bg-lab-primary text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200'
                }`}
            >
              <Image className="w-4 h-4" />
              Ataque a Classificadores
            </button>
            <button
              onClick={() => setActiveTab('text-protection')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all
                ${activeTab === 'text-protection'
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200'
                }`}
            >
              <Lock className="w-4 h-4" />
              Proteção Anti-OCR
            </button>
            <button
              onClick={() => setActiveTab('camouflage')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all
                ${activeTab === 'camouflage'
                  ? 'bg-indigo-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200'
                }`}
            >
              <EyeOff className="w-4 h-4" />
              Camuflagem de Texto
            </button>
            <button
              onClick={() => setActiveTab('media-protection')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all
                ${activeTab === 'media-protection'
                  ? 'bg-pink-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200'
                }`}
            >
              <Video className="w-4 h-4" />
              Proteção de Mídia
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'classifier' ? (
          <>
            {/* Info Banner - Classifier Attack */}
            <div className="mb-6 p-4 bg-blue-900/30 border border-blue-700/50 rounded-lg">
              <div className="flex items-start gap-3">
                <Info className="w-5 h-5 text-blue-400 mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="text-blue-200 font-medium mb-1">
                    Sobre o Ataque PGD (Projected Gradient Descent)
                  </p>
                  <p className="text-gray-300">
                    O PGD é um ataque iterativo que usa o gradiente da função de perda
                    para encontrar perturbações que maximizam o erro de classificação.
                    A cada iteração, a perturbação é atualizada na direção do gradiente
                    e projetada de volta para a bola L∞ de raio ε.
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Coluna Esquerda - Controles */}
              <div className="lg:col-span-1 space-y-6">
                {/* Upload de Imagem */}
                <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
                  <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Zap className="w-5 h-5 text-yellow-400" />
                    Configuração do Ataque
                  </h2>

                  <ImageUpload
                    onImageUpload={handleImageUpload}
                    preview={imagePreview}
                  />

                  <div className="mt-4">
                    <ModelSelector
                      selectedModel={selectedModel}
                      onModelChange={setSelectedModel}
                    />
                  </div>
                </div>

                {/* Controles de Hiperparâmetros */}
                <HyperparameterControls
                  epsilon={epsilon}
                  setEpsilon={setEpsilon}
                  alpha={alpha}
                  setAlpha={setAlpha}
                  iterations={iterations}
                  setIterations={setIterations}
                />

                {/* Botão de Ataque */}
                <button
                  onClick={executeAttack}
                  disabled={!selectedImage || isAttacking}
                  className={`w-full py-3 px-4 rounded-lg font-semibold text-white
                    transition-all duration-200 flex items-center justify-center gap-2
                    ${!selectedImage || isAttacking
                      ? 'bg-gray-600 cursor-not-allowed'
                      : 'gradient-danger hover:opacity-90 hover:shadow-lg'
                    }`}
                >
                  {isAttacking ? (
                    <>
                      <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      Executando PGD...
                    </>
                  ) : (
                    <>
                      <AlertTriangle className="w-5 h-5" />
                      Executar Ataque PGD
                    </>
                  )}
                </button>

                {/* Mensagem de Erro */}
                {error && (
                  <div className="p-4 bg-red-900/30 border border-red-700/50 rounded-lg">
                    <p className="text-red-300 text-sm">{error}</p>
                  </div>
                )}
              </div>

              {/* Coluna Direita - Resultados */}
              <div className="lg:col-span-2 space-y-6">
                {attackResults ? (
                  <>
                    {/* Comparação Visual */}
                    <ImageComparison
                      originalImage={attackResults.original_image}
                      adversarialImage={attackResults.adversarial_image}
                      heatmap={attackResults.perturbation_heatmap}
                      originalFile={selectedImage}
                      attackParams={{
                        epsilon: epsilon,
                        alpha: alpha,
                        num_iterations: iterations,
                        model_id: selectedModel
                      }}
                    />

                    {/* Gráficos de Confiança */}
                    <ConfidenceChart
                      originalPredictions={attackResults.original_predictions}
                      adversarialPredictions={attackResults.adversarial_predictions}
                    />

                    {/* Métricas Detalhadas */}
                    <MetricsPanel
                      metrics={attackResults.metrics}
                      params={attackResults.attack_params}
                    />
                  </>
                ) : (
                  /* Placeholder quando não há resultados */
                  <div className="bg-lab-dark rounded-lg border border-gray-700 p-12 text-center">
                    <Shield className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-gray-400 mb-2">
                      Nenhum Ataque Executado
                    </h3>
                    <p className="text-gray-500 max-w-md mx-auto">
                      Faça upload de uma imagem, configure os hiperparâmetros
                      e clique em "Executar Ataque PGD" para analisar a
                      vulnerabilidade do modelo.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </>
        ) : activeTab === 'text-protection' ? (
          /* Text Protection Tab */
          <TextProtection />
        ) : activeTab === 'camouflage' ? (
          /* Camouflage Tab */
          <TextCamouflage />
        ) : (
          /* Media Protection Tab */
          <MediaProtection />
        )}
      </main>

      {/* Footer */}
      <footer className="mt-12 py-6 border-t border-gray-800">
        <div className="max-w-7xl mx-auto px-4 text-center text-sm text-gray-500">
          <p>
            RobustnessLab - Ferramenta de diagnóstico para pesquisadores de IA.
          </p>
          <p className="mt-1">
            Use responsavelmente para testar a robustez de seus próprios modelos.
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
