/**
 * ModelSelector Component
 * =======================
 *
 * Seletor de modelo alvo para o ataque.
 * Exibe informações sobre cada modelo disponível.
 */

import React from 'react'
import { Cpu } from 'lucide-react'

// Modelos disponíveis (deve corresponder ao backend)
const AVAILABLE_MODELS = [
  {
    id: 'resnet50',
    name: 'ResNet-50',
    inputSize: 224,
    description: 'Rede residual profunda. Boa robustez base.',
    accuracy: '76.1%'
  },
  {
    id: 'inception_v3',
    name: 'Inception V3',
    inputSize: 299,
    description: 'Módulos inception paralelos. Alta precisão.',
    accuracy: '77.3%'
  },
  {
    id: 'vgg16',
    name: 'VGG-16',
    inputSize: 224,
    description: 'Arquitetura sequencial clássica.',
    accuracy: '71.6%'
  },
  {
    id: 'mobilenet_v2',
    name: 'MobileNet V2',
    inputSize: 224,
    description: 'Arquitetura leve para mobile.',
    accuracy: '71.9%'
  }
]

function ModelSelector({ selectedModel, onModelChange }) {
  const currentModel = AVAILABLE_MODELS.find(m => m.id === selectedModel)

  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-gray-300">
        Modelo Alvo
      </label>

      <div className="relative">
        <select
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
          className="w-full px-4 py-2.5 bg-gray-800 border border-gray-600 rounded-lg
                     text-gray-100 appearance-none cursor-pointer
                     focus:outline-none focus:ring-2 focus:ring-lab-primary focus:border-transparent"
        >
          {AVAILABLE_MODELS.map(model => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </select>

        {/* Ícone do dropdown */}
        <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none">
          <Cpu className="w-5 h-5 text-gray-400" />
        </div>
      </div>

      {/* Info do modelo selecionado */}
      {currentModel && (
        <div className="p-3 bg-gray-800/50 rounded-lg border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-gray-300">
              {currentModel.name}
            </span>
            <span className="text-xs px-2 py-0.5 bg-lab-primary/20 text-lab-primary rounded">
              Top-1: {currentModel.accuracy}
            </span>
          </div>
          <p className="text-xs text-gray-400">
            {currentModel.description}
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Input: {currentModel.inputSize}x{currentModel.inputSize}px
          </p>
        </div>
      )}
    </div>
  )
}

export default ModelSelector
