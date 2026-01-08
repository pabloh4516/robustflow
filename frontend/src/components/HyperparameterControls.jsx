/**
 * HyperparameterControls Component
 * ================================
 *
 * Controles para configurar a intensidade do ataque
 * com explicações simples de cada parâmetro.
 */

import React from 'react'
import { Sliders, Info, Eye, Zap, RotateCw } from 'lucide-react'

function HyperparameterControls({
  epsilon,
  setEpsilon,
  alpha,
  setAlpha,
  iterations,
  setIterations
}) {
  return (
    <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
      <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Sliders className="w-5 h-5 text-purple-400" />
        Configurações do Ataque
      </h2>

      <div className="space-y-5">
        {/* Força da Perturbação */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Eye className="w-4 h-4 text-yellow-400" />
              Força da Perturbação
            </label>
            <span className="text-xs px-2 py-0.5 bg-yellow-900/30 text-yellow-400 rounded font-mono">
              {(epsilon * 100).toFixed(1)}%
            </span>
          </div>

          <input
            type="range"
            min="0.01"
            max="0.15"
            step="0.005"
            value={epsilon}
            onChange={(e) => setEpsilon(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
                       accent-yellow-500"
          />

          <div className="flex justify-between text-xs text-gray-500">
            <span>Sutil</span>
            <span>Forte</span>
          </div>

          <div className="flex items-start gap-2 p-2 bg-gray-800/50 rounded text-xs">
            <Info className="w-3 h-3 text-yellow-400 mt-0.5 flex-shrink-0" />
            <p className="text-gray-400">
              <span className="text-yellow-300">Quanto a imagem será alterada.</span>
              {' '}Valores baixos = alterações invisíveis. Valores altos = alterações visíveis mas mais eficaz.
            </p>
          </div>
        </div>

        {/* Velocidade de Ajuste */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Zap className="w-4 h-4 text-purple-400" />
              Velocidade de Ajuste
            </label>
            <span className="text-xs px-2 py-0.5 bg-purple-900/30 text-purple-400 rounded font-mono">
              {(alpha * 100).toFixed(1)}%
            </span>
          </div>

          <input
            type="range"
            min="0.001"
            max="0.03"
            step="0.001"
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
                       accent-purple-500"
          />

          <div className="flex justify-between text-xs text-gray-500">
            <span>Preciso</span>
            <span>Rápido</span>
          </div>

          <div className="flex items-start gap-2 p-2 bg-gray-800/50 rounded text-xs">
            <Info className="w-3 h-3 text-purple-400 mt-0.5 flex-shrink-0" />
            <p className="text-gray-400">
              <span className="text-purple-300">Tamanho de cada ajuste.</span>
              {' '}Valores baixos = ajustes pequenos e precisos. Valores altos = ajustes grandes e rápidos.
            </p>
          </div>
        </div>

        {/* Número de Repetições */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <RotateCw className="w-4 h-4 text-green-400" />
              Repetições
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={iterations}
              onChange={(e) => setIterations(parseInt(e.target.value) || 1)}
              className="w-16 px-2 py-1 bg-gray-800 border border-gray-600 rounded
                         text-gray-100 text-sm text-center
                         focus:outline-none focus:ring-2 focus:ring-green-500"
            />
          </div>

          <input
            type="range"
            min="1"
            max="50"
            step="1"
            value={iterations}
            onChange={(e) => setIterations(parseInt(e.target.value))}
            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer
                       accent-green-500"
          />

          <div className="flex justify-between text-xs text-gray-500">
            <span>Rápido</span>
            <span>Mais eficaz</span>
          </div>

          <div className="flex items-start gap-2 p-2 bg-gray-800/50 rounded text-xs">
            <Info className="w-3 h-3 text-green-400 mt-0.5 flex-shrink-0" />
            <p className="text-gray-400">
              <span className="text-green-300">Quantas vezes o ataque é refinado.</span>
              {' '}Mais repetições = ataque mais eficaz, mas demora mais. Recomendado: 5-20.
            </p>
          </div>
        </div>

        {/* Resumo */}
        <div className="p-3 bg-gray-900 rounded-lg border border-gray-700">
          <p className="text-xs text-gray-400 mb-2">Resumo da configuração:</p>
          <div className="grid grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-yellow-400 font-bold">{(epsilon * 100).toFixed(0)}%</div>
              <div className="text-xs text-gray-500">Força</div>
            </div>
            <div>
              <div className="text-purple-400 font-bold">{(alpha * 100).toFixed(1)}%</div>
              <div className="text-xs text-gray-500">Velocidade</div>
            </div>
            <div>
              <div className="text-green-400 font-bold">{iterations}x</div>
              <div className="text-xs text-gray-500">Repetições</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default HyperparameterControls
