/**
 * MetricsPanel Component
 * ======================
 *
 * Painel de métricas detalhadas do ataque adversário.
 * Exibe informações técnicas sobre a perturbação e seu impacto.
 */

import React from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts'
import { Activity, Target, Zap, CheckCircle, XCircle } from 'lucide-react'

function MetricsPanel({ metrics, params }) {
  // Prepara dados do histórico de loss para o gráfico
  const lossHistory = metrics.loss_history.map((loss, index) => ({
    iteration: index + 1,
    loss: loss.toFixed(4)
  }))

  return (
    <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
      <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <Activity className="w-5 h-5 text-pink-400" />
        Métricas do Ataque
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Métricas Numéricas */}
        <div className="space-y-4">
          {/* Status do Ataque */}
          <div className={`p-4 rounded-lg border ${
            metrics.attack_success
              ? 'bg-green-900/20 border-green-700/30'
              : 'bg-yellow-900/20 border-yellow-700/30'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              {metrics.attack_success ? (
                <CheckCircle className="w-5 h-5 text-green-400" />
              ) : (
                <XCircle className="w-5 h-5 text-yellow-400" />
              )}
              <span className={`font-semibold ${
                metrics.attack_success ? 'text-green-400' : 'text-yellow-400'
              }`}>
                {metrics.attack_success
                  ? 'Ataque Bem-Sucedido'
                  : 'Classificação Mantida'
                }
              </span>
            </div>
            <p className="text-sm text-gray-400">
              {metrics.attack_success
                ? 'O modelo foi enganado para classificar incorretamente.'
                : 'O modelo manteve a classificação correta (pode precisar de mais iterações ou maior ε).'
              }
            </p>
          </div>

          {/* Métricas de Perturbação */}
          <div className="grid grid-cols-2 gap-3">
            <MetricCard
              label="Norma L∞"
              value={metrics.perturbation_linf.toFixed(4)}
              subvalue={`${Math.round(metrics.perturbation_linf * 255)}/255`}
              icon={<Target className="w-4 h-4" />}
              color="blue"
              tooltip="Magnitude máxima de perturbação por pixel"
            />
            <MetricCard
              label="Norma L2"
              value={metrics.perturbation_l2.toFixed(2)}
              icon={<Zap className="w-4 h-4" />}
              color="purple"
              tooltip="Magnitude euclidiana total da perturbação"
            />
            <MetricCard
              label="Confiança Original"
              value={`${(metrics.original_confidence * 100).toFixed(1)}%`}
              icon={<Activity className="w-4 h-4" />}
              color="green"
              tooltip="Confiança na classe original antes do ataque"
            />
            <MetricCard
              label="Confiança Adversária"
              value={`${(metrics.adversarial_confidence * 100).toFixed(1)}%`}
              icon={<Activity className="w-4 h-4" />}
              color="red"
              tooltip="Confiança na classe top após o ataque"
            />
          </div>

          {/* Parâmetros Utilizados */}
          <div className="p-3 bg-gray-800/50 rounded-lg">
            <h4 className="text-sm font-medium text-gray-300 mb-2">
              Parâmetros Utilizados
            </h4>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="p-2 bg-gray-900 rounded">
                <span className="text-gray-500">ε =</span>
                <span className="text-yellow-400 ml-1 mono">
                  {params.epsilon.toFixed(4)}
                </span>
              </div>
              <div className="p-2 bg-gray-900 rounded">
                <span className="text-gray-500">α =</span>
                <span className="text-purple-400 ml-1 mono">
                  {params.alpha.toFixed(4)}
                </span>
              </div>
              <div className="p-2 bg-gray-900 rounded">
                <span className="text-gray-500">iter =</span>
                <span className="text-green-400 ml-1 mono">
                  {params.num_iterations}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Gráfico de Evolução da Loss */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-gray-300">
            Evolução da Loss Durante o Ataque
          </h4>
          <p className="text-xs text-gray-500">
            O PGD maximiza a loss iterativamente para confundir o modelo.
          </p>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={lossHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="iteration"
                  tick={{ fill: '#9CA3AF', fontSize: 11 }}
                  label={{
                    value: 'Iteração',
                    position: 'insideBottom',
                    offset: -5,
                    fill: '#6B7280',
                    fontSize: 11
                  }}
                />
                <YAxis
                  tick={{ fill: '#9CA3AF', fontSize: 11 }}
                  label={{
                    value: 'Loss',
                    angle: -90,
                    position: 'insideLeft',
                    fill: '#6B7280',
                    fontSize: 11
                  }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }}
                  labelStyle={{ color: '#9CA3AF' }}
                  formatter={(value) => [value, 'Loss']}
                />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#F59E0B"
                  strokeWidth={2}
                  dot={{ fill: '#F59E0B', r: 3 }}
                  activeDot={{ r: 5, fill: '#FBBF24' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-gray-500 text-center">
            Loss crescente indica que o ataque está maximizando o erro de classificação.
          </p>
        </div>
      </div>

      {/* Explicação Técnica */}
      <div className="mt-6 p-4 bg-gray-900 rounded-lg border border-gray-700">
        <h4 className="text-sm font-medium text-gray-300 mb-2 flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          Interpretação dos Resultados
        </h4>
        <div className="text-xs text-gray-400 space-y-2">
          <p>
            <span className="text-cyan-400">Norma L∞:</span> Representa a maior
            mudança individual em qualquer pixel. Um valor de {metrics.perturbation_linf.toFixed(4)}
            {' '}significa que nenhum pixel foi alterado mais que {Math.round(metrics.perturbation_linf * 255)}
            {' '}níveis (de 0-255).
          </p>
          <p>
            <span className="text-cyan-400">Norma L2:</span> Mede a magnitude
            total da perturbação. Útil para comparar diferentes ataques.
          </p>
          <p>
            <span className="text-cyan-400">Direção do Gradiente:</span> O PGD
            usa sign(∇L) para garantir que cada pixel seja perturbado na direção
            que mais aumenta a perda, maximizando o impacto dentro do orçamento ε.
          </p>
        </div>
      </div>
    </div>
  )
}

/**
 * Card de métrica individual
 */
function MetricCard({ label, value, subvalue, icon, color, tooltip }) {
  const colorClasses = {
    blue: 'text-blue-400 bg-blue-900/20 border-blue-700/30',
    purple: 'text-purple-400 bg-purple-900/20 border-purple-700/30',
    green: 'text-green-400 bg-green-900/20 border-green-700/30',
    red: 'text-red-400 bg-red-900/20 border-red-700/30',
    yellow: 'text-yellow-400 bg-yellow-900/20 border-yellow-700/30'
  }

  return (
    <div
      className={`p-3 rounded-lg border ${colorClasses[color]}`}
      title={tooltip}
    >
      <div className="flex items-center gap-1.5 text-xs text-gray-400 mb-1">
        {icon}
        <span>{label}</span>
      </div>
      <p className={`text-lg font-bold mono ${colorClasses[color].split(' ')[0]}`}>
        {value}
      </p>
      {subvalue && (
        <p className="text-xs text-gray-500">{subvalue}</p>
      )}
    </div>
  )
}

export default MetricsPanel
