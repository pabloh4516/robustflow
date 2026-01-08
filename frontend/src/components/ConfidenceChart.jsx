/**
 * ConfidenceChart Component
 * =========================
 *
 * Gráfico de barras comparativo mostrando as probabilidades
 * de classificação antes e depois do ataque PGD.
 *
 * Visualiza:
 * - Top-5 predições originais
 * - Top-5 predições após ataque
 * - Queda de confiança na classe correta
 * - Aumento de confiança em classes erradas
 */

import React from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts'
import { TrendingDown, TrendingUp, BarChart2 } from 'lucide-react'

function ConfidenceChart({ originalPredictions, adversarialPredictions }) {
  // Prepara dados para o gráfico comparativo
  const chartData = originalPredictions.map((orig, index) => {
    // Encontra a mesma classe nas predições adversárias
    const advPred = adversarialPredictions.find(
      adv => adv.class_idx === orig.class_idx
    )

    return {
      name: truncateLabel(orig.label, 15),
      fullLabel: orig.label,
      original: (orig.probability * 100).toFixed(1),
      adversarial: advPred ? (advPred.probability * 100).toFixed(1) : 0,
      classIdx: orig.class_idx
    }
  })

  // Adiciona classes que só aparecem após o ataque
  adversarialPredictions.forEach(adv => {
    const exists = chartData.find(d => d.classIdx === adv.class_idx)
    if (!exists && adv.probability > 0.01) {
      chartData.push({
        name: truncateLabel(adv.label, 15),
        fullLabel: adv.label,
        original: 0,
        adversarial: (adv.probability * 100).toFixed(1),
        classIdx: adv.class_idx
      })
    }
  })

  // Ordena por probabilidade original decrescente
  chartData.sort((a, b) => parseFloat(b.original) - parseFloat(a.original))

  // Calcula métricas de impacto
  const originalTop = originalPredictions[0]
  const adversarialTop = adversarialPredictions[0]
  const confidenceDrop = (
    (originalTop.probability -
      (adversarialPredictions.find(p => p.class_idx === originalTop.class_idx)?.probability || 0)
    ) * 100
  ).toFixed(1)

  return (
    <div className="bg-lab-dark rounded-lg border border-gray-700 p-4">
      <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
        <BarChart2 className="w-5 h-5 text-blue-400" />
        Análise de Confiança
      </h2>

      {/* Cards de resumo */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {/* Classe Original */}
        <div className="p-3 bg-green-900/20 border border-green-700/30 rounded-lg">
          <div className="flex items-center gap-2 text-green-400 text-sm mb-1">
            <TrendingDown className="w-4 h-4" />
            Classe Original
          </div>
          <p className="text-white font-medium truncate" title={originalTop.label}>
            {originalTop.label}
          </p>
          <p className="text-2xl font-bold text-green-400">
            {(originalTop.probability * 100).toFixed(1)}%
          </p>
        </div>

        {/* Classe Adversária */}
        <div className="p-3 bg-red-900/20 border border-red-700/30 rounded-lg">
          <div className="flex items-center gap-2 text-red-400 text-sm mb-1">
            <TrendingUp className="w-4 h-4" />
            Classe Adversária
          </div>
          <p className="text-white font-medium truncate" title={adversarialTop.label}>
            {adversarialTop.label}
          </p>
          <p className="text-2xl font-bold text-red-400">
            {(adversarialTop.probability * 100).toFixed(1)}%
          </p>
        </div>

        {/* Queda de Confiança */}
        <div className="p-3 bg-yellow-900/20 border border-yellow-700/30 rounded-lg">
          <div className="flex items-center gap-2 text-yellow-400 text-sm mb-1">
            <TrendingDown className="w-4 h-4" />
            Queda de Confiança
          </div>
          <p className="text-white font-medium">
            Na classe original
          </p>
          <p className="text-2xl font-bold text-yellow-400">
            -{confidenceDrop}%
          </p>
        </div>
      </div>

      {/* Gráfico de Barras */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
            barCategoryGap="20%"
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="name"
              tick={{ fill: '#9CA3AF', fontSize: 11 }}
              angle={-45}
              textAnchor="end"
              height={80}
              interval={0}
            />
            <YAxis
              tick={{ fill: '#9CA3AF', fontSize: 12 }}
              domain={[0, 100]}
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }}
            />
            <Legend
              wrapperStyle={{ paddingTop: '20px' }}
              formatter={(value) => (
                <span style={{ color: '#9CA3AF' }}>
                  {value === 'original' ? 'Antes do Ataque' : 'Após Ataque'}
                </span>
              )}
            />
            <Bar
              dataKey="original"
              name="original"
              fill="#10B981"
              radius={[4, 4, 0, 0]}
            />
            <Bar
              dataKey="adversarial"
              name="adversarial"
              fill="#EF4444"
              radius={[4, 4, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Explicação */}
      <div className="mt-4 p-3 bg-gray-800/50 rounded-lg text-xs text-gray-400">
        <p>
          <span className="text-green-400">■</span> Barras verdes mostram a confiança
          do modelo antes do ataque.
          <span className="text-red-400 ml-2">■</span> Barras vermelhas mostram como
          as probabilidades mudaram após a perturbação adversária.
        </p>
      </div>
    </div>
  )
}

/**
 * Tooltip customizado para o gráfico
 */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload) return null

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-lg">
      <p className="text-white font-medium mb-2">
        {payload[0]?.payload?.fullLabel || label}
      </p>
      {payload.map((entry, index) => (
        <p key={index} className="text-sm" style={{ color: entry.color }}>
          {entry.name === 'original' ? 'Antes' : 'Após'}: {entry.value}%
        </p>
      ))}
      {payload.length === 2 && (
        <p className="text-xs text-gray-400 mt-1 border-t border-gray-700 pt-1">
          Δ = {(parseFloat(payload[1].value) - parseFloat(payload[0].value)).toFixed(1)}%
        </p>
      )}
    </div>
  )
}

/**
 * Trunca label para caber no gráfico
 */
function truncateLabel(label, maxLength) {
  if (label.length <= maxLength) return label
  return label.substring(0, maxLength - 2) + '...'
}

export default ConfidenceChart
