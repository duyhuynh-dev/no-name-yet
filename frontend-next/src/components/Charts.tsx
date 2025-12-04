'use client';

import { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

interface ChartProps {
  title: string;
  data: number[];
  labels?: string[];
  yAxisPrefix?: string;
  changePercent?: number;
}

export default function Chart({
  title,
  data,
  labels,
  yAxisPrefix = '',
  changePercent,
}: ChartProps) {
  const chartRef = useRef<ChartJS<'line'>>(null);

  const chartLabels = labels || data.map((_, i) => `${i}`);

  const chartData = {
    labels: chartLabels,
    datasets: [
      {
        data,
        borderColor: '#10b981',
        backgroundColor: 'rgba(16, 185, 129, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        borderWidth: 2,
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#1e293b',
        titleColor: '#f1f5f9',
        bodyColor: '#94a3b8',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        displayColors: false,
        callbacks: {
          label: (ctx: { raw: number }) =>
            `${yAxisPrefix}${ctx.raw.toLocaleString('en-US', { minimumFractionDigits: 2 })}`,
        },
      },
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#64748b', maxTicksLimit: 10 },
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: {
          color: '#64748b',
          callback: (value: number | string) =>
            `${yAxisPrefix}${typeof value === 'number' ? value.toLocaleString() : value}`,
        },
      },
    },
    interaction: {
      intersect: false,
      mode: 'index' as const,
    },
  };

  return (
    <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 p-5">
      <div className="flex justify-between items-center mb-4">
        <h2 className="font-medium text-white">{title}</h2>
        {changePercent !== undefined && (
          <span
            className={`text-sm font-mono font-semibold px-2 py-0.5 rounded ${
              changePercent >= 0
                ? 'text-emerald-400 bg-emerald-500/10'
                : 'text-red-400 bg-red-500/10'
            }`}
          >
            {changePercent >= 0 ? '+' : ''}
            {changePercent.toFixed(2)}%
          </span>
        )}
      </div>
      <div className="h-52 w-full">
        <Line ref={chartRef} data={chartData} options={options} />
      </div>
    </div>
  );
}

