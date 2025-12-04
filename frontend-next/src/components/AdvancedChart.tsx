'use client';

/**
 * Advanced Chart Component
 * 
 * Interactive charts with technical indicators and drawing tools
 */

import React, { useState, useMemo, useCallback, useRef } from 'react';
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  BarElement,
  Title, 
  Tooltip, 
  Legend,
  Filler,
  ChartOptions,
  ChartData,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';
import { 
  TrendingUp, 
  Settings, 
  Maximize2,
  Layers,
  BarChart2,
  LineChartIcon,
  CandlestickChart,
  Clock,
  Crosshair,
} from 'lucide-react';

ChartJS.register(
  CategoryScale, 
  LinearScale, 
  PointElement, 
  LineElement, 
  BarElement, 
  Title, 
  Tooltip, 
  Legend,
  Filler
);

interface OHLCV {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Indicator {
  name: string;
  type: 'overlay' | 'separate';
  color: string;
  data: number[];
}

interface AdvancedChartProps {
  data: OHLCV[];
  indicators?: Indicator[];
  symbol?: string;
  interval?: string;
}

type ChartType = 'line' | 'candle' | 'area';
type TimeFrame = '1m' | '5m' | '15m' | '1h' | '4h' | '1d';

const TIMEFRAMES: { label: string; value: TimeFrame }[] = [
  { label: '1M', value: '1m' },
  { label: '5M', value: '5m' },
  { label: '15M', value: '15m' },
  { label: '1H', value: '1h' },
  { label: '4H', value: '4h' },
  { label: '1D', value: '1d' },
];

const INDICATORS = [
  { id: 'sma20', name: 'SMA 20', type: 'overlay' },
  { id: 'sma50', name: 'SMA 50', type: 'overlay' },
  { id: 'ema20', name: 'EMA 20', type: 'overlay' },
  { id: 'bb', name: 'Bollinger Bands', type: 'overlay' },
  { id: 'rsi', name: 'RSI', type: 'separate' },
  { id: 'macd', name: 'MACD', type: 'separate' },
  { id: 'volume', name: 'Volume', type: 'separate' },
];

export function AdvancedChart({ 
  data, 
  indicators = [],
  symbol = 'AAPL',
  interval = '1H',
}: AdvancedChartProps) {
  const [chartType, setChartType] = useState<ChartType>('line');
  const [timeframe, setTimeframe] = useState<TimeFrame>('1h');
  const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['sma20', 'volume']);
  const [showIndicatorPanel, setShowIndicatorPanel] = useState(false);
  const [crosshairEnabled, setCrosshairEnabled] = useState(true);
  const chartRef = useRef(null);

  // Calculate price change
  const priceChange = useMemo(() => {
    if (data.length < 2) return { value: 0, percent: 0 };
    const first = data[0].close;
    const last = data[data.length - 1].close;
    return {
      value: last - first,
      percent: ((last - first) / first) * 100,
    };
  }, [data]);

  // Generate chart data
  const chartData = useMemo((): ChartData<'line'> => {
    const labels = data.map(d => {
      const date = new Date(d.time);
      return date.toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
      });
    });

    const datasets = [
      {
        label: symbol,
        data: data.map(d => d.close),
        borderColor: priceChange.value >= 0 ? '#10b981' : '#ef4444',
        backgroundColor: chartType === 'area' 
          ? priceChange.value >= 0 
            ? 'rgba(16, 185, 129, 0.1)' 
            : 'rgba(239, 68, 68, 0.1)'
          : 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        fill: chartType === 'area',
        tension: 0.1,
      },
    ];

    // Add overlay indicators
    const overlayIndicators = indicators.filter(ind => 
      ind.type === 'overlay' && selectedIndicators.includes(ind.name.toLowerCase())
    );

    overlayIndicators.forEach(ind => {
      datasets.push({
        label: ind.name,
        data: ind.data,
        borderColor: ind.color,
        backgroundColor: 'transparent',
        borderWidth: 1,
        pointRadius: 0,
        pointHoverRadius: 0,
        fill: false,
        tension: 0,
        borderDash: [5, 5],
      } as typeof datasets[0]);
    });

    return { labels, datasets };
  }, [data, symbol, chartType, priceChange, indicators, selectedIndicators]);

  // Chart options
  const chartOptions = useMemo((): ChartOptions<'line'> => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      intersect: false,
      mode: 'index',
    },
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: crosshairEnabled,
        backgroundColor: 'rgba(15, 23, 42, 0.9)',
        titleColor: '#f8fafc',
        bodyColor: '#cbd5e1',
        borderColor: 'rgba(100, 116, 139, 0.3)',
        borderWidth: 1,
        padding: 12,
        displayColors: true,
        callbacks: {
          label: (context) => {
            const price = context.raw as number;
            return `${context.dataset.label}: $${price.toFixed(2)}`;
          },
        },
      },
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: false,
        },
        ticks: {
          color: '#64748b',
          maxTicksLimit: 8,
          font: { size: 10 },
        },
      },
      y: {
        display: true,
        position: 'right',
        grid: {
          color: 'rgba(100, 116, 139, 0.1)',
        },
        ticks: {
          color: '#64748b',
          font: { size: 10 },
          callback: (value) => `$${value}`,
        },
      },
    },
  }), [crosshairEnabled]);

  // Volume chart data
  const volumeData = useMemo((): ChartData<'bar'> => ({
    labels: data.map(d => {
      const date = new Date(d.time);
      return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    }),
    datasets: [{
      label: 'Volume',
      data: data.map(d => d.volume),
      backgroundColor: data.map((d, i) => {
        if (i === 0) return 'rgba(100, 116, 139, 0.5)';
        return d.close >= data[i - 1].close 
          ? 'rgba(16, 185, 129, 0.5)' 
          : 'rgba(239, 68, 68, 0.5)';
      }),
      borderWidth: 0,
    }],
  }), [data]);

  const volumeOptions: ChartOptions<'bar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: { enabled: false },
    },
    scales: {
      x: { display: false },
      y: {
        display: true,
        position: 'right',
        grid: { display: false },
        ticks: {
          color: '#64748b',
          font: { size: 9 },
          callback: (value) => {
            const num = value as number;
            if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
            if (num >= 1000) return `${(num / 1000).toFixed(0)}K`;
            return value.toString();
          },
        },
      },
    },
  };

  const toggleIndicator = useCallback((id: string) => {
    setSelectedIndicators(prev => 
      prev.includes(id) 
        ? prev.filter(i => i !== id)
        : [...prev, id]
    );
  }, []);

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700/50">
        <div className="flex items-center justify-between">
          {/* Symbol Info */}
          <div className="flex items-center gap-4">
            <div>
              <h3 className="text-lg font-bold text-white">{symbol}</h3>
              <p className="text-xs text-slate-400">{interval} Chart</p>
            </div>
            <div className={`text-right ${priceChange.value >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              <p className="text-lg font-bold">
                ${data[data.length - 1]?.close.toFixed(2) || '0.00'}
              </p>
              <p className="text-xs flex items-center gap-1">
                <TrendingUp className="w-3 h-3" />
                {priceChange.value >= 0 ? '+' : ''}{priceChange.percent.toFixed(2)}%
              </p>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            {/* Timeframe Selector */}
            <div className="flex bg-slate-700/50 rounded-lg p-0.5">
              {TIMEFRAMES.map(tf => (
                <button
                  key={tf.value}
                  onClick={() => setTimeframe(tf.value)}
                  className={`
                    px-2 py-1 text-xs font-medium rounded-md transition-all
                    ${timeframe === tf.value 
                      ? 'bg-cyan-500 text-white' 
                      : 'text-slate-400 hover:text-white'}
                  `}
                >
                  {tf.label}
                </button>
              ))}
            </div>

            {/* Chart Type */}
            <div className="flex bg-slate-700/50 rounded-lg p-0.5">
              <button
                onClick={() => setChartType('line')}
                className={`p-1.5 rounded-md transition-all ${
                  chartType === 'line' ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:text-white'
                }`}
                title="Line Chart"
              >
                <LineChartIcon className="w-4 h-4" />
              </button>
              <button
                onClick={() => setChartType('candle')}
                className={`p-1.5 rounded-md transition-all ${
                  chartType === 'candle' ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:text-white'
                }`}
                title="Candlestick"
              >
                <CandlestickChart className="w-4 h-4" />
              </button>
              <button
                onClick={() => setChartType('area')}
                className={`p-1.5 rounded-md transition-all ${
                  chartType === 'area' ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:text-white'
                }`}
                title="Area Chart"
              >
                <BarChart2 className="w-4 h-4" />
              </button>
            </div>

            {/* Tools */}
            <button
              onClick={() => setCrosshairEnabled(!crosshairEnabled)}
              className={`p-2 rounded-lg transition-all ${
                crosshairEnabled ? 'bg-cyan-500/20 text-cyan-400' : 'bg-slate-700/50 text-slate-400'
              }`}
              title="Crosshair"
            >
              <Crosshair className="w-4 h-4" />
            </button>

            <button
              onClick={() => setShowIndicatorPanel(!showIndicatorPanel)}
              className={`p-2 rounded-lg transition-all ${
                showIndicatorPanel ? 'bg-cyan-500/20 text-cyan-400' : 'bg-slate-700/50 text-slate-400'
              }`}
              title="Indicators"
            >
              <Layers className="w-4 h-4" />
            </button>

            <button className="p-2 rounded-lg bg-slate-700/50 text-slate-400 hover:text-white transition-all">
              <Settings className="w-4 h-4" />
            </button>

            <button className="p-2 rounded-lg bg-slate-700/50 text-slate-400 hover:text-white transition-all">
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Indicator Panel */}
      {showIndicatorPanel && (
        <div className="px-4 py-2 border-b border-slate-700/50 bg-slate-800/30">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs text-slate-400 mr-2">Indicators:</span>
            {INDICATORS.map(ind => (
              <button
                key={ind.id}
                onClick={() => toggleIndicator(ind.id)}
                className={`
                  px-2 py-1 text-xs rounded-md transition-all
                  ${selectedIndicators.includes(ind.id)
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                    : 'bg-slate-700/50 text-slate-400 border border-transparent hover:border-slate-600'}
                `}
              >
                {ind.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Main Chart */}
      <div className="p-4">
        <div className="h-[350px]">
          <Line ref={chartRef} data={chartData} options={chartOptions} />
        </div>
      </div>

      {/* Volume Chart */}
      {selectedIndicators.includes('volume') && (
        <div className="px-4 pb-4">
          <div className="h-[80px]">
            <Bar data={volumeData} options={volumeOptions} />
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="px-4 py-2 border-t border-slate-700/50 bg-slate-800/30">
        <div className="flex items-center justify-between text-xs text-slate-400">
          <div className="flex items-center gap-4">
            <span>O: ${data[data.length - 1]?.open.toFixed(2) || '-'}</span>
            <span>H: ${data[data.length - 1]?.high.toFixed(2) || '-'}</span>
            <span>L: ${data[data.length - 1]?.low.toFixed(2) || '-'}</span>
            <span>C: ${data[data.length - 1]?.close.toFixed(2) || '-'}</span>
          </div>
          <div className="flex items-center gap-1">
            <Clock className="w-3 h-3" />
            <span>
              {new Date(data[data.length - 1]?.time || Date.now()).toLocaleString()}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

