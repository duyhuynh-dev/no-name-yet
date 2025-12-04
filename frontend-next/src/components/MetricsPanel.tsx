'use client';

import type { BacktestResponse } from '@/types';

interface MetricsPanelProps {
  results: BacktestResponse | null;
  horizontal?: boolean;
}

interface MetricCardProps {
  label: string;
  value: string;
  isPositive?: boolean;
  isNegative?: boolean;
  compact?: boolean;
}

function MetricCard({ label, value, isPositive, isNegative, compact }: MetricCardProps) {
  return (
    <div className={`bg-slate-700/30 rounded-lg ${compact ? 'p-3' : 'p-4'} text-center`}>
      <p className={`${compact ? 'text-[9px]' : 'text-[10px]'} font-medium uppercase tracking-wider text-slate-500 mb-1`}>
        {label}
      </p>
      <p
        className={`${compact ? 'text-base' : 'text-lg'} font-bold font-mono ${
          isPositive ? 'text-emerald-400' : isNegative ? 'text-red-400' : 'text-white'
        }`}
      >
        {value}
      </p>
    </div>
  );
}

export default function MetricsPanel({ results, horizontal = false }: MetricsPanelProps) {
  const totalReturn = results?.total_return ?? 0;
  const sharpeRatio = results?.sharpe_ratio ?? 0;
  const maxDrawdown = results?.max_drawdown ?? 0;
  const winRate = results?.win_rate ?? 0;
  const numTrades = results?.num_trades ?? 0;
  const finalBalance = results?.final_balance ?? 10000;

  if (horizontal) {
    return (
      <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 p-4">
        <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-3">
          Performance Metrics
        </h3>
        <div className="grid grid-cols-6 gap-2">
          <MetricCard
            label="Return"
            value={`${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(1)}%`}
            isPositive={totalReturn > 0}
            isNegative={totalReturn < 0}
            compact
          />
          <MetricCard label="Sharpe" value={sharpeRatio.toFixed(2)} compact />
          <MetricCard
            label="Drawdown"
            value={`${maxDrawdown.toFixed(1)}%`}
            isNegative={maxDrawdown > 5}
            compact
          />
          <MetricCard label="Win Rate" value={`${winRate.toFixed(0)}%`} compact />
          <MetricCard label="Trades" value={numTrades.toString()} compact />
          <MetricCard
            label="Balance"
            value={`$${(finalBalance / 1000).toFixed(1)}k`}
            isPositive={finalBalance > 10000}
            compact
          />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 p-5">
      <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-4">
        Performance Metrics
      </h3>

      <div className="grid grid-cols-2 gap-3">
        <MetricCard
          label="Total Return"
          value={`${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%`}
          isPositive={totalReturn > 0}
          isNegative={totalReturn < 0}
        />
        <MetricCard label="Sharpe Ratio" value={sharpeRatio.toFixed(3)} />
        <MetricCard
          label="Max Drawdown"
          value={`${maxDrawdown.toFixed(2)}%`}
          isNegative={maxDrawdown > 5}
        />
        <MetricCard label="Win Rate" value={`${winRate.toFixed(1)}%`} />
        <MetricCard label="Total Trades" value={numTrades.toString()} />
        <MetricCard
          label="Final Balance"
          value={`$${finalBalance.toLocaleString('en-US', { minimumFractionDigits: 2 })}`}
          isPositive={finalBalance > 10000}
        />
      </div>
    </div>
  );
}

