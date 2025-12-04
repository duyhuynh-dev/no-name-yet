'use client';

import type { Trade } from '@/types';

interface TradesPanelProps {
  trades: Trade[];
}

export default function TradesPanel({ trades }: TradesPanelProps) {
  const recentTrades = trades.slice(-10).reverse();

  return (
    <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 p-5">
      <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-4">
        Recent Trades
      </h3>

      <div className="max-h-48 overflow-y-auto space-y-2">
        {recentTrades.length === 0 ? (
          <p className="text-sm text-slate-500 text-center py-4">No trades yet</p>
        ) : (
          recentTrades.map((trade, index) => {
            const isBuy = trade.type.includes('buy');
            const pnl = trade.pnl;
            return (
              <div
                key={index}
                className="flex items-center justify-between px-3 py-2 bg-slate-700/30 rounded-lg"
              >
                <span
                  className={`text-xs font-semibold uppercase tracking-wide ${
                    isBuy ? 'text-emerald-400' : 'text-red-400'
                  }`}
                >
                  {trade.type.replace('_', ' ')}
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  ${trade.price.toFixed(2)}
                </span>
                {pnl !== undefined && pnl !== 0 && (
                  <span
                    className={`text-xs font-mono font-semibold ${
                      pnl >= 0 ? 'text-emerald-400' : 'text-red-400'
                    }`}
                  >
                    {pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}
                  </span>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

