'use client';

import { useState, useEffect } from 'react';
import { Play, Loader2 } from 'lucide-react';
import { getSymbols } from '@/lib/api';

interface BacktestPanelProps {
  onRunBacktest: (symbol: string, initialBalance: number) => void;
  isLoading: boolean;
}

export default function BacktestPanel({ onRunBacktest, isLoading }: BacktestPanelProps) {
  const [symbols, setSymbols] = useState<string[]>(['SPY']);
  const [selectedSymbol, setSelectedSymbol] = useState('SPY');
  const [initialBalance, setInitialBalance] = useState(10000);

  useEffect(() => {
    const fetchSymbols = async () => {
      try {
        const syms = await getSymbols();
        if (syms.length > 0) {
          setSymbols(syms);
          setSelectedSymbol(syms[0]);
        }
      } catch (error) {
        console.error('Failed to fetch symbols:', error);
      }
    };
    fetchSymbols();
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onRunBacktest(selectedSymbol, initialBalance);
  };

  return (
    <div className="bg-slate-800/50 rounded-xl border border-slate-700/50 p-5">
      <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-4">
        Backtest
      </h3>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-xs text-slate-500 mb-1.5">Symbol</label>
          <select
            value={selectedSymbol}
            onChange={(e) => setSelectedSymbol(e.target.value)}
            className="w-full px-3 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white font-mono focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all"
          >
            {symbols.map((sym) => (
              <option key={sym} value={sym}>
                {sym}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-xs text-slate-500 mb-1.5">Initial Balance</label>
          <input
            type="number"
            value={initialBalance}
            onChange={(e) => setInitialBalance(Number(e.target.value))}
            min={1000}
            step={1000}
            className="w-full px-3 py-2.5 bg-slate-700/50 border border-slate-600 rounded-lg text-white font-mono focus:outline-none focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50 transition-all"
          />
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-gradient-to-r from-emerald-600 to-emerald-500 hover:from-emerald-500 hover:to-emerald-400 text-slate-900 font-semibold rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-emerald-500/20 hover:shadow-emerald-500/30"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Running...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Run Backtest
            </>
          )}
        </button>
      </form>
    </div>
  );
}

