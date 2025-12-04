'use client';

import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { PredictionResponse, ActionType } from '@/types';

interface PredictionPanelProps {
  prediction: PredictionResponse | null;
  isLoading: boolean;
}

const actionConfig: Record<ActionType, { icon: typeof TrendingUp; color: string; bgColor: string }> = {
  buy: {
    icon: TrendingUp,
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-500/15 border-emerald-500/40',
  },
  sell: {
    icon: TrendingDown,
    color: 'text-red-400',
    bgColor: 'bg-red-500/15 border-red-500/40',
  },
  hold: {
    icon: Minus,
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/15 border-amber-500/40',
  },
};

export default function PredictionPanel({ prediction, isLoading }: PredictionPanelProps) {
  const action = prediction?.action || 'hold';
  const config = actionConfig[action];
  const Icon = config.icon;

  return (
    <div className="bg-slate-800/50 rounded-xl border border-emerald-500/20 p-5">
      <h3 className="text-xs font-semibold uppercase tracking-widest text-slate-400 mb-4">
        Live Prediction
      </h3>

      <div className="text-center">
        {/* Action Display */}
        <div
          className={`inline-flex items-center gap-3 px-6 py-3 rounded-xl border-2 mb-4 transition-all ${
            isLoading ? 'animate-pulse bg-slate-700/50 border-slate-600' : config.bgColor
          }`}
        >
          <Icon className={`w-6 h-6 ${isLoading ? 'text-slate-500' : config.color}`} />
          <span
            className={`text-lg font-bold uppercase tracking-wider font-mono ${
              isLoading ? 'text-slate-500' : config.color
            }`}
          >
            {isLoading ? '...' : action.toUpperCase()}
          </span>
        </div>

        {/* Confidence Bar */}
        <div className="mb-2">
          <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400 transition-all duration-500"
              style={{ width: `${(prediction?.confidence || 0.5) * 100}%` }}
            />
          </div>
        </div>
        <p className="text-sm text-slate-400">
          Confidence:{' '}
          <span className="text-emerald-400 font-mono font-semibold">
            {prediction ? `${(prediction.confidence * 100).toFixed(1)}%` : '--'}
          </span>
        </p>
      </div>

      {/* Probabilities */}
      <div className="mt-5 pt-4 border-t border-slate-700/50 space-y-2.5">
        {(['hold', 'buy', 'sell'] as ActionType[]).map((act) => {
          const prob = prediction?.probabilities[act] || 0.33;
          const actConfig = actionConfig[act];
          return (
            <div key={act} className="flex items-center gap-3">
              <span className="w-10 text-xs text-slate-500 capitalize">{act}</span>
              <div className="flex-1 h-1 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${
                    act === 'buy'
                      ? 'bg-emerald-500'
                      : act === 'sell'
                      ? 'bg-red-500'
                      : 'bg-amber-500'
                  }`}
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
              <span className="w-12 text-xs text-slate-500 font-mono text-right">
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

