'use client';

/**
 * Agent Signals Component
 * 
 * Displays real-time signals from trading agents
 */

import React from 'react';
import { Bot, TrendingUp, TrendingDown, Minus, Activity } from 'lucide-react';
import { useAgentSignals } from '@/hooks/useWebSocket';
import { AgentSignal } from '@/lib/websocket';

const AGENT_COLORS: Record<string, string> = {
  momentum: 'from-blue-500 to-cyan-500',
  mean_reversion: 'from-purple-500 to-pink-500',
  breakout: 'from-amber-500 to-orange-500',
  market_maker: 'from-emerald-500 to-teal-500',
  ensemble: 'from-slate-500 to-slate-400',
};

interface SignalCardProps {
  signal: AgentSignal;
}

function SignalCard({ signal }: SignalCardProps) {
  const gradient = AGENT_COLORS[signal.agent_type] || AGENT_COLORS.ensemble;
  
  const ActionIcon = {
    buy: TrendingUp,
    sell: TrendingDown,
    hold: Minus,
  }[signal.action];

  const actionColor = {
    buy: 'text-emerald-400 bg-emerald-500/20',
    sell: 'text-red-400 bg-red-500/20',
    hold: 'text-slate-400 bg-slate-500/20',
  }[signal.action];

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50 hover:border-slate-600/50 transition-all">
      {/* Agent Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${gradient} flex items-center justify-center`}>
            <Bot className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-sm font-medium text-white capitalize">
              {signal.agent_type.replace('_', ' ')}
            </p>
            <p className="text-xs text-slate-500">{signal.agent_id}</p>
          </div>
        </div>
        <span className="text-xs text-slate-500">{formatTime(signal.timestamp)}</span>
      </div>

      {/* Signal Details */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${actionColor}`}>
            <ActionIcon className="w-4 h-4" />
          </div>
          <div>
            <p className="text-white font-medium">{signal.symbol}</p>
            <p className={`text-xs uppercase font-medium ${actionColor.split(' ')[0]}`}>
              {signal.action}
            </p>
          </div>
        </div>

        {/* Confidence */}
        <div className="text-right">
          <p className="text-xs text-slate-400 mb-1">Confidence</p>
          <div className="flex items-center gap-2">
            <div className="w-16 h-2 bg-slate-700 rounded-full overflow-hidden">
              <div 
                className={`h-full rounded-full transition-all duration-300 ${
                  signal.confidence > 0.7 ? 'bg-emerald-500' :
                  signal.confidence > 0.4 ? 'bg-amber-500' : 'bg-red-500'
                }`}
                style={{ width: `${signal.confidence * 100}%` }}
              />
            </div>
            <span className="text-sm font-medium text-white">
              {(signal.confidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export function AgentSignals() {
  const signals = useAgentSignals();

  // Group by agent type
  const latestByAgent = React.useMemo(() => {
    const latest = new Map<string, AgentSignal>();
    signals.forEach(signal => {
      if (!latest.has(signal.agent_id)) {
        latest.set(signal.agent_id, signal);
      }
    });
    return Array.from(latest.values());
  }, [signals]);

  if (signals.length === 0) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6">
        <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
          <Bot className="w-5 h-5 text-purple-400" />
          Agent Signals
        </h3>
        <div className="flex flex-col items-center justify-center py-8 text-slate-400">
          <Activity className="w-10 h-10 mb-3 opacity-50 animate-pulse" />
          <p className="text-sm">Awaiting agent signals...</p>
        </div>
      </div>
    );
  }

  // Stats
  const buySignals = signals.filter(s => s.action === 'buy').length;
  const sellSignals = signals.filter(s => s.action === 'sell').length;
  const avgConfidence = signals.reduce((sum, s) => sum + s.confidence, 0) / signals.length;

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700/50">
        <div className="flex items-center justify-between">
          <h3 className="text-white font-semibold flex items-center gap-2">
            <Bot className="w-5 h-5 text-purple-400" />
            Agent Signals
          </h3>
          <div className="flex items-center gap-3 text-xs">
            <span className="text-emerald-400">{buySignals} buy</span>
            <span className="text-red-400">{sellSignals} sell</span>
            <span className="text-slate-400">
              Avg: {(avgConfidence * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Latest Signals Grid */}
      <div className="p-4">
        <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-3">
          Latest by Agent
        </h4>
        <div className="grid gap-3">
          {latestByAgent.map(signal => (
            <SignalCard key={signal.agent_id} signal={signal} />
          ))}
        </div>
      </div>

      {/* Recent History */}
      <div className="px-4 pb-4">
        <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-3">
          Recent Signals
        </h4>
        <div className="space-y-2 max-h-[200px] overflow-y-auto">
          {signals.slice(0, 10).map((signal, index) => (
            <div 
              key={`${signal.agent_id}-${signal.timestamp}-${index}`}
              className="flex items-center justify-between py-2 border-b border-slate-700/30 last:border-0"
            >
              <div className="flex items-center gap-2">
                <span className={`
                  w-2 h-2 rounded-full
                  ${signal.action === 'buy' ? 'bg-emerald-500' : 
                    signal.action === 'sell' ? 'bg-red-500' : 'bg-slate-500'}
                `} />
                <span className="text-sm text-white">{signal.symbol}</span>
                <span className="text-xs text-slate-500 capitalize">
                  {signal.agent_type.replace('_', ' ')}
                </span>
              </div>
              <span className="text-xs text-slate-500">
                {new Date(signal.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

