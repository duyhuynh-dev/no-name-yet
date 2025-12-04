'use client';

/**
 * Agent Manager Component
 * 
 * UI for managing trading agents
 */

import React, { useState } from 'react';
import { 
  Bot, 
  Play, 
  Pause, 
  Settings, 
  TrendingUp, 
  TrendingDown,
  Activity,
  Zap,
  Target,
  BarChart2,
  RefreshCw,
} from 'lucide-react';

interface Agent {
  id: string;
  name: string;
  type: string;
  status: 'running' | 'paused' | 'stopped';
  metrics: {
    totalReturn: number;
    sharpeRatio: number;
    winRate: number;
    trades: number;
  };
  params: Record<string, number | string>;
}

const MOCK_AGENTS: Agent[] = [
  {
    id: 'momentum-1',
    name: 'Momentum Alpha',
    type: 'momentum',
    status: 'running',
    metrics: {
      totalReturn: 12.5,
      sharpeRatio: 1.8,
      winRate: 0.58,
      trades: 145,
    },
    params: {
      lookback: 20,
      threshold: 0.02,
      maxPosition: 0.1,
    },
  },
  {
    id: 'meanrev-1',
    name: 'Mean Reversion Pro',
    type: 'mean_reversion',
    status: 'running',
    metrics: {
      totalReturn: 8.3,
      sharpeRatio: 2.1,
      winRate: 0.65,
      trades: 89,
    },
    params: {
      window: 30,
      zScore: 2.0,
      holdPeriod: 5,
    },
  },
  {
    id: 'breakout-1',
    name: 'Breakout Hunter',
    type: 'breakout',
    status: 'paused',
    metrics: {
      totalReturn: -2.1,
      sharpeRatio: 0.4,
      winRate: 0.42,
      trades: 34,
    },
    params: {
      period: 20,
      atrMultiplier: 1.5,
      volumeFilter: true,
    },
  },
  {
    id: 'mm-1',
    name: 'Market Maker',
    type: 'market_maker',
    status: 'stopped',
    metrics: {
      totalReturn: 5.7,
      sharpeRatio: 1.5,
      winRate: 0.72,
      trades: 423,
    },
    params: {
      spread: 0.001,
      inventoryLimit: 100,
      fadeIntensity: 0.5,
    },
  },
];

const AGENT_COLORS: Record<string, string> = {
  momentum: 'from-blue-500 to-cyan-500',
  mean_reversion: 'from-purple-500 to-pink-500',
  breakout: 'from-amber-500 to-orange-500',
  market_maker: 'from-emerald-500 to-teal-500',
};

interface AgentCardProps {
  agent: Agent;
  onToggle: (id: string) => void;
  onSettings: (id: string) => void;
}

function AgentCard({ agent, onToggle, onSettings }: AgentCardProps) {
  const gradient = AGENT_COLORS[agent.type] || 'from-slate-500 to-slate-400';
  const isActive = agent.status === 'running';
  const isProfitable = agent.metrics.totalReturn >= 0;

  return (
    <div className={`
      bg-slate-800/50 rounded-xl border transition-all duration-300
      ${isActive 
        ? 'border-cyan-500/30 shadow-lg shadow-cyan-500/5' 
        : 'border-slate-700/50 opacity-75'}
    `}>
      {/* Header */}
      <div className="p-4 border-b border-slate-700/50">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${gradient} flex items-center justify-center`}>
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h4 className="font-medium text-white">{agent.name}</h4>
              <p className="text-xs text-slate-400 capitalize">
                {agent.type.replace('_', ' ')}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Status Indicator */}
            <span className={`
              flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium
              ${agent.status === 'running' 
                ? 'bg-emerald-500/20 text-emerald-400' 
                : agent.status === 'paused'
                ? 'bg-amber-500/20 text-amber-400'
                : 'bg-slate-500/20 text-slate-400'}
            `}>
              <span className={`w-1.5 h-1.5 rounded-full ${
                agent.status === 'running' ? 'bg-emerald-400 animate-pulse' :
                agent.status === 'paused' ? 'bg-amber-400' : 'bg-slate-400'
              }`} />
              {agent.status}
            </span>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="p-4 grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-slate-400 mb-1">Return</p>
          <p className={`text-lg font-bold flex items-center gap-1 ${
            isProfitable ? 'text-emerald-400' : 'text-red-400'
          }`}>
            {isProfitable ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
            {isProfitable ? '+' : ''}{agent.metrics.totalReturn.toFixed(1)}%
          </p>
        </div>

        <div>
          <p className="text-xs text-slate-400 mb-1">Sharpe</p>
          <p className="text-lg font-bold text-white flex items-center gap-1">
            <Zap className="w-4 h-4 text-amber-400" />
            {agent.metrics.sharpeRatio.toFixed(2)}
          </p>
        </div>

        <div>
          <p className="text-xs text-slate-400 mb-1">Win Rate</p>
          <p className="text-lg font-bold text-white flex items-center gap-1">
            <Target className="w-4 h-4 text-blue-400" />
            {(agent.metrics.winRate * 100).toFixed(0)}%
          </p>
        </div>

        <div>
          <p className="text-xs text-slate-400 mb-1">Trades</p>
          <p className="text-lg font-bold text-white flex items-center gap-1">
            <BarChart2 className="w-4 h-4 text-purple-400" />
            {agent.metrics.trades}
          </p>
        </div>
      </div>

      {/* Performance Bar */}
      <div className="px-4 pb-4">
        <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
          <span>Performance</span>
          <span>{agent.metrics.winRate * 100}% win rate</span>
        </div>
        <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
          <div 
            className={`h-full rounded-full transition-all duration-500 ${
              isProfitable ? 'bg-gradient-to-r from-emerald-500 to-cyan-500' : 'bg-red-500'
            }`}
            style={{ width: `${agent.metrics.winRate * 100}%` }}
          />
        </div>
      </div>

      {/* Actions */}
      <div className="p-4 border-t border-slate-700/50 flex items-center gap-2">
        <button
          onClick={() => onToggle(agent.id)}
          className={`
            flex-1 flex items-center justify-center gap-2 py-2 rounded-lg
            font-medium text-sm transition-all
            ${isActive 
              ? 'bg-amber-500/20 text-amber-400 hover:bg-amber-500/30' 
              : 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30'}
          `}
        >
          {isActive ? (
            <>
              <Pause className="w-4 h-4" />
              Pause
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Start
            </>
          )}
        </button>

        <button
          onClick={() => onSettings(agent.id)}
          className="p-2 rounded-lg bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
        >
          <Settings className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

export function AgentManager() {
  const [agents, setAgents] = useState<Agent[]>(MOCK_AGENTS);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);

  const handleToggle = (id: string) => {
    setAgents(prev => prev.map(agent => {
      if (agent.id === id) {
        return {
          ...agent,
          status: agent.status === 'running' ? 'paused' : 'running',
        };
      }
      return agent;
    }));
  };

  const handleSettings = (id: string) => {
    setSelectedAgent(id);
  };

  // Stats
  const runningAgents = agents.filter(a => a.status === 'running').length;
  const avgReturn = agents.reduce((sum, a) => sum + a.metrics.totalReturn, 0) / agents.length;
  const totalTrades = agents.reduce((sum, a) => sum + a.metrics.trades, 0);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <Bot className="w-6 h-6 text-purple-400" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-white">Agent Manager</h2>
              <p className="text-sm text-slate-400">
                {runningAgents} of {agents.length} agents running
              </p>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="text-right">
              <p className="text-xs text-slate-400">Avg Return</p>
              <p className={`text-lg font-bold ${avgReturn >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {avgReturn >= 0 ? '+' : ''}{avgReturn.toFixed(1)}%
              </p>
            </div>
            <div className="text-right">
              <p className="text-xs text-slate-400">Total Trades</p>
              <p className="text-lg font-bold text-white">{totalTrades}</p>
            </div>
            <button className="flex items-center gap-2 px-4 py-2 bg-cyan-500 text-white rounded-lg hover:bg-cyan-600 transition-colors">
              <RefreshCw className="w-4 h-4" />
              Sync
            </button>
          </div>
        </div>
      </div>

      {/* Agent Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
        {agents.map(agent => (
          <AgentCard
            key={agent.id}
            agent={agent}
            onToggle={handleToggle}
            onSettings={handleSettings}
          />
        ))}
      </div>

      {/* Performance Comparison */}
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-4">
        <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
          <Activity className="w-5 h-5 text-cyan-400" />
          Performance Comparison
        </h3>
        <div className="space-y-3">
          {agents
            .sort((a, b) => b.metrics.totalReturn - a.metrics.totalReturn)
            .map((agent, index) => (
              <div key={agent.id} className="flex items-center gap-4">
                <span className="text-sm font-medium text-slate-400 w-6">
                  #{index + 1}
                </span>
                <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${AGENT_COLORS[agent.type]} flex items-center justify-center`}>
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm text-white">{agent.name}</span>
                    <span className={`text-sm font-medium ${
                      agent.metrics.totalReturn >= 0 ? 'text-emerald-400' : 'text-red-400'
                    }`}>
                      {agent.metrics.totalReturn >= 0 ? '+' : ''}{agent.metrics.totalReturn.toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-500 bg-gradient-to-r ${AGENT_COLORS[agent.type]}`}
                      style={{ 
                        width: `${Math.max(0, Math.min(100, (agent.metrics.totalReturn + 10) * 5))}%` 
                      }}
                    />
                  </div>
                </div>
              </div>
            ))
          }
        </div>
      </div>
    </div>
  );
}

