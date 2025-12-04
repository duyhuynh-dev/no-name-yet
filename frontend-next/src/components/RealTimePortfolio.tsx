'use client';

/**
 * Real-Time Portfolio Display Component
 * 
 * Shows live portfolio metrics with WebSocket updates
 */

import React from 'react';
import { TrendingUp, TrendingDown, DollarSign, Activity, Target, Zap } from 'lucide-react';
import { usePortfolioUpdates } from '@/hooks/useWebSocket';

interface MetricCardProps {
  title: string;
  value: string;
  change?: string;
  changeType?: 'positive' | 'negative' | 'neutral';
  icon: React.ReactNode;
  subtitle?: string;
}

function MetricCard({ title, value, change, changeType = 'neutral', icon, subtitle }: MetricCardProps) {
  const changeColor = {
    positive: 'text-emerald-400',
    negative: 'text-red-400',
    neutral: 'text-slate-400',
  }[changeType];

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-4 hover:border-cyan-500/30 transition-all duration-300">
      <div className="flex items-center justify-between mb-3">
        <span className="text-slate-400 text-sm font-medium">{title}</span>
        <div className="p-2 bg-slate-700/50 rounded-lg">
          {icon}
        </div>
      </div>
      <div className="space-y-1">
        <p className="text-2xl font-bold text-white">{value}</p>
        {change && (
          <p className={`text-sm ${changeColor} flex items-center gap-1`}>
            {changeType === 'positive' ? <TrendingUp className="w-3 h-3" /> : 
             changeType === 'negative' ? <TrendingDown className="w-3 h-3" /> : null}
            {change}
          </p>
        )}
        {subtitle && (
          <p className="text-xs text-slate-500">{subtitle}</p>
        )}
      </div>
    </div>
  );
}

function PulsingDot() {
  return (
    <span className="relative flex h-2 w-2">
      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
      <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
    </span>
  );
}

export function RealTimePortfolio() {
  const portfolio = usePortfolioUpdates();

  // Default values when not connected
  const data = portfolio || {
    portfolio_value: 0,
    cash: 0,
    daily_pnl: 0,
    daily_pnl_pct: 0,
    total_pnl: 0,
    total_pnl_pct: 0,
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercent = (value: number) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  return (
    <div className="space-y-4">
      {/* Header with live indicator */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Activity className="w-5 h-5 text-cyan-400" />
          Portfolio Overview
        </h2>
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <PulsingDot />
          <span>Live</span>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <MetricCard
          title="Portfolio Value"
          value={formatCurrency(data.portfolio_value)}
          icon={<DollarSign className="w-4 h-4 text-cyan-400" />}
        />
        
        <MetricCard
          title="Available Cash"
          value={formatCurrency(data.cash)}
          icon={<Target className="w-4 h-4 text-emerald-400" />}
        />
        
        <MetricCard
          title="Daily P&L"
          value={formatCurrency(data.daily_pnl)}
          change={formatPercent(data.daily_pnl_pct)}
          changeType={data.daily_pnl >= 0 ? 'positive' : 'negative'}
          icon={data.daily_pnl >= 0 ? 
            <TrendingUp className="w-4 h-4 text-emerald-400" /> :
            <TrendingDown className="w-4 h-4 text-red-400" />
          }
        />
        
        <MetricCard
          title="Total P&L"
          value={formatCurrency(data.total_pnl)}
          change={formatPercent(data.total_pnl_pct)}
          changeType={data.total_pnl >= 0 ? 'positive' : 'negative'}
          icon={<Zap className="w-4 h-4 text-amber-400" />}
        />
        
        <MetricCard
          title="Invested"
          value={formatCurrency(data.portfolio_value - data.cash)}
          subtitle={`${((data.portfolio_value - data.cash) / data.portfolio_value * 100 || 0).toFixed(1)}% deployed`}
          icon={<Activity className="w-4 h-4 text-purple-400" />}
        />
        
        <MetricCard
          title="Daily Return"
          value={formatPercent(data.daily_pnl_pct)}
          changeType={data.daily_pnl_pct >= 0 ? 'positive' : 'negative'}
          icon={<TrendingUp className="w-4 h-4 text-blue-400" />}
        />
      </div>
    </div>
  );
}

