'use client';

/**
 * Real-Time Trade Feed Component
 * 
 * Displays live trade executions
 */

import React from 'react';
import { ArrowUpCircle, ArrowDownCircle, Clock, Zap } from 'lucide-react';
import { useTradeUpdates } from '@/hooks/useWebSocket';
import { TradeExecuted } from '@/lib/websocket';

interface TradeItemProps {
  trade: TradeExecuted;
  isNew?: boolean;
}

function TradeItem({ trade, isNew = false }: TradeItemProps) {
  const isBuy = trade.side === 'buy';
  
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  return (
    <div 
      className={`
        flex items-center gap-3 p-3 rounded-lg transition-all duration-300
        ${isNew ? 'bg-cyan-500/10 border border-cyan-500/30' : 'bg-slate-800/30 border border-transparent'}
        hover:bg-slate-700/50
      `}
    >
      {/* Direction Icon */}
      <div className={`
        p-2 rounded-lg
        ${isBuy ? 'bg-emerald-500/20' : 'bg-red-500/20'}
      `}>
        {isBuy ? (
          <ArrowUpCircle className="w-5 h-5 text-emerald-400" />
        ) : (
          <ArrowDownCircle className="w-5 h-5 text-red-400" />
        )}
      </div>

      {/* Trade Details */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-white">{trade.symbol}</span>
          <span className={`
            text-xs px-2 py-0.5 rounded-full font-medium
            ${isBuy ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}
          `}>
            {trade.side.toUpperCase()}
          </span>
          {isNew && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-400 animate-pulse">
              NEW
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-400 mt-1">
          <span>{trade.quantity} @ ${trade.price.toFixed(2)}</span>
          <span>•</span>
          <span>${(trade.quantity * trade.price).toLocaleString()}</span>
        </div>
      </div>

      {/* Timestamp */}
      <div className="text-right">
        <div className="flex items-center gap-1 text-xs text-slate-400">
          <Clock className="w-3 h-3" />
          {formatTime(trade.executed_at)}
        </div>
        {trade.commission > 0 && (
          <p className="text-xs text-slate-500 mt-1">
            Fee: ${trade.commission.toFixed(2)}
          </p>
        )}
      </div>
    </div>
  );
}

export function TradeFeed() {
  const trades = useTradeUpdates(20);

  if (trades.length === 0) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6">
        <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
          <Zap className="w-5 h-5 text-amber-400" />
          Trade Feed
        </h3>
        <div className="flex flex-col items-center justify-center py-8 text-slate-400">
          <Clock className="w-10 h-10 mb-3 opacity-50" />
          <p className="text-sm">Waiting for trades...</p>
        </div>
      </div>
    );
  }

  // Calculate stats
  const buyCount = trades.filter(t => t.side === 'buy').length;
  const sellCount = trades.filter(t => t.side === 'sell').length;
  const totalVolume = trades.reduce((sum, t) => sum + t.quantity * t.price, 0);

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700/50">
        <div className="flex items-center justify-between">
          <h3 className="text-white font-semibold flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-400" />
            Trade Feed
          </h3>
          <div className="flex items-center gap-3 text-xs">
            <span className="text-emerald-400">{buyCount} buys</span>
            <span className="text-red-400">{sellCount} sells</span>
            <span className="text-slate-400">
              Vol: ${totalVolume.toLocaleString()}
            </span>
          </div>
        </div>
      </div>

      {/* Trade List */}
      <div className="max-h-[400px] overflow-y-auto">
        <div className="p-3 space-y-2">
          {trades.map((trade, index) => (
            <TradeItem 
              key={trade.trade_id} 
              trade={trade} 
              isNew={index === 0}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

