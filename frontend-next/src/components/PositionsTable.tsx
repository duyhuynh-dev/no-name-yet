'use client';

/**
 * Real-Time Positions Table Component
 * 
 * Displays live position data with updates
 */

import React, { useMemo } from 'react';
import { TrendingUp, TrendingDown, Package } from 'lucide-react';
import { usePositionUpdates } from '@/hooks/useWebSocket';
import { PositionUpdate } from '@/lib/websocket';

interface PositionRowProps {
  position: PositionUpdate;
}

function PositionRow({ position }: PositionRowProps) {
  const isProfitable = position.unrealized_pnl >= 0;

  return (
    <tr className="border-b border-slate-700/50 hover:bg-slate-800/50 transition-colors">
      <td className="py-3 px-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-500/20 flex items-center justify-center">
            <span className="text-xs font-bold text-cyan-400">
              {position.symbol.slice(0, 2)}
            </span>
          </div>
          <div>
            <p className="font-medium text-white">{position.symbol}</p>
            <p className="text-xs text-slate-500">{position.quantity} shares</p>
          </div>
        </div>
      </td>
      
      <td className="py-3 px-4 text-right">
        <p className="text-white font-medium">${position.current_price.toFixed(2)}</p>
        <p className="text-xs text-slate-500">Avg: ${position.avg_cost.toFixed(2)}</p>
      </td>
      
      <td className="py-3 px-4 text-right">
        <p className="text-white font-medium">
          ${position.market_value.toLocaleString()}
        </p>
      </td>
      
      <td className="py-3 px-4 text-right">
        <div className={`flex items-center justify-end gap-1 ${
          isProfitable ? 'text-emerald-400' : 'text-red-400'
        }`}>
          {isProfitable ? 
            <TrendingUp className="w-4 h-4" /> : 
            <TrendingDown className="w-4 h-4" />
          }
          <span className="font-medium">
            ${Math.abs(position.unrealized_pnl).toLocaleString()}
          </span>
        </div>
        <p className={`text-xs ${isProfitable ? 'text-emerald-400' : 'text-red-400'}`}>
          {isProfitable ? '+' : ''}{position.unrealized_pnl_pct.toFixed(2)}%
        </p>
      </td>
      
      <td className="py-3 px-4">
        <div className="w-24 bg-slate-700 rounded-full h-2 overflow-hidden">
          <div 
            className={`h-full rounded-full transition-all duration-500 ${
              isProfitable ? 'bg-emerald-500' : 'bg-red-500'
            }`}
            style={{ 
              width: `${Math.min(Math.abs(position.unrealized_pnl_pct) * 5, 100)}%` 
            }}
          />
        </div>
      </td>
    </tr>
  );
}

export function PositionsTable() {
  const positionsMap = usePositionUpdates();
  
  const positions = useMemo(() => {
    return Array.from(positionsMap.values()).sort(
      (a, b) => Math.abs(b.unrealized_pnl) - Math.abs(a.unrealized_pnl)
    );
  }, [positionsMap]);

  const totalValue = useMemo(() => 
    positions.reduce((sum, p) => sum + p.market_value, 0), 
    [positions]
  );

  const totalPnL = useMemo(() => 
    positions.reduce((sum, p) => sum + p.unrealized_pnl, 0), 
    [positions]
  );

  if (positions.length === 0) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-8">
        <div className="flex flex-col items-center justify-center text-slate-400">
          <Package className="w-12 h-12 mb-4 opacity-50" />
          <p className="text-lg font-medium">No Positions</p>
          <p className="text-sm">Open positions will appear here</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700/50 flex items-center justify-between">
        <h3 className="text-white font-semibold flex items-center gap-2">
          <Package className="w-5 h-5 text-cyan-400" />
          Positions ({positions.length})
        </h3>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-slate-400">
            Total: <span className="text-white font-medium">${totalValue.toLocaleString()}</span>
          </span>
          <span className={totalPnL >= 0 ? 'text-emerald-400' : 'text-red-400'}>
            P&L: {totalPnL >= 0 ? '+' : ''}${totalPnL.toLocaleString()}
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-xs text-slate-400 uppercase tracking-wider bg-slate-800/30">
              <th className="py-3 px-4 text-left font-medium">Symbol</th>
              <th className="py-3 px-4 text-right font-medium">Price</th>
              <th className="py-3 px-4 text-right font-medium">Value</th>
              <th className="py-3 px-4 text-right font-medium">P&L</th>
              <th className="py-3 px-4 text-left font-medium">Performance</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((position) => (
              <PositionRow key={position.symbol} position={position} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

