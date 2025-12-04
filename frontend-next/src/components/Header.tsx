'use client';

import { useEffect, useState } from 'react';
import { Activity, Zap } from 'lucide-react';
import { checkHealth } from '@/lib/api';

interface HeaderProps {
  latency: number | null;
}

export default function Header({ latency }: HeaderProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [modelName, setModelName] = useState<string | null>(null);

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const health = await checkHealth();
        setIsConnected(health.status === 'healthy');
        setModelName(health.model_name);
      } catch {
        setIsConnected(false);
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="flex justify-between items-center px-6 py-4 bg-slate-900/80 border-b border-slate-700/50 backdrop-blur-sm">
      <div className="flex items-center gap-3">
        <div className="text-2xl text-emerald-400 animate-pulse">◈</div>
        <h1 className="text-xl font-semibold tracking-wide">
          HFT<span className="text-emerald-400">RL</span>
        </h1>
      </div>

      <div className="flex items-center gap-4">
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${
            isConnected
              ? 'bg-emerald-500/10 border-emerald-500/30'
              : 'bg-red-500/10 border-red-500/30'
          }`}
        >
          <Activity
            className={`w-4 h-4 ${isConnected ? 'text-emerald-400' : 'text-red-400'}`}
          />
          <span className="text-sm font-mono">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 rounded-lg border border-slate-700">
          <Zap className="w-4 h-4 text-emerald-400" />
          <span className="text-sm font-mono text-emerald-400">
            {latency !== null ? `${latency.toFixed(1)}ms` : '--ms'}
          </span>
        </div>

        {modelName && (
          <div className="hidden md:block text-xs text-slate-500 font-mono">
            Model: {modelName}
          </div>
        )}
      </div>
    </header>
  );
}

