'use client';

/**
 * Enhanced Dashboard Page
 * 
 * Real-time trading dashboard with WebSocket support
 */

import React, { useState, useMemo } from 'react';
import { 
  LayoutDashboard, 
  Bot, 
  Settings, 
  BarChart2,
  Activity,
  Bell,
} from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { 
  RealTimePortfolio, 
  PositionsTable, 
  TradeFeed, 
  AgentSignals,
  AdvancedChart,
  AlertsPanel,
  AgentManager,
  SettingsPanel,
} from '@/components';

type TabId = 'overview' | 'agents' | 'charts' | 'settings';

// Generate mock OHLCV data for demo
const generateMockOHLCV = (count: number) => {
  const data = [];
  let price = 150;
  const now = Date.now();
  
  for (let i = 0; i < count; i++) {
    const change = (Math.random() - 0.5) * 5;
    const open = price;
    const close = price + change;
    const high = Math.max(open, close) + Math.random() * 2;
    const low = Math.min(open, close) - Math.random() * 2;
    const volume = Math.floor(Math.random() * 1000000) + 100000;
    
    data.push({
      time: new Date(now - (count - i) * 3600000).toISOString(),
      open,
      high,
      low,
      close,
      volume,
    });
    
    price = close;
  }
  
  return data;
};

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const { isConnected, connectionState } = useWebSocket({ autoConnect: true });
  
  const chartData = useMemo(() => generateMockOHLCV(100), []);

  const tabs = [
    { id: 'overview' as TabId, label: 'Overview', icon: LayoutDashboard },
    { id: 'agents' as TabId, label: 'Agents', icon: Bot },
    { id: 'charts' as TabId, label: 'Charts', icon: BarChart2 },
    { id: 'settings' as TabId, label: 'Settings', icon: Settings },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Top Navigation */}
      <nav className="border-b border-slate-800/50 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-[1800px] mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            {/* Logo & Tabs */}
            <div className="flex items-center gap-8">
              <div className="flex items-center gap-2">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                  <Activity className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-white">HFT Platform</h1>
                  <p className="text-xs text-slate-400">v2.0 Enhanced</p>
                </div>
              </div>

              <div className="flex items-center gap-1 bg-slate-800/50 rounded-lg p-1">
                {tabs.map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`
                      flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium
                      transition-all duration-200
                      ${activeTab === tab.id 
                        ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/25' 
                        : 'text-slate-400 hover:text-white hover:bg-slate-700/50'}
                    `}
                  >
                    <tab.icon className="w-4 h-4" />
                    {tab.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Status & Alerts */}
            <div className="flex items-center gap-4">
              <div className={`
                flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium
                ${isConnected 
                  ? 'bg-emerald-500/20 text-emerald-400' 
                  : 'bg-red-500/20 text-red-400'}
              `}>
                <span className={`w-2 h-2 rounded-full ${
                  isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'
                }`} />
                {connectionState}
              </div>

              <button className="relative p-2 rounded-lg bg-slate-800/50 text-slate-400 hover:text-white transition-colors">
                <Bell className="w-5 h-5" />
                <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full" />
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-[1800px] mx-auto p-4">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Portfolio Overview */}
            <RealTimePortfolio />

            {/* Charts & Positions */}
            <div className="grid grid-cols-12 gap-6">
              {/* Main Chart */}
              <div className="col-span-12 xl:col-span-8">
                <AdvancedChart 
                  data={chartData} 
                  symbol="AAPL" 
                  interval="1H" 
                />
              </div>

              {/* Sidebar */}
              <div className="col-span-12 xl:col-span-4 space-y-6">
                <AlertsPanel />
                <AgentSignals />
              </div>
            </div>

            {/* Bottom Section */}
            <div className="grid grid-cols-12 gap-6">
              {/* Positions */}
              <div className="col-span-12 lg:col-span-7">
                <PositionsTable />
              </div>

              {/* Trade Feed */}
              <div className="col-span-12 lg:col-span-5">
                <TradeFeed />
              </div>
            </div>
          </div>
        )}

        {activeTab === 'agents' && (
          <AgentManager />
        )}

        {activeTab === 'charts' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <AdvancedChart 
                data={chartData} 
                symbol="AAPL" 
                interval="1H" 
              />
              <AdvancedChart 
                data={generateMockOHLCV(100)} 
                symbol="GOOGL" 
                interval="1H" 
              />
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <AdvancedChart 
                data={generateMockOHLCV(100)} 
                symbol="MSFT" 
                interval="1H" 
              />
              <AdvancedChart 
                data={generateMockOHLCV(100)} 
                symbol="BTC/USD" 
                interval="4H" 
              />
            </div>
          </div>
        )}

        {activeTab === 'settings' && (
          <SettingsPanel />
        )}
      </main>
    </div>
  );
}

