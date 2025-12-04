'use client';

/**
 * Settings Panel Component
 * 
 * Configuration UI for trading platform
 */

import React, { useState } from 'react';
import { 
  Settings, 
  Shield, 
  Bell, 
  Palette, 
  Globe, 
  Database,
  Key,
  User,
  Save,
  RefreshCw,
  ToggleLeft,
  ToggleRight,
} from 'lucide-react';

interface ToggleProps {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
  label: string;
  description?: string;
}

function Toggle({ enabled, onChange, label, description }: ToggleProps) {
  return (
    <div className="flex items-center justify-between py-3">
      <div>
        <p className="text-sm font-medium text-white">{label}</p>
        {description && (
          <p className="text-xs text-slate-400 mt-0.5">{description}</p>
        )}
      </div>
      <button
        onClick={() => onChange(!enabled)}
        className={`
          relative w-11 h-6 rounded-full transition-colors duration-200
          ${enabled ? 'bg-cyan-500' : 'bg-slate-600'}
        `}
      >
        <span
          className={`
            absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm
            transition-transform duration-200
            ${enabled ? 'translate-x-5' : 'translate-x-0'}
          `}
        />
      </button>
    </div>
  );
}

interface SliderProps {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  label: string;
  unit?: string;
}

function Slider({ value, min, max, step, onChange, label, unit = '' }: SliderProps) {
  return (
    <div className="py-3">
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm font-medium text-white">{label}</p>
        <span className="text-sm text-cyan-400">
          {value}{unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-slate-700 rounded-full appearance-none cursor-pointer
          [&::-webkit-slider-thumb]:appearance-none
          [&::-webkit-slider-thumb]:w-4
          [&::-webkit-slider-thumb]:h-4
          [&::-webkit-slider-thumb]:rounded-full
          [&::-webkit-slider-thumb]:bg-cyan-500
          [&::-webkit-slider-thumb]:cursor-pointer
          [&::-webkit-slider-thumb]:shadow-lg"
      />
      <div className="flex justify-between text-xs text-slate-500 mt-1">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  );
}

type TabId = 'trading' | 'risk' | 'notifications' | 'api' | 'appearance';

interface Tab {
  id: TabId;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const TABS: Tab[] = [
  { id: 'trading', label: 'Trading', icon: Settings },
  { id: 'risk', label: 'Risk', icon: Shield },
  { id: 'notifications', label: 'Notifications', icon: Bell },
  { id: 'api', label: 'API Keys', icon: Key },
  { id: 'appearance', label: 'Appearance', icon: Palette },
];

export function SettingsPanel() {
  const [activeTab, setActiveTab] = useState<TabId>('trading');
  const [hasChanges, setHasChanges] = useState(false);

  // Trading settings
  const [tradingMode, setTradingMode] = useState<'paper' | 'live'>('paper');
  const [autoTrade, setAutoTrade] = useState(true);
  const [confirmOrders, setConfirmOrders] = useState(false);

  // Risk settings
  const [maxDrawdown, setMaxDrawdown] = useState(10);
  const [maxPositionSize, setMaxPositionSize] = useState(10);
  const [maxLeverage, setMaxLeverage] = useState(2);
  const [stopLossEnabled, setStopLossEnabled] = useState(true);
  const [dailyLossLimit, setDailyLossLimit] = useState(5);

  // Notification settings
  const [tradeAlerts, setTradeAlerts] = useState(true);
  const [riskAlerts, setRiskAlerts] = useState(true);
  const [performanceAlerts, setPerformanceAlerts] = useState(true);
  const [emailNotifications, setEmailNotifications] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(true);

  // Appearance
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [compactMode, setCompactMode] = useState(false);
  const [showAnimations, setShowAnimations] = useState(true);

  const handleSave = () => {
    // Save settings
    setHasChanges(false);
    // Show success toast
  };

  const handleChange = <T,>(setter: (value: T) => void) => (value: T) => {
    setter(value);
    setHasChanges(true);
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'trading':
        return (
          <div className="space-y-6">
            {/* Trading Mode */}
            <div>
              <h4 className="text-sm font-medium text-white mb-3">Trading Mode</h4>
              <div className="grid grid-cols-2 gap-3">
                {(['paper', 'live'] as const).map(mode => (
                  <button
                    key={mode}
                    onClick={() => handleChange(setTradingMode)(mode)}
                    className={`
                      p-4 rounded-lg border transition-all
                      ${tradingMode === mode 
                        ? 'border-cyan-500 bg-cyan-500/10' 
                        : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'}
                    `}
                  >
                    <div className="flex items-center gap-3">
                      {mode === 'paper' ? (
                        <Database className={`w-5 h-5 ${tradingMode === mode ? 'text-cyan-400' : 'text-slate-400'}`} />
                      ) : (
                        <Globe className={`w-5 h-5 ${tradingMode === mode ? 'text-cyan-400' : 'text-slate-400'}`} />
                      )}
                      <div className="text-left">
                        <p className={`font-medium capitalize ${tradingMode === mode ? 'text-cyan-400' : 'text-white'}`}>
                          {mode}
                        </p>
                        <p className="text-xs text-slate-400">
                          {mode === 'paper' ? 'Simulated trades' : 'Real money'}
                        </p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
              {tradingMode === 'live' && (
                <p className="text-xs text-amber-400 mt-2 flex items-center gap-1">
                  <Shield className="w-3 h-3" />
                  Warning: Live trading involves real money
                </p>
              )}
            </div>

            <div className="border-t border-slate-700/50 pt-4">
              <Toggle
                enabled={autoTrade}
                onChange={handleChange(setAutoTrade)}
                label="Auto Trading"
                description="Execute trades automatically based on agent signals"
              />
              <Toggle
                enabled={confirmOrders}
                onChange={handleChange(setConfirmOrders)}
                label="Confirm Orders"
                description="Require manual confirmation before executing"
              />
            </div>
          </div>
        );

      case 'risk':
        return (
          <div className="space-y-4">
            <Slider
              value={maxDrawdown}
              min={1}
              max={25}
              step={1}
              onChange={handleChange(setMaxDrawdown)}
              label="Max Drawdown"
              unit="%"
            />
            <Slider
              value={maxPositionSize}
              min={1}
              max={25}
              step={1}
              onChange={handleChange(setMaxPositionSize)}
              label="Max Position Size"
              unit="%"
            />
            <Slider
              value={maxLeverage}
              min={1}
              max={5}
              step={0.5}
              onChange={handleChange(setMaxLeverage)}
              label="Max Leverage"
              unit="x"
            />
            <Slider
              value={dailyLossLimit}
              min={1}
              max={10}
              step={0.5}
              onChange={handleChange(setDailyLossLimit)}
              label="Daily Loss Limit"
              unit="%"
            />
            <div className="border-t border-slate-700/50 pt-4">
              <Toggle
                enabled={stopLossEnabled}
                onChange={handleChange(setStopLossEnabled)}
                label="Auto Stop Loss"
                description="Automatically set stop loss for all positions"
              />
            </div>
          </div>
        );

      case 'notifications':
        return (
          <div className="space-y-4">
            <Toggle
              enabled={tradeAlerts}
              onChange={handleChange(setTradeAlerts)}
              label="Trade Alerts"
              description="Notify when trades are executed"
            />
            <Toggle
              enabled={riskAlerts}
              onChange={handleChange(setRiskAlerts)}
              label="Risk Alerts"
              description="Warn when risk limits are approached"
            />
            <Toggle
              enabled={performanceAlerts}
              onChange={handleChange(setPerformanceAlerts)}
              label="Performance Alerts"
              description="Daily performance summary"
            />
            <div className="border-t border-slate-700/50 pt-4">
              <Toggle
                enabled={emailNotifications}
                onChange={handleChange(setEmailNotifications)}
                label="Email Notifications"
                description="Send alerts to email"
              />
              <Toggle
                enabled={soundEnabled}
                onChange={handleChange(setSoundEnabled)}
                label="Sound Effects"
                description="Play sounds for notifications"
              />
            </div>
          </div>
        );

      case 'api':
        return (
          <div className="space-y-4">
            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-green-500 to-emerald-500 flex items-center justify-center">
                  <span className="text-white font-bold text-xs">AL</span>
                </div>
                <div>
                  <p className="font-medium text-white">Alpaca</p>
                  <p className="text-xs text-emerald-400">Connected</p>
                </div>
              </div>
              <div className="space-y-2">
                <div>
                  <label className="text-xs text-slate-400">API Key</label>
                  <input
                    type="password"
                    value="••••••••••••••••"
                    readOnly
                    className="w-full mt-1 px-3 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-400">Secret Key</label>
                  <input
                    type="password"
                    value="••••••••••••••••"
                    readOnly
                    className="w-full mt-1 px-3 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white text-sm"
                  />
                </div>
              </div>
            </div>

            <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center">
                  <span className="text-white font-bold text-xs">IB</span>
                </div>
                <div>
                  <p className="font-medium text-white">Interactive Brokers</p>
                  <p className="text-xs text-slate-400">Not Connected</p>
                </div>
              </div>
              <button className="w-full py-2 bg-slate-700/50 text-slate-400 rounded-lg hover:bg-slate-700 transition-colors text-sm">
                Configure Connection
              </button>
            </div>
          </div>
        );

      case 'appearance':
        return (
          <div className="space-y-4">
            <div>
              <h4 className="text-sm font-medium text-white mb-3">Theme</h4>
              <div className="grid grid-cols-2 gap-3">
                {(['dark', 'light'] as const).map(t => (
                  <button
                    key={t}
                    onClick={() => handleChange(setTheme)(t)}
                    className={`
                      p-4 rounded-lg border transition-all
                      ${theme === t 
                        ? 'border-cyan-500 bg-cyan-500/10' 
                        : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'}
                    `}
                  >
                    <p className={`font-medium capitalize ${theme === t ? 'text-cyan-400' : 'text-white'}`}>
                      {t}
                    </p>
                  </button>
                ))}
              </div>
            </div>
            <div className="border-t border-slate-700/50 pt-4">
              <Toggle
                enabled={compactMode}
                onChange={handleChange(setCompactMode)}
                label="Compact Mode"
                description="Reduce spacing for more data"
              />
              <Toggle
                enabled={showAnimations}
                onChange={handleChange(setShowAnimations)}
                label="Animations"
                description="Enable UI animations"
              />
            </div>
          </div>
        );
    }
  };

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700/50 flex items-center justify-between">
        <h3 className="text-white font-semibold flex items-center gap-2">
          <Settings className="w-5 h-5 text-slate-400" />
          Settings
        </h3>
        {hasChanges && (
          <button
            onClick={handleSave}
            className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500 text-white rounded-lg text-sm hover:bg-cyan-600 transition-colors"
          >
            <Save className="w-4 h-4" />
            Save
          </button>
        )}
      </div>

      <div className="flex">
        {/* Sidebar */}
        <div className="w-48 border-r border-slate-700/50 p-2">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-all
                ${activeTab === tab.id 
                  ? 'bg-cyan-500/10 text-cyan-400' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-700/50'}
              `}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 p-4">
          {renderContent()}
        </div>
      </div>
    </div>
  );
}

