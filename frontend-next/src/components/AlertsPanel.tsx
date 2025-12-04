'use client';

/**
 * Alerts Panel Component
 * 
 * Displays real-time system and trading alerts
 */

import React from 'react';
import { Bell, AlertTriangle, AlertCircle, Info, X, CheckCircle } from 'lucide-react';
import { useAlerts } from '@/hooks/useWebSocket';
import { AlertMessage } from '@/lib/websocket';

interface AlertItemProps {
  alert: AlertMessage;
  onDismiss?: (id: string) => void;
}

function AlertItem({ alert, onDismiss }: AlertItemProps) {
  const config = {
    info: {
      icon: Info,
      bg: 'bg-blue-500/10',
      border: 'border-blue-500/30',
      text: 'text-blue-400',
    },
    warning: {
      icon: AlertTriangle,
      bg: 'bg-amber-500/10',
      border: 'border-amber-500/30',
      text: 'text-amber-400',
    },
    critical: {
      icon: AlertCircle,
      bg: 'bg-red-500/10',
      border: 'border-red-500/30',
      text: 'text-red-400',
    },
  }[alert.severity];

  const Icon = config.icon;

  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <div className={`
      p-3 rounded-lg border ${config.bg} ${config.border}
      transition-all duration-300 hover:scale-[1.01]
    `}>
      <div className="flex items-start gap-3">
        <div className={`p-1.5 rounded-lg ${config.bg}`}>
          <Icon className={`w-4 h-4 ${config.text}`} />
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <h4 className="text-sm font-medium text-white truncate">
              {alert.title}
            </h4>
            <span className="text-xs text-slate-500 whitespace-nowrap">
              {formatTime(alert.timestamp)}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1 line-clamp-2">
            {alert.message}
          </p>
        </div>

        {onDismiss && (
          <button
            onClick={() => onDismiss(alert.alert_id)}
            className="p-1 rounded hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-3 h-3" />
          </button>
        )}
      </div>
    </div>
  );
}

export function AlertsPanel() {
  const alerts = useAlerts();
  const [dismissed, setDismissed] = React.useState<Set<string>>(new Set());

  const visibleAlerts = alerts.filter(a => !dismissed.has(a.alert_id));
  const criticalCount = visibleAlerts.filter(a => a.severity === 'critical').length;
  const warningCount = visibleAlerts.filter(a => a.severity === 'warning').length;

  const handleDismiss = (id: string) => {
    setDismissed(prev => new Set([...prev, id]));
  };

  const handleDismissAll = () => {
    setDismissed(new Set(alerts.map(a => a.alert_id)));
  };

  if (visibleAlerts.length === 0) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6">
        <h3 className="text-white font-semibold flex items-center gap-2 mb-4">
          <Bell className="w-5 h-5 text-slate-400" />
          Alerts
        </h3>
        <div className="flex flex-col items-center justify-center py-8 text-slate-400">
          <CheckCircle className="w-10 h-10 mb-3 text-emerald-500/50" />
          <p className="text-sm font-medium text-emerald-400">All Clear</p>
          <p className="text-xs text-slate-500">No active alerts</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-slate-700/50">
        <div className="flex items-center justify-between">
          <h3 className="text-white font-semibold flex items-center gap-2">
            <Bell className="w-5 h-5 text-amber-400" />
            Alerts
            {visibleAlerts.length > 0 && (
              <span className="px-2 py-0.5 text-xs rounded-full bg-amber-500/20 text-amber-400">
                {visibleAlerts.length}
              </span>
            )}
          </h3>
          <div className="flex items-center gap-3">
            {criticalCount > 0 && (
              <span className="text-xs text-red-400">
                {criticalCount} critical
              </span>
            )}
            {warningCount > 0 && (
              <span className="text-xs text-amber-400">
                {warningCount} warning
              </span>
            )}
            <button
              onClick={handleDismissAll}
              className="text-xs text-slate-400 hover:text-white transition-colors"
            >
              Dismiss all
            </button>
          </div>
        </div>
      </div>

      {/* Alerts List */}
      <div className="p-3 space-y-2 max-h-[400px] overflow-y-auto">
        {/* Critical alerts first */}
        {visibleAlerts
          .sort((a, b) => {
            const priority = { critical: 0, warning: 1, info: 2 };
            return priority[a.severity] - priority[b.severity];
          })
          .map(alert => (
            <AlertItem 
              key={alert.alert_id} 
              alert={alert} 
              onDismiss={handleDismiss}
            />
          ))
        }
      </div>
    </div>
  );
}

