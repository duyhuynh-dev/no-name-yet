/**
 * React Hooks for WebSocket Integration
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import {
  WebSocketClient,
  getWebSocketClient,
  PortfolioUpdate,
  PositionUpdate,
  TradeExecuted,
  MarketDataUpdate,
  AgentSignal,
  AlertMessage,
} from '@/lib/websocket';

interface UseWebSocketOptions {
  url?: string;
  autoConnect?: boolean;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  connectionState: string;
  connect: () => Promise<void>;
  disconnect: () => void;
  subscribe: (channel: string) => void;
  unsubscribe: (channel: string) => void;
}

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const { url, autoConnect = true } = options;
  const [isConnected, setIsConnected] = useState(false);
  const [connectionState, setConnectionState] = useState('DISCONNECTED');
  const clientRef = useRef<WebSocketClient | null>(null);

  useEffect(() => {
    clientRef.current = getWebSocketClient(url);

    const checkConnection = () => {
      if (clientRef.current) {
        setIsConnected(clientRef.current.isConnected);
        setConnectionState(clientRef.current.connectionState);
      }
    };

    // Poll connection state
    const interval = setInterval(checkConnection, 1000);

    if (autoConnect) {
      clientRef.current.connect().catch(console.error);
    }

    return () => {
      clearInterval(interval);
    };
  }, [url, autoConnect]);

  const connect = useCallback(async () => {
    if (clientRef.current) {
      await clientRef.current.connect();
    }
  }, []);

  const disconnect = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.disconnect();
    }
  }, []);

  const subscribe = useCallback((channel: string) => {
    if (clientRef.current) {
      clientRef.current.subscribe(channel);
    }
  }, []);

  const unsubscribe = useCallback((channel: string) => {
    if (clientRef.current) {
      clientRef.current.unsubscribe(channel);
    }
  }, []);

  return {
    isConnected,
    connectionState,
    connect,
    disconnect,
    subscribe,
    unsubscribe,
  };
}

/**
 * Hook for real-time portfolio updates
 */
export function usePortfolioUpdates(): PortfolioUpdate | null {
  const [portfolio, setPortfolio] = useState<PortfolioUpdate | null>(null);

  useEffect(() => {
    const client = getWebSocketClient();
    
    const unsubscribe = client.on('portfolio_update', (data) => {
      setPortfolio(data);
    });

    client.subscribe('portfolio');

    return () => {
      unsubscribe();
      client.unsubscribe('portfolio');
    };
  }, []);

  return portfolio;
}

/**
 * Hook for real-time position updates
 */
export function usePositionUpdates(): Map<string, PositionUpdate> {
  const [positions, setPositions] = useState<Map<string, PositionUpdate>>(new Map());

  useEffect(() => {
    const client = getWebSocketClient();
    
    const unsubscribe = client.on('position_update', (data) => {
      setPositions(prev => {
        const next = new Map(prev);
        next.set(data.symbol, data);
        return next;
      });
    });

    client.subscribe('positions');

    return () => {
      unsubscribe();
      client.unsubscribe('positions');
    };
  }, []);

  return positions;
}

/**
 * Hook for real-time trade feed
 */
export function useTradeUpdates(limit: number = 50): TradeExecuted[] {
  const [trades, setTrades] = useState<TradeExecuted[]>([]);

  useEffect(() => {
    const client = getWebSocketClient();
    
    const unsubscribe = client.on('trade_executed', (data) => {
      setTrades(prev => [data, ...prev].slice(0, limit));
    });

    client.subscribe('trades');

    return () => {
      unsubscribe();
      client.unsubscribe('trades');
    };
  }, [limit]);

  return trades;
}

/**
 * Hook for real-time market data
 */
export function useMarketData(symbols: string[]): Map<string, MarketDataUpdate> {
  const [marketData, setMarketData] = useState<Map<string, MarketDataUpdate>>(new Map());

  useEffect(() => {
    const client = getWebSocketClient();
    
    const unsubscribe = client.on('market_data', (data) => {
      if (symbols.includes(data.symbol)) {
        setMarketData(prev => {
          const next = new Map(prev);
          next.set(data.symbol, data);
          return next;
        });
      }
    });

    symbols.forEach(symbol => {
      client.subscribe(`market:${symbol}`);
    });

    return () => {
      unsubscribe();
      symbols.forEach(symbol => {
        client.unsubscribe(`market:${symbol}`);
      });
    };
  }, [symbols]);

  return marketData;
}

/**
 * Hook for agent signals
 */
export function useAgentSignals(): AgentSignal[] {
  const [signals, setSignals] = useState<AgentSignal[]>([]);

  useEffect(() => {
    const client = getWebSocketClient();
    
    const unsubscribe = client.on('agent_signal', (data) => {
      setSignals(prev => [data, ...prev].slice(0, 100));
    });

    client.subscribe('agents');

    return () => {
      unsubscribe();
      client.unsubscribe('agents');
    };
  }, []);

  return signals;
}

/**
 * Hook for alerts
 */
export function useAlerts(): AlertMessage[] {
  const [alerts, setAlerts] = useState<AlertMessage[]>([]);

  useEffect(() => {
    const client = getWebSocketClient();
    
    const unsubscribe = client.on('alert', (data) => {
      setAlerts(prev => [data, ...prev].slice(0, 50));
    });

    client.subscribe('alerts');

    return () => {
      unsubscribe();
      client.unsubscribe('alerts');
    };
  }, []);

  return alerts;
}

