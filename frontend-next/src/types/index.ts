// API Types

export type ActionType = 'hold' | 'buy' | 'sell';

export interface MarketState {
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
  position: number;
  cash: number;
  shares: number;
}

export interface PredictionResponse {
  action: ActionType;
  action_id: number;
  confidence: number;
  probabilities: {
    hold: number;
    buy: number;
    sell: number;
  };
  reasoning: string | null;
  latency_ms: number;
}

export interface BacktestRequest {
  symbol: string;
  start_date?: string;
  end_date?: string;
  initial_balance: number;
}

export interface Trade {
  step: number;
  type: string;
  price: number;
  pnl?: number;
  shares?: number;
}

export interface BacktestResponse {
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  num_trades: number;
  initial_balance: number;
  final_balance: number;
  equity_curve: number[];
  trades: Trade[];
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  model_name: string | null;
  version: string;
}

export interface ModelInfo {
  loaded: boolean;
  name?: string;
  device?: string;
  window_size?: number;
  n_features?: number;
  loaded_at?: string;
}

export interface SymbolsResponse {
  symbols: string[];
}

