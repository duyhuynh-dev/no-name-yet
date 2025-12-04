'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  Header,
  PredictionPanel,
  MetricsPanel,
  BacktestPanel,
  TradesPanel,
  Chart,
} from '@/components';
import { predict, runBacktest, generateSampleOHLCV } from '@/lib/api';
import type { PredictionResponse, BacktestResponse } from '@/types';

export default function Dashboard() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [backtestResults, setBacktestResults] = useState<BacktestResponse | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isBacktesting, setIsBacktesting] = useState(false);

  // Make a prediction with sample data
  const makePrediction = useCallback(async () => {
    setIsPredicting(true);
    try {
      const sampleData = generateSampleOHLCV(35);
      const result = await predict(sampleData);
      setPrediction(result);
      setLatency(result.latency_ms);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setIsPredicting(false);
    }
  }, []);

  // Run backtest
  const handleBacktest = async (symbol: string, initialBalance: number) => {
    setIsBacktesting(true);
    try {
      const result = await runBacktest({ symbol, initial_balance: initialBalance });
      setBacktestResults(result);
      // Also trigger a prediction
      await makePrediction();
    } catch (error) {
      console.error('Backtest failed:', error);
      alert('Backtest failed. Make sure the API server is running.');
    } finally {
      setIsBacktesting(false);
    }
  };

  // Initial prediction on mount
  useEffect(() => {
    makePrediction();
  }, [makePrediction]);

  // Generate price data from equity curve for display
  const generatePriceData = (equityCurve: number[]) => {
    if (!equityCurve.length) return [];
    const basePrice = 100;
    let price = basePrice;
    const prices = [price];

    for (let i = 1; i < equityCurve.length; i++) {
      const change = (equityCurve[i] - equityCurve[i - 1]) / equityCurve[i - 1];
      price *= 1 + change * 0.5;
      prices.push(price);
    }
    return prices;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Subtle background effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-emerald-500/5 rounded-full blur-3xl" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl" />
      </div>

      <div className="relative z-10 flex flex-col min-h-screen">
        <Header latency={latency} />

        <main className="flex-1 p-6 overflow-hidden">
          <div className="max-w-7xl mx-auto flex flex-col gap-4 h-full">
            {/* Top Row - Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Chart
                title="Price Action & Signals"
                data={
                  backtestResults
                    ? generatePriceData(backtestResults.equity_curve)
                    : [100, 101, 102, 101, 103, 104, 103, 105]
                }
              />
              <Chart
                title="Portfolio Equity"
                data={backtestResults?.equity_curve || [10000]}
                yAxisPrefix="$"
                changePercent={backtestResults?.total_return}
              />
            </div>

            {/* Metrics Row - Full Width Horizontal */}
            <MetricsPanel results={backtestResults} horizontal />

            {/* Bottom Row - Controls */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <PredictionPanel prediction={prediction} isLoading={isPredicting} />
              <BacktestPanel onRunBacktest={handleBacktest} isLoading={isBacktesting} />
              <TradesPanel trades={backtestResults?.trades || []} />
            </div>
          </div>
        </main>

        <footer className="text-center py-4 text-xs text-slate-600 border-t border-slate-800">
          HFT RL Trading Simulator • PPO Agent • Next.js + TypeScript
        </footer>
      </div>
    </div>
  );
}
