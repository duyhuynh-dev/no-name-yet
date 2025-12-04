/**
 * HFT RL Trading Dashboard
 * Frontend application for the trading agent
 */

const API_BASE = 'http://localhost:8000';

// State
let priceChart = null;
let equityChart = null;
let backtestData = null;

// DOM Elements
const elements = {
    apiStatus: document.getElementById('apiStatus'),
    latencyBadge: document.getElementById('latencyBadge'),
    actionDisplay: document.getElementById('actionDisplay'),
    confidenceFill: document.getElementById('confidenceFill'),
    confidenceValue: document.getElementById('confidenceValue'),
    actionProbs: document.getElementById('actionProbs'),
    symbolInput: document.getElementById('symbolInput'),
    balanceInput: document.getElementById('balanceInput'),
    runBacktestBtn: document.getElementById('runBacktestBtn'),
    totalReturn: document.getElementById('totalReturn'),
    sharpeRatio: document.getElementById('sharpeRatio'),
    maxDrawdown: document.getElementById('maxDrawdown'),
    winRate: document.getElementById('winRate'),
    totalTrades: document.getElementById('totalTrades'),
    finalBalance: document.getElementById('finalBalance'),
    equityChange: document.getElementById('equityChange'),
    tradesList: document.getElementById('tradesList'),
    modelInfo: document.getElementById('modelInfo'),
};

// Chart colors
const chartColors = {
    primary: '#00d4aa',
    secondary: '#00b894',
    buy: '#00d4aa',
    sell: '#ff6b6b',
    hold: '#feca57',
    grid: 'rgba(255, 255, 255, 0.05)',
    text: '#8892a4',
};

// Initialize charts
function initCharts() {
    // Common chart options
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: true,
        aspectRatio: 2.5,  // Width:Height ratio
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#1a2332',
                titleColor: '#e8edf5',
                bodyColor: '#8892a4',
                borderColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 1,
                displayColors: false,
            }
        },
        scales: {
            x: {
                grid: { color: chartColors.grid },
                ticks: { color: chartColors.text, maxTicksLimit: 10 }
            },
            y: {
                grid: { color: chartColors.grid },
                ticks: { color: chartColors.text }
            }
        },
        interaction: {
            intersect: false,
            mode: 'index',
        }
    };

    // Price Chart
    const priceCtx = document.getElementById('priceChart').getContext('2d');
    priceChart = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Price',
                data: [],
                borderColor: chartColors.primary,
                backgroundColor: 'rgba(0, 212, 170, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                borderWidth: 2,
            }]
        },
        options: {
            ...commonOptions,
        }
    });

    // Equity Chart
    const equityCtx = document.getElementById('equityChart').getContext('2d');
    equityChart = new Chart(equityCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: chartColors.primary,
                backgroundColor: createGradient(equityCtx),
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                borderWidth: 2,
            }]
        },
        options: {
            ...commonOptions,
            plugins: {
                ...commonOptions.plugins,
                tooltip: {
                    ...commonOptions.plugins.tooltip,
                    callbacks: {
                        label: (ctx) => `$${ctx.raw.toLocaleString('en-US', { minimumFractionDigits: 2 })}`
                    }
                }
            },
            scales: {
                ...commonOptions.scales,
                y: {
                    grid: { color: chartColors.grid },
                    ticks: { 
                        color: chartColors.text,
                        callback: (v) => '$' + v.toLocaleString()
                    }
                }
            },
        }
    });
}

function createGradient(ctx) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 200);
    gradient.addColorStop(0, 'rgba(0, 212, 170, 0.25)');
    gradient.addColorStop(0.5, 'rgba(0, 212, 170, 0.1)');
    gradient.addColorStop(1, 'rgba(0, 212, 170, 0)');
    return gradient;
}

// API Functions
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        
        elements.apiStatus.classList.remove('error');
        elements.apiStatus.classList.add('connected');
        elements.apiStatus.querySelector('.status-text').textContent = 'Connected';
        
        if (data.model_loaded) {
            elements.modelInfo.textContent = `Model: ${data.model_name}`;
        }
        
        return true;
    } catch (error) {
        elements.apiStatus.classList.remove('connected');
        elements.apiStatus.classList.add('error');
        elements.apiStatus.querySelector('.status-text').textContent = 'Disconnected';
        return false;
    }
}

async function fetchSymbols() {
    try {
        const response = await fetch(`${API_BASE}/symbols`);
        const data = await response.json();
        
        elements.symbolInput.innerHTML = '';
        data.symbols.forEach(symbol => {
            const option = document.createElement('option');
            option.value = symbol;
            option.textContent = symbol;
            elements.symbolInput.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to fetch symbols:', error);
    }
}

async function makePrediction(ohlcvData) {
    try {
        const startTime = performance.now();
        
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                open: ohlcvData.open,
                high: ohlcvData.high,
                low: ohlcvData.low,
                close: ohlcvData.close,
                volume: ohlcvData.volume,
                position: 0,
                cash: 10000,
                shares: 0,
            })
        });
        
        const data = await response.json();
        const latency = data.latency_ms || (performance.now() - startTime);
        
        updatePredictionUI(data, latency);
        return data;
    } catch (error) {
        console.error('Prediction failed:', error);
        return null;
    }
}

async function runBacktest() {
    const symbol = elements.symbolInput.value;
    const initialBalance = parseFloat(elements.balanceInput.value);
    
    elements.runBacktestBtn.disabled = true;
    elements.runBacktestBtn.innerHTML = '<span class="btn-icon">⏳</span> Running...';
    
    try {
        const response = await fetch(`${API_BASE}/backtest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol,
                initial_balance: initialBalance,
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.detail || data.error);
        }
        
        backtestData = data;
        updateBacktestUI(data);
        
    } catch (error) {
        console.error('Backtest failed:', error);
        alert(`Backtest failed: ${error.message}`);
    } finally {
        elements.runBacktestBtn.disabled = false;
        elements.runBacktestBtn.innerHTML = '<span class="btn-icon">▶</span> Run Backtest';
    }
}

// UI Update Functions
function updatePredictionUI(data, latency) {
    // Update latency
    elements.latencyBadge.querySelector('.latency-value').textContent = latency.toFixed(1);
    
    // Update action display
    const actionClasses = { hold: 'hold', buy: 'buy', sell: 'sell' };
    const actionIcons = { hold: '◯', buy: '▲', sell: '▼' };
    
    elements.actionDisplay.className = `action-display ${actionClasses[data.action]}`;
    elements.actionDisplay.querySelector('.action-icon').textContent = actionIcons[data.action];
    elements.actionDisplay.querySelector('.action-text').textContent = data.action.toUpperCase();
    
    // Update confidence
    const confidence = (data.confidence * 100).toFixed(1);
    elements.confidenceFill.style.width = `${confidence}%`;
    elements.confidenceValue.textContent = `${confidence}%`;
    
    // Update probabilities
    const probItems = elements.actionProbs.querySelectorAll('.prob-item');
    const actions = ['hold', 'buy', 'sell'];
    
    probItems.forEach((item, i) => {
        const prob = (data.probabilities[actions[i]] * 100).toFixed(1);
        item.querySelector('.prob-fill').style.width = `${prob}%`;
        item.querySelector('.prob-value').textContent = `${prob}%`;
    });
}

function updateBacktestUI(data) {
    // Update metrics
    const totalReturn = data.total_return;
    elements.totalReturn.textContent = `${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%`;
    elements.totalReturn.className = `metric-value ${totalReturn >= 0 ? 'positive' : 'negative'}`;
    
    elements.sharpeRatio.textContent = data.sharpe_ratio.toFixed(3);
    
    elements.maxDrawdown.textContent = `${data.max_drawdown.toFixed(2)}%`;
    
    elements.winRate.textContent = `${data.win_rate.toFixed(1)}%`;
    
    elements.totalTrades.textContent = data.num_trades;
    
    elements.finalBalance.textContent = `$${data.final_balance.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
    
    // Update equity change badge
    elements.equityChange.textContent = `${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%`;
    elements.equityChange.className = `equity-change ${totalReturn >= 0 ? 'positive' : 'negative'}`;
    
    // Update equity chart
    updateEquityChart(data.equity_curve);
    
    // Update trades list
    updateTradesList(data.trades);
    
    // Simulate prediction with last data point
    simulatePrediction();
}

function updateEquityChart(equityCurve) {
    const labels = equityCurve.map((_, i) => `Day ${i}`);
    
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = equityCurve;
    equityChart.update('none');
    
    // Also update price chart with simulated data
    updatePriceChartFromEquity(equityCurve);
}

function updatePriceChartFromEquity(equityCurve) {
    // Generate simulated price data from equity changes
    const basePrice = 100;
    let price = basePrice;
    const prices = [price];
    
    for (let i = 1; i < equityCurve.length; i++) {
        const change = (equityCurve[i] - equityCurve[i-1]) / equityCurve[i-1];
        price *= (1 + change * 0.5); // Scale down changes
        prices.push(price);
    }
    
    const labels = prices.map((_, i) => `${i}`);
    
    priceChart.data.labels = labels;
    priceChart.data.datasets[0].data = prices;
    priceChart.update('none');
}

function updateTradesList(trades) {
    if (!trades || trades.length === 0) {
        elements.tradesList.innerHTML = '<div class="no-trades">No trades executed</div>';
        return;
    }
    
    const recentTrades = trades.slice(-10).reverse();
    
    elements.tradesList.innerHTML = recentTrades.map(trade => {
        const type = trade.type || 'unknown';
        const isBuy = type.includes('buy');
        const pnl = trade.pnl || 0;
        const price = trade.price || 0;
        
        return `
            <div class="trade-item">
                <span class="trade-type ${isBuy ? 'buy' : 'sell'}">${type.replace('_', ' ')}</span>
                <span class="trade-price">$${price.toFixed(2)}</span>
                ${pnl !== 0 ? `<span class="trade-pnl ${pnl >= 0 ? 'positive' : 'negative'}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)}</span>` : ''}
            </div>
        `;
    }).join('');
}

async function simulatePrediction() {
    // Generate sample OHLCV data for prediction demo
    const n = 35;
    const basePrice = 100;
    const data = {
        open: [],
        high: [],
        low: [],
        close: [],
        volume: [],
    };
    
    let price = basePrice;
    for (let i = 0; i < n; i++) {
        const change = (Math.random() - 0.5) * 2;
        const open = price;
        const close = price + change;
        const high = Math.max(open, close) + Math.random();
        const low = Math.min(open, close) - Math.random();
        
        data.open.push(open);
        data.high.push(high);
        data.low.push(low);
        data.close.push(close);
        data.volume.push(1000000 + Math.random() * 500000);
        
        price = close;
    }
    
    await makePrediction(data);
}

// Event Listeners
elements.runBacktestBtn.addEventListener('click', runBacktest);

// Initialize
async function init() {
    initCharts();
    
    const connected = await checkHealth();
    if (connected) {
        await fetchSymbols();
        await simulatePrediction();
    }
    
    // Periodic health check
    setInterval(checkHealth, 10000);
}

// Start app
document.addEventListener('DOMContentLoaded', init);

