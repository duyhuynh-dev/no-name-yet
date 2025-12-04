"""
Pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_rows = 100
    
    # Generate realistic price data
    base_price = 100.0
    returns = np.random.randn(n_rows) * 0.02  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_rows) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(n_rows)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(n_rows)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, n_rows).astype(float),
    })
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    # Add datetime index
    data.index = pd.date_range(start='2023-01-01', periods=n_rows, freq='D')
    
    return data


@pytest.fixture
def sample_features_data(sample_ohlcv_data):
    """Generate sample data with technical indicators."""
    df = sample_ohlcv_data.copy()
    
    # Add basic features
    df['returns'] = df['close'].pct_change().fillna(0)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    df['volatility'] = df['returns'].rolling(20).std().fillna(0)
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = (ema12 - ema26).fillna(0)
    
    # Moving averages
    df['sma_5'] = df['close'].rolling(5).mean().fillna(df['close'])
    df['sma_20'] = df['close'].rolling(20).mean().fillna(df['close'])
    
    # Bollinger Bands
    sma20 = df['close'].rolling(20).mean()
    std20 = df['close'].rolling(20).std()
    df['bb_upper'] = (sma20 + 2 * std20).fillna(df['close'] * 1.02)
    df['bb_lower'] = (sma20 - 2 * std20).fillna(df['close'] * 0.98)
    
    # More features to match expected count
    df['momentum_5'] = (df['close'] / df['close'].shift(5) - 1).fillna(0)
    df['momentum_10'] = (df['close'] / df['close'].shift(10) - 1).fillna(0)
    df['volume_ma'] = df['volume'].rolling(20).mean().fillna(df['volume'])
    df['volume_ratio'] = (df['volume'] / df['volume_ma']).fillna(1)
    df['hl_range'] = ((df['high'] - df['low']) / df['close']).fillna(0)
    df['oc_range'] = ((df['close'] - df['open']) / df['open']).fillna(0)
    
    # Pad to 27 features
    for i in range(27 - len(df.columns)):
        df[f'feature_{i}'] = np.random.randn(len(df)) * 0.01
    
    # Ensure exactly 27 columns
    df = df.iloc[:, :27]
    
    return df


@pytest.fixture
def trading_env(sample_features_data):
    """Create a trading environment for testing."""
    from src.env import TradingEnv
    
    return TradingEnv(
        df=sample_features_data,
        window_size=30,
        initial_balance=10000.0,
        transaction_cost=0.001,
        normalize_obs=True,
        random_start=False,
    )


@pytest.fixture
def api_client():
    """Create a test client for the API."""
    from fastapi.testclient import TestClient
    from api.main import app
    
    return TestClient(app)

