"""
Pydantic schemas for API request/response models.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class ActionType(str, Enum):
    """Trading action types."""
    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"


class MarketState(BaseModel):
    """Market state input for prediction."""
    
    # OHLCV data (list of recent candles)
    open: List[float] = Field(..., description="Open prices", min_length=1)
    high: List[float] = Field(..., description="High prices", min_length=1)
    low: List[float] = Field(..., description="Low prices", min_length=1)
    close: List[float] = Field(..., description="Close prices", min_length=1)
    volume: List[float] = Field(..., description="Volume", min_length=1)
    
    # Optional technical indicators (if not provided, will be calculated)
    rsi: Optional[List[float]] = Field(None, description="RSI values")
    macd: Optional[List[float]] = Field(None, description="MACD values")
    macd_signal: Optional[List[float]] = Field(None, description="MACD signal")
    bb_upper: Optional[List[float]] = Field(None, description="Bollinger upper band")
    bb_lower: Optional[List[float]] = Field(None, description="Bollinger lower band")
    
    # Portfolio state
    position: int = Field(0, description="Current position: -1 (short), 0 (flat), 1 (long)")
    cash: float = Field(10000.0, description="Available cash")
    shares: float = Field(0.0, description="Number of shares held")
    
    class Config:
        json_schema_extra = {
            "example": {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000000, 1100000, 1200000],
                "position": 0,
                "cash": 10000.0,
                "shares": 0.0
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response from the model."""
    
    action: ActionType = Field(..., description="Recommended action")
    action_id: int = Field(..., description="Action ID (0=hold, 1=buy, 2=sell)")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    probabilities: Dict[str, float] = Field(..., description="Action probabilities")
    
    # Optional reasoning
    reasoning: Optional[str] = Field(None, description="Explanation for the action")
    
    # Latency info
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "action": "buy",
                "action_id": 1,
                "confidence": 0.75,
                "probabilities": {"hold": 0.15, "buy": 0.75, "sell": 0.10},
                "reasoning": "Strong upward momentum detected",
                "latency_ms": 12.5
            }
        }


class BacktestRequest(BaseModel):
    """Request for backtesting."""
    
    symbol: str = Field(..., description="Trading symbol")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    initial_balance: float = Field(10000.0, description="Initial balance")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "SPY",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "initial_balance": 10000.0
            }
        }


class BacktestResponse(BaseModel):
    """Backtest results."""
    
    # Performance metrics
    total_return: float = Field(..., description="Total return percentage")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Win rate percentage")
    num_trades: int = Field(..., description="Number of trades")
    
    # Portfolio tracking
    initial_balance: float
    final_balance: float
    
    # Detailed data
    equity_curve: List[float] = Field(..., description="Portfolio value over time")
    trades: List[Dict[str, Any]] = Field(..., description="List of trades")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_return": 15.5,
                "sharpe_ratio": 1.2,
                "max_drawdown": 5.3,
                "win_rate": 55.0,
                "num_trades": 42,
                "initial_balance": 10000.0,
                "final_balance": 11550.0,
                "equity_curve": [10000, 10100, 10250],
                "trades": []
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Loaded model name")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "ppo_trading_v1",
                "version": "1.0.0"
            }
        }


class ModelInfo(BaseModel):
    """Model information."""
    
    name: str
    type: str
    features_dim: int
    window_size: int
    device: str
    loaded_at: str


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str
    detail: Optional[str] = None
    status_code: int

