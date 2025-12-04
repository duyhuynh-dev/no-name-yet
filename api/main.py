"""
FastAPI Application for HFT RL Trading Agent.

Features:
- Model inference endpoint (<50ms latency)
- Health check
- Backtesting endpoint
- Model info endpoint
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.schemas import (
    MarketState,
    PredictionResponse,
    BacktestRequest,
    BacktestResponse,
    HealthResponse,
    ErrorResponse,
    ActionType,
)
from api.model_service import ModelService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model service
model_service: Optional[ModelService] = None

# API version
API_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    global model_service
    
    # Startup
    logger.info("Starting HFT RL Trading API...")
    
    # Initialize model service
    model_path = os.environ.get("MODEL_PATH", "models/mlp_100k_baseline/final_model")
    window_size = int(os.environ.get("WINDOW_SIZE", "30"))
    n_features = int(os.environ.get("N_FEATURES", "27"))
    
    model_service = ModelService(
        window_size=window_size,
        n_features=n_features,
    )
    
    # Try to load model
    if Path(model_path).exists() or Path(f"{model_path}.zip").exists():
        model_service.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}. Use /load endpoint to load a model.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HFT RL Trading API...")


# Create FastAPI app
app = FastAPI(
    title="HFT RL Trading API",
    description="High-Frequency Trading Market Simulator using Reinforcement Learning",
    version=API_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": "HFT RL Trading API",
        "version": API_VERSION,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.is_loaded() if model_service else False,
        model_name=model_service.model_name if model_service else None,
        version=API_VERSION,
    )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information."""
    if not model_service:
        raise HTTPException(status_code=500, detail="Model service not initialized")
    
    return model_service.get_model_info()


@app.post("/model/load", tags=["Model"])
async def load_model(model_path: str = Query(..., description="Path to model file")):
    """Load a model from disk."""
    if not model_service:
        raise HTTPException(status_code=500, detail="Model service not initialized")
    
    success = model_service.load_model(model_path)
    
    if success:
        return {"status": "success", "message": f"Model loaded from {model_path}"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to load model from {model_path}")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(state: MarketState):
    """
    Make a trading prediction.
    
    Accepts market state (OHLCV + indicators) and returns recommended action.
    Target latency: <50ms
    """
    if not model_service or not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess state
        ohlcv_data = {
            "open": state.open,
            "high": state.high,
            "low": state.low,
            "close": state.close,
            "volume": state.volume,
        }
        
        indicators = {}
        if state.rsi:
            indicators["rsi"] = state.rsi
        if state.macd:
            indicators["macd"] = state.macd
        
        preprocessed = model_service.preprocess_state(
            ohlcv_data=ohlcv_data,
            position=state.position,
            cash=state.cash,
            shares=state.shares,
            indicators=indicators if indicators else None,
        )
        
        # Make prediction
        action_id, probs, latency_ms = model_service.predict(preprocessed)
        action_name = model_service.get_action_name(action_id)
        
        # Calculate confidence
        confidence = float(probs[action_id])
        
        # Generate reasoning
        reasoning = None
        if action_name == "buy":
            reasoning = f"Buy signal with {confidence:.1%} confidence"
        elif action_name == "sell":
            reasoning = f"Sell signal with {confidence:.1%} confidence"
        else:
            reasoning = f"Hold position with {confidence:.1%} confidence"
        
        return PredictionResponse(
            action=ActionType(action_name),
            action_id=action_id,
            confidence=confidence,
            probabilities={
                "hold": float(probs[0]),
                "buy": float(probs[1]),
                "sell": float(probs[2]),
            },
            reasoning=reasoning,
            latency_ms=latency_ms,
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest", response_model=BacktestResponse, tags=["Backtest"])
async def run_backtest(request: BacktestRequest):
    """
    Run a backtest on historical data.
    """
    if not model_service or not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import pandas as pd
        from src.env import TradingEnv
        from src.mlops import Backtester
        
        # Load test data
        data_path = Path(f"data/processed/{request.symbol}_test.parquet")
        if not data_path.exists():
            raise HTTPException(status_code=404, detail=f"Data not found for {request.symbol}")
        
        df = pd.read_parquet(data_path)
        
        # Filter by date if provided
        if request.start_date:
            df = df[df.index >= request.start_date]
        if request.end_date:
            df = df[df.index <= request.end_date]
        
        if len(df) < model_service.window_size + 1:
            raise HTTPException(status_code=400, detail="Insufficient data for backtest")
        
        # Create environment
        env = TradingEnv(
            df=df,
            window_size=model_service.window_size,
            initial_balance=request.initial_balance,
            normalize_obs=True,
            random_start=False,
        )
        
        # Run backtest
        backtester = Backtester(
            transaction_cost=0.001,
            slippage=0.0005,
        )
        result = backtester.run(model_service.model, env)
        
        # Convert numpy types to Python types
        equity_curve = [float(x) for x in result.equity_curve]
        trades = []
        for trade in result.trades:
            clean_trade = {}
            for k, v in trade.items():
                if hasattr(v, 'item'):
                    clean_trade[k] = v.item()
                else:
                    clean_trade[k] = v
            trades.append(clean_trade)
        
        return BacktestResponse(
            total_return=float(result.total_return * 100),
            sharpe_ratio=float(result.sharpe_ratio),
            max_drawdown=float(result.max_drawdown * 100),
            win_rate=float(result.win_rate * 100),
            num_trades=int(result.num_trades),
            initial_balance=float(request.initial_balance),
            final_balance=float(result.final_balance),
            equity_curve=equity_curve,
            trades=trades,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/symbols", tags=["Data"])
async def list_symbols():
    """List available symbols for backtesting."""
    data_dir = Path("data/processed")
    
    if not data_dir.exists():
        return {"symbols": []}
    
    symbols = set()
    for f in data_dir.glob("*_*.parquet"):
        # Extract symbol from filename (e.g., "SPY_train.parquet" -> "SPY")
        parts = f.stem.split("_")
        if len(parts) >= 2:
            symbols.add(parts[0])
    
    return {"symbols": sorted(symbols)}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )

