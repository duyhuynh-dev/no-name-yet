"""
Trading Session

Orchestrates the trading workflow, connecting agents to execution.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import logging
import pandas as pd

from .base import BaseExchange, Order, OrderSide, OrderStatus, Position
from .order_manager import OrderManager, RiskLimits
from .execution_engine import ExecutionEngine
from ..agents import BaseAgent, AgentSignal, AgentAction, EnsembleAgent

logger = logging.getLogger(__name__)


@dataclass
class SessionConfig:
    """Trading session configuration."""
    symbols: List[str]
    position_size: float = 100  # Default shares per trade
    max_positions: int = 5
    trading_interval_seconds: int = 60
    use_ensemble: bool = False
    paper_trading: bool = True


@dataclass 
class SessionState:
    """Current session state."""
    is_running: bool = False
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    
    # Counters
    signals_generated: int = 0
    orders_submitted: int = 0
    orders_filled: int = 0
    
    # P&L
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Errors
    errors: List[str] = field(default_factory=list)


class TradingSession:
    """
    Trading Session that connects agents to real/paper trading.
    
    Features:
    - Agent signal processing
    - Position management
    - Risk-aware execution
    - Session monitoring
    """
    
    def __init__(
        self,
        exchange: BaseExchange,
        agent: BaseAgent,
        config: Optional[SessionConfig] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        """
        Initialize the Trading Session.
        
        Args:
            exchange: Exchange to trade on
            agent: Trading agent (single or ensemble)
            config: Session configuration
            risk_limits: Risk limits for order manager
        """
        self.exchange = exchange
        self.agent = agent
        self.config = config or SessionConfig(symbols=["SPY"])
        
        # Initialize components
        self.order_manager = OrderManager(exchange, risk_limits)
        self.execution_engine = ExecutionEngine(self.order_manager)
        
        # State
        self.state = SessionState()
        self._task: Optional[asyncio.Task] = None
        
        # Data storage
        self._market_data: Dict[str, pd.DataFrame] = {}
        self._signal_history: List[Dict[str, Any]] = []
        
        # Callbacks
        self._on_signal: List[Callable[[str, AgentSignal], None]] = []
        self._on_trade: List[Callable[[Order], None]] = []
    
    def on_signal(self, callback: Callable[[str, AgentSignal], None]) -> None:
        """Register callback for signals."""
        self._on_signal.append(callback)
    
    def on_trade(self, callback: Callable[[Order], None]) -> None:
        """Register callback for trades."""
        self._on_trade.append(callback)
        self.order_manager.on_fill(callback)
    
    async def start(self) -> None:
        """Start the trading session."""
        if self.state.is_running:
            logger.warning("Session already running")
            return
        
        # Connect to exchange
        if not self.exchange.is_connected:
            connected = await self.exchange.connect()
            if not connected:
                raise ConnectionError("Failed to connect to exchange")
        
        self.state.is_running = True
        self.state.started_at = datetime.now()
        
        logger.info(f"Trading session started for {self.config.symbols}")
        
        # Start main loop
        self._task = asyncio.create_task(self._trading_loop())
    
    async def stop(self) -> None:
        """Stop the trading session."""
        if not self.state.is_running:
            return
        
        self.state.is_running = False
        self.state.stopped_at = datetime.now()
        
        # Cancel main task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # Cancel pending orders
        await self.order_manager.cancel_all_orders()
        
        logger.info("Trading session stopped")
    
    async def _trading_loop(self) -> None:
        """Main trading loop."""
        try:
            while self.state.is_running:
                for symbol in self.config.symbols:
                    try:
                        await self._process_symbol(symbol)
                    except Exception as e:
                        error_msg = f"Error processing {symbol}: {e}"
                        logger.error(error_msg)
                        self.state.errors.append(error_msg)
                
                # Update unrealized P&L
                await self._update_pnl()
                
                # Wait for next interval
                await asyncio.sleep(self.config.trading_interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            self.state.errors.append(str(e))
    
    async def _process_symbol(self, symbol: str) -> None:
        """Process a single symbol."""
        # Get market data
        data = await self._get_market_data(symbol)
        if data is None or len(data) < 50:
            return
        
        # Get current position
        position = await self.exchange.get_position(symbol)
        current_position = 0
        if position:
            current_position = 1 if position.quantity > 0 else (-1 if position.quantity < 0 else 0)
        
        # Generate signal
        signal = self.agent.generate_signal(data, position=current_position)
        self.state.signals_generated += 1
        
        # Log signal
        self._signal_history.append({
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": signal.action.name,
            "confidence": signal.confidence,
            "strength": signal.strength,
            "current_position": current_position,
        })
        
        # Emit signal event
        for callback in self._on_signal:
            callback(symbol, signal)
        
        # Decide on action
        await self._execute_signal(symbol, signal, position)
    
    async def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get recent market data for a symbol."""
        try:
            bars = await self.exchange.get_bars(symbol, timeframe="1Min", limit=100)
            
            if not bars:
                # For paper trading, generate synthetic data
                quote = await self.exchange.get_quote(symbol)
                if quote and quote.get("last", 0) > 0:
                    # Create minimal dataframe
                    return self._generate_synthetic_data(quote["last"])
                return None
            
            df = pd.DataFrame(bars)
            df.columns = [c.lower() for c in df.columns]
            
            self._market_data[symbol] = df
            return df
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    def _generate_synthetic_data(self, current_price: float, n: int = 100) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        import numpy as np
        
        prices = [current_price]
        for i in range(n - 1):
            change = np.random.normal(0, current_price * 0.001)
            prices.append(prices[-1] + change)
        
        prices = np.array(prices[::-1])  # Reverse so current is last
        
        return pd.DataFrame({
            "open": prices * (1 + np.random.uniform(-0.001, 0.001, n)),
            "high": prices * (1 + np.random.uniform(0, 0.002, n)),
            "low": prices * (1 - np.random.uniform(0, 0.002, n)),
            "close": prices,
            "volume": np.random.uniform(1e5, 1e6, n),
        })
    
    async def _execute_signal(
        self,
        symbol: str,
        signal: AgentSignal,
        position: Optional[Position]
    ) -> None:
        """Execute a trading signal."""
        # Check confidence threshold
        if signal.confidence < 0.6:
            return
        
        # Determine action
        action = signal.action
        
        # Skip hold signals
        if action == AgentAction.HOLD:
            return
        
        # Check position limits
        positions = await self.exchange.get_positions()
        if len(positions) >= self.config.max_positions:
            if action in [AgentAction.BUY, AgentAction.STRONG_BUY]:
                if not position or position.quantity <= 0:
                    logger.info(f"Max positions reached, skipping buy for {symbol}")
                    return
        
        # Determine order side and quantity
        if action in [AgentAction.BUY, AgentAction.STRONG_BUY]:
            side = OrderSide.BUY
            # Double size for strong signals
            quantity = self.config.position_size * (1.5 if action == AgentAction.STRONG_BUY else 1.0)
        else:
            side = OrderSide.SELL
            quantity = self.config.position_size * (1.5 if action == AgentAction.STRONG_SELL else 1.0)
        
        # If we have a position, close it for opposite signals
        if position:
            if side == OrderSide.BUY and position.quantity < 0:
                # Close short position
                quantity = abs(position.quantity)
            elif side == OrderSide.SELL and position.quantity > 0:
                # Close long position
                quantity = position.quantity
        
        # Submit order
        order = await self.order_manager.submit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
        )
        
        self.state.orders_submitted += 1
        
        if order.is_filled:
            self.state.orders_filled += 1
            logger.info(
                f"Executed {side.value} {quantity} {symbol} @ {order.filled_avg_price:.2f}"
            )
    
    async def _update_pnl(self) -> None:
        """Update P&L tracking."""
        try:
            positions = await self.exchange.get_positions()
            self.state.unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        except Exception as e:
            logger.error(f"Failed to update P&L: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get session status."""
        runtime = None
        if self.state.started_at:
            end = self.state.stopped_at or datetime.now()
            runtime = (end - self.state.started_at).total_seconds()
        
        return {
            "is_running": self.state.is_running,
            "started_at": self.state.started_at.isoformat() if self.state.started_at else None,
            "stopped_at": self.state.stopped_at.isoformat() if self.state.stopped_at else None,
            "runtime_seconds": runtime,
            "symbols": self.config.symbols,
            "signals_generated": self.state.signals_generated,
            "orders_submitted": self.state.orders_submitted,
            "orders_filled": self.state.orders_filled,
            "realized_pnl": self.state.realized_pnl,
            "unrealized_pnl": self.state.unrealized_pnl,
            "total_pnl": self.state.realized_pnl + self.state.unrealized_pnl,
            "errors": self.state.errors[-10:],  # Last 10 errors
        }
    
    def get_signal_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent signal history."""
        return self._signal_history[-limit:]
    
    async def get_positions_summary(self) -> List[Dict[str, Any]]:
        """Get summary of current positions."""
        positions = await self.exchange.get_positions()
        return [p.to_dict() for p in positions]

