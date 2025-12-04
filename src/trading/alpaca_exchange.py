"""
Alpaca Exchange Integration

Connects to Alpaca Trading API for stocks and crypto trading.
Supports both paper and live trading modes.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from .base import (
    BaseExchange,
    Order,
    Position,
    Trade,
    AccountInfo,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
)

logger = logging.getLogger(__name__)


class AlpacaExchange(BaseExchange):
    """
    Alpaca Trading API integration.
    
    Supports:
    - Paper and live trading
    - Stocks and crypto
    - Market, limit, stop orders
    - Real-time streaming
    
    Requires:
    - ALPACA_API_KEY
    - ALPACA_SECRET_KEY
    - ALPACA_PAPER (optional, defaults to True)
    """
    
    def __init__(
        self,
        name: str = "alpaca",
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper_trading: bool = True,
        **kwargs
    ):
        """
        Initialize the Alpaca exchange.
        
        Args:
            name: Exchange identifier
            api_key: Alpaca API key (or set ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (or set ALPACA_SECRET_KEY env var)
            paper_trading: Whether to use paper trading
        """
        super().__init__(name=name, paper_trading=paper_trading, **kwargs)
        
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        
        # API clients (lazy loaded)
        self._trading_client = None
        self._data_client = None
        self._stream_client = None
        
        # Cache
        self._account_cache: Optional[AccountInfo] = None
        self._positions_cache: Dict[str, Position] = {}
        self._orders_cache: Dict[str, Order] = {}
    
    def _ensure_credentials(self) -> None:
        """Ensure API credentials are available."""
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_SECRET_KEY environment variables or pass to constructor."
            )
    
    async def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            self._ensure_credentials()
            
            # Import Alpaca SDK
            try:
                from alpaca.trading.client import TradingClient
                from alpaca.data.historical import StockHistoricalDataClient
            except ImportError:
                logger.error("alpaca-py package not installed. Install with: pip install alpaca-py")
                return False
            
            # Initialize clients
            self._trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper_trading,
            )
            
            self._data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
            )
            
            # Test connection by getting account
            account = self._trading_client.get_account()
            self._connected = True
            
            logger.info(f"Connected to Alpaca ({'paper' if self.paper_trading else 'live'})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            self._connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        self._trading_client = None
        self._data_client = None
        self._connected = False
        logger.info("Disconnected from Alpaca")
    
    async def get_account(self) -> AccountInfo:
        """Get account information from Alpaca."""
        if not self._trading_client:
            raise ConnectionError("Not connected to Alpaca")
        
        account = self._trading_client.get_account()
        
        return AccountInfo(
            account_id=account.account_number,
            cash=float(account.cash),
            equity=float(account.equity),
            buying_power=float(account.buying_power),
            portfolio_value=float(account.portfolio_value),
            margin_used=float(account.initial_margin) if account.initial_margin else 0.0,
            margin_available=float(account.regt_buying_power) if account.regt_buying_power else 0.0,
            day_trades_remaining=account.daytrade_count if hasattr(account, 'daytrade_count') else 3,
            pattern_day_trader=account.pattern_day_trader,
            trading_blocked=account.trading_blocked,
            account_blocked=account.account_blocked,
            last_updated=datetime.now(),
        )
    
    async def get_positions(self) -> List[Position]:
        """Get all open positions from Alpaca."""
        if not self._trading_client:
            raise ConnectionError("Not connected to Alpaca")
        
        positions = self._trading_client.get_all_positions()
        
        result = []
        for pos in positions:
            position = Position(
                symbol=pos.symbol,
                quantity=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_pct=float(pos.unrealized_plpc) * 100,
            )
            result.append(position)
            self._positions_cache[pos.symbol] = position
        
        return result
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        if not self._trading_client:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            pos = self._trading_client.get_open_position(symbol)
            return Position(
                symbol=pos.symbol,
                quantity=float(pos.qty),
                avg_entry_price=float(pos.avg_entry_price),
                current_price=float(pos.current_price),
                market_value=float(pos.market_value),
                unrealized_pnl=float(pos.unrealized_pl),
                unrealized_pnl_pct=float(pos.unrealized_plpc) * 100,
            )
        except Exception:
            return None
    
    def _convert_order_type(self, order_type: OrderType) -> str:
        """Convert our OrderType to Alpaca's order type."""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit",
            OrderType.TRAILING_STOP: "trailing_stop",
        }
        return mapping.get(order_type, "market")
    
    def _convert_time_in_force(self, tif: TimeInForce) -> str:
        """Convert our TimeInForce to Alpaca's."""
        mapping = {
            TimeInForce.DAY: "day",
            TimeInForce.GTC: "gtc",
            TimeInForce.IOC: "ioc",
            TimeInForce.FOK: "fok",
            TimeInForce.OPG: "opg",
            TimeInForce.CLS: "cls",
        }
        return mapping.get(tif, "day")
    
    async def submit_order(self, order: Order) -> Order:
        """Submit an order to Alpaca."""
        if not self._trading_client:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                StopOrderRequest,
                StopLimitOrderRequest,
            )
            from alpaca.trading.enums import OrderSide as AlpacaOrderSide, TimeInForce as AlpacaTIF
            
            # Build order request based on type
            side = AlpacaOrderSide.BUY if order.side == OrderSide.BUY else AlpacaOrderSide.SELL
            tif = AlpacaTIF(self._convert_time_in_force(order.time_in_force))
            
            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                )
            elif order.order_type == OrderType.LIMIT:
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                )
            elif order.order_type == OrderType.STOP:
                request = StopOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    stop_price=order.stop_price,
                )
            elif order.order_type == OrderType.STOP_LIMIT:
                request = StopLimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=tif,
                    stop_price=order.stop_price,
                    limit_price=order.limit_price,
                )
            else:
                raise ValueError(f"Unsupported order type: {order.order_type}")
            
            # Submit order
            alpaca_order = self._trading_client.submit_order(request)
            
            # Update our order with Alpaca's response
            order.exchange_order_id = str(alpaca_order.id)
            order.submitted_at = datetime.now()
            order.status = self._convert_alpaca_status(alpaca_order.status)
            
            if alpaca_order.filled_qty:
                order.filled_quantity = float(alpaca_order.filled_qty)
            if alpaca_order.filled_avg_price:
                order.filled_avg_price = float(alpaca_order.filled_avg_price)
            
            self._orders_cache[order.order_id] = order
            self._emit("on_order_update", order)
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to submit order: {e}")
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = str(e)
            return order
    
    def _convert_alpaca_status(self, status) -> OrderStatus:
        """Convert Alpaca order status to our OrderStatus."""
        status_str = str(status).lower()
        mapping = {
            "new": OrderStatus.SUBMITTED,
            "accepted": OrderStatus.ACCEPTED,
            "pending_new": OrderStatus.PENDING,
            "partially_filled": OrderStatus.PARTIAL,
            "filled": OrderStatus.FILLED,
            "canceled": OrderStatus.CANCELLED,
            "cancelled": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "expired": OrderStatus.EXPIRED,
        }
        return mapping.get(status_str, OrderStatus.PENDING)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order on Alpaca."""
        if not self._trading_client:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            # Find the exchange order ID
            if order_id in self._orders_cache:
                exchange_id = self._orders_cache[order_id].exchange_order_id
            else:
                exchange_id = order_id
            
            self._trading_client.cancel_order_by_id(exchange_id)
            
            if order_id in self._orders_cache:
                self._orders_cache[order_id].status = OrderStatus.CANCELLED
                self._orders_cache[order_id].cancelled_at = datetime.now()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders_cache.get(order_id)
    
    async def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None
    ) -> List[Order]:
        """Get orders with optional filtering."""
        orders = list(self._orders_cache.values())
        
        if status:
            orders = [o for o in orders if o.status == status]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return orders
    
    async def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get current quote for a symbol."""
        if not self._data_client:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self._data_client.get_stock_latest_quote(request)
            
            quote = quotes[symbol]
            return {
                "bid": float(quote.bid_price),
                "ask": float(quote.ask_price),
                "last": (float(quote.bid_price) + float(quote.ask_price)) / 2,
                "volume": float(quote.bid_size) + float(quote.ask_size),
            }
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return {"bid": 0, "ask": 0, "last": 0, "volume": 0}
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical bars from Alpaca."""
        if not self._data_client:
            raise ConnectionError("Not connected to Alpaca")
        
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
            
            # Parse timeframe
            tf_mapping = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day,
            }
            tf = tf_mapping.get(timeframe, TimeFrame.Minute)
            
            end = datetime.now()
            start = end - timedelta(days=7)  # Last 7 days
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                limit=limit,
            )
            
            bars = self._data_client.get_stock_bars(request)
            
            result = []
            for bar in bars[symbol]:
                result.append({
                    "timestamp": bar.timestamp.isoformat(),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                })
            
            return result[-limit:]
            
        except Exception as e:
            logger.error(f"Failed to get bars for {symbol}: {e}")
            return []

