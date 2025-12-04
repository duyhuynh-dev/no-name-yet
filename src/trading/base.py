"""
Base Trading Infrastructure

Defines abstract base classes and data structures for trading operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import uuid


class OrderSide(Enum):
    """Order side (direction)."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Order lifecycle status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Order time in force."""
    DAY = "day"          # Valid for the trading day
    GTC = "gtc"          # Good till cancelled
    IOC = "ioc"          # Immediate or cancel
    FOK = "fok"          # Fill or kill
    OPG = "opg"          # Market on open
    CLS = "cls"          # Market on close


@dataclass
class Order:
    """
    Represents a trading order.
    """
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    
    # Auto-generated fields
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    client_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_avg_price: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    # Additional info
    exchange_order_id: Optional[str] = None
    fees: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIAL,
        ]
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def remaining_quantity(self) -> float:
        """Get unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_avg_price": self.filled_avg_price,
            "created_at": self.created_at.isoformat(),
            "fees": self.fees,
        }


@dataclass
class Position:
    """
    Represents a trading position.
    """
    symbol: str
    quantity: float  # Positive = long, Negative = short
    avg_entry_price: float
    current_price: float = 0.0
    
    # Calculated fields
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    realized_pnl: float = 0.0
    
    # Metadata
    opened_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_price(self, price: float) -> None:
        """Update position with current market price."""
        self.current_price = price
        self.market_value = self.quantity * price
        self.unrealized_pnl = self.quantity * (price - self.avg_entry_price)
        if self.avg_entry_price > 0:
            self.unrealized_pnl_pct = (price - self.avg_entry_price) / self.avg_entry_price * 100
        self.last_updated = datetime.now()
    
    @property
    def side(self) -> str:
        """Get position side."""
        if self.quantity > 0:
            return "long"
        elif self.quantity < 0:
            return "short"
        return "flat"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "side": self.side,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "realized_pnl": self.realized_pnl,
        }


@dataclass
class Trade:
    """
    Represents an executed trade.
    """
    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    fees: float = 0.0
    
    @property
    def value(self) -> float:
        """Get trade value."""
        return self.quantity * self.price
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "value": self.value,
            "fees": self.fees,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AccountInfo:
    """
    Trading account information.
    """
    account_id: str
    cash: float
    equity: float
    buying_power: float
    portfolio_value: float
    
    # Margin info
    margin_used: float = 0.0
    margin_available: float = 0.0
    
    # Day trading info
    day_trades_remaining: int = 3
    pattern_day_trader: bool = False
    
    # Status
    trading_blocked: bool = False
    account_blocked: bool = False
    
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "account_id": self.account_id,
            "cash": self.cash,
            "equity": self.equity,
            "buying_power": self.buying_power,
            "portfolio_value": self.portfolio_value,
            "margin_used": self.margin_used,
            "margin_available": self.margin_available,
            "trading_blocked": self.trading_blocked,
        }


class BaseExchange(ABC):
    """
    Abstract base class for exchange connectivity.
    
    All exchange implementations must inherit from this class
    and implement the required methods.
    """
    
    def __init__(
        self,
        name: str,
        paper_trading: bool = True,
        **kwargs
    ):
        """
        Initialize the exchange.
        
        Args:
            name: Exchange identifier
            paper_trading: Whether to use paper trading mode
            **kwargs: Exchange-specific configuration
        """
        self.name = name
        self.paper_trading = paper_trading
        self._connected = False
        self._callbacks: Dict[str, List[Callable]] = {
            "on_order_update": [],
            "on_trade": [],
            "on_position_update": [],
            "on_quote": [],
            "on_bar": [],
        }
    
    # Connection management
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the exchange.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the exchange."""
        pass
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to exchange."""
        return self._connected
    
    # Account operations
    
    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """Get account information."""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        pass
    
    # Order operations
    
    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the exchange.
        
        Args:
            order: Order to submit
            
        Returns:
            Updated order with exchange info
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            True if cancellation successful
        """
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        pass
    
    @abstractmethod
    async def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None
    ) -> List[Order]:
        """Get orders with optional filtering."""
        pass
    
    # Market data
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Dict[str, float]:
        """
        Get current quote for a symbol.
        
        Returns:
            Dict with 'bid', 'ask', 'last', 'volume'
        """
        pass
    
    @abstractmethod
    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical bars."""
        pass
    
    # Streaming (optional)
    
    async def subscribe_quotes(self, symbols: List[str]) -> None:
        """Subscribe to real-time quotes."""
        pass
    
    async def subscribe_trades(self, symbols: List[str]) -> None:
        """Subscribe to real-time trades."""
        pass
    
    async def subscribe_bars(self, symbols: List[str], timeframe: str = "1Min") -> None:
        """Subscribe to real-time bars."""
        pass
    
    # Callbacks
    
    def on(self, event: str, callback: Callable) -> None:
        """Register event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit(self, event: str, data: Any) -> None:
        """Emit event to registered callbacks."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    print(f"Callback error for {event}: {e}")
    
    # Utility methods
    
    def create_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float
    ) -> Order:
        """Create a market order."""
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )
    
    def create_limit_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        limit_price: float
    ) -> Order:
        """Create a limit order."""
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
        )
    
    def create_stop_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        stop_price: float
    ) -> Order:
        """Create a stop order."""
        return Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.STOP,
            stop_price=stop_price,
        )
    
    def __repr__(self) -> str:
        mode = "paper" if self.paper_trading else "live"
        status = "connected" if self._connected else "disconnected"
        return f"{self.__class__.__name__}(name='{self.name}', mode={mode}, status={status})"

