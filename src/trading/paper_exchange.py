"""
Paper Trading Exchange

Simulates a trading exchange for testing strategies without real money.
Includes realistic slippage, fees, and order filling simulation.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import uuid

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


class PaperExchange(BaseExchange):
    """
    Paper trading exchange that simulates order execution.
    
    Features:
    - Realistic slippage modeling
    - Configurable fees
    - Market impact simulation
    - Order book simulation
    """
    
    def __init__(
        self,
        name: str = "paper",
        initial_cash: float = 100000.0,
        fee_rate: float = 0.001,  # 0.1% per trade
        slippage_rate: float = 0.0005,  # 0.05% slippage
        fill_probability: float = 0.95,  # 95% fill rate for limit orders
        **kwargs
    ):
        """
        Initialize the paper exchange.
        
        Args:
            name: Exchange identifier
            initial_cash: Starting cash balance
            fee_rate: Trading fee as decimal (0.001 = 0.1%)
            slippage_rate: Slippage as decimal
            fill_probability: Probability of limit order fill
        """
        super().__init__(name=name, paper_trading=True, **kwargs)
        
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.fill_probability = fill_probability
        
        # State
        self._cash = initial_cash
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._trades: List[Trade] = []
        self._prices: Dict[str, float] = {}  # Simulated prices
        
        # Statistics
        self._total_fees = 0.0
        self._total_trades = 0
    
    async def connect(self) -> bool:
        """Connect to paper exchange (always succeeds)."""
        self._connected = True
        return True
    
    async def disconnect(self) -> None:
        """Disconnect from paper exchange."""
        self._connected = False
    
    def set_price(self, symbol: str, price: float) -> None:
        """
        Set the current price for a symbol.
        
        Args:
            symbol: Symbol to set price for
            price: Current price
        """
        self._prices[symbol] = price
        
        # Update positions
        if symbol in self._positions:
            self._positions[symbol].update_price(price)
    
    def set_prices(self, prices: Dict[str, float]) -> None:
        """Set prices for multiple symbols."""
        for symbol, price in prices.items():
            self.set_price(symbol, price)
    
    async def get_account(self) -> AccountInfo:
        """Get account information."""
        # Calculate portfolio value
        positions_value = sum(
            pos.market_value for pos in self._positions.values()
        )
        equity = self._cash + positions_value
        
        return AccountInfo(
            account_id="paper_account",
            cash=self._cash,
            equity=equity,
            buying_power=self._cash,  # Simplified: no margin
            portfolio_value=equity,
            margin_used=0.0,
            margin_available=self._cash,
            last_updated=datetime.now(),
        )
    
    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self._positions.values())
    
    async def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        return self._positions.get(symbol)
    
    def _calculate_slippage(
        self,
        price: float,
        side: OrderSide,
        quantity: float
    ) -> float:
        """
        Calculate slippage based on order size and side.
        
        Slippage is worse for larger orders and works against the trader.
        """
        # Base slippage
        base_slip = price * self.slippage_rate
        
        # Size impact (larger orders = more slippage)
        size_factor = 1.0 + (quantity / 1000) * 0.1  # 10% more per 1000 shares
        
        # Random component
        random_factor = np.random.uniform(0.5, 1.5)
        
        slippage = base_slip * size_factor * random_factor
        
        # Slippage direction (against trader)
        if side == OrderSide.BUY:
            return slippage  # Pay more
        else:
            return -slippage  # Receive less
    
    def _calculate_fees(self, quantity: float, price: float) -> float:
        """Calculate trading fees."""
        return quantity * price * self.fee_rate
    
    async def _execute_market_order(self, order: Order) -> Order:
        """Execute a market order immediately."""
        if order.symbol not in self._prices:
            order.status = OrderStatus.REJECTED
            order.metadata["reject_reason"] = "No price available"
            return order
        
        base_price = self._prices[order.symbol]
        slippage = self._calculate_slippage(base_price, order.side, order.quantity)
        fill_price = base_price + slippage
        
        # Check if we have enough buying power
        order_value = order.quantity * fill_price
        fees = self._calculate_fees(order.quantity, fill_price)
        
        if order.side == OrderSide.BUY:
            if order_value + fees > self._cash:
                order.status = OrderStatus.REJECTED
                order.metadata["reject_reason"] = "Insufficient funds"
                return order
        
        # Execute the order
        order.filled_quantity = order.quantity
        order.filled_avg_price = fill_price
        order.fees = fees
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        
        # Update cash
        if order.side == OrderSide.BUY:
            self._cash -= (order_value + fees)
        else:
            self._cash += (order_value - fees)
        
        # Update position
        await self._update_position(order)
        
        # Record trade
        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            timestamp=datetime.now(),
            fees=fees,
        )
        self._trades.append(trade)
        self._total_trades += 1
        self._total_fees += fees
        
        # Emit events
        self._emit("on_trade", trade)
        self._emit("on_order_update", order)
        
        return order
    
    async def _execute_limit_order(self, order: Order) -> Order:
        """Check and potentially execute a limit order."""
        if order.symbol not in self._prices:
            return order  # Keep pending
        
        current_price = self._prices[order.symbol]
        
        # Check if limit price is hit
        should_fill = False
        if order.side == OrderSide.BUY and current_price <= order.limit_price:
            should_fill = True
        elif order.side == OrderSide.SELL and current_price >= order.limit_price:
            should_fill = True
        
        if should_fill and np.random.random() < self.fill_probability:
            # Fill at limit price (no slippage for limit orders)
            order.filled_quantity = order.quantity
            order.filled_avg_price = order.limit_price
            order.fees = self._calculate_fees(order.quantity, order.limit_price)
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            
            # Update cash and position
            order_value = order.quantity * order.limit_price
            if order.side == OrderSide.BUY:
                self._cash -= (order_value + order.fees)
            else:
                self._cash += (order_value - order.fees)
            
            await self._update_position(order)
            
            # Record trade
            trade = Trade(
                trade_id=str(uuid.uuid4())[:8],
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                price=order.limit_price,
                timestamp=datetime.now(),
                fees=order.fees,
            )
            self._trades.append(trade)
            self._emit("on_trade", trade)
            self._emit("on_order_update", order)
        
        return order
    
    async def _update_position(self, order: Order) -> None:
        """Update position after order execution."""
        symbol = order.symbol
        fill_qty = order.filled_quantity
        fill_price = order.filled_avg_price
        
        if order.side == OrderSide.SELL:
            fill_qty = -fill_qty
        
        if symbol in self._positions:
            pos = self._positions[symbol]
            old_qty = pos.quantity
            new_qty = old_qty + fill_qty
            
            if new_qty == 0:
                # Position closed
                pnl = (fill_price - pos.avg_entry_price) * old_qty
                if order.side == OrderSide.SELL:
                    pnl = (fill_price - pos.avg_entry_price) * abs(fill_qty)
                pos.realized_pnl += pnl
                del self._positions[symbol]
            else:
                # Update position
                if (old_qty > 0 and fill_qty > 0) or (old_qty < 0 and fill_qty < 0):
                    # Adding to position
                    total_cost = (pos.avg_entry_price * abs(old_qty) + 
                                  fill_price * abs(fill_qty))
                    pos.avg_entry_price = total_cost / abs(new_qty)
                pos.quantity = new_qty
                pos.update_price(self._prices.get(symbol, fill_price))
        else:
            # New position
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=fill_qty,
                avg_entry_price=fill_price,
                current_price=fill_price,
            )
        
        self._emit("on_position_update", self._positions.get(symbol))
    
    async def submit_order(self, order: Order) -> Order:
        """Submit an order to the paper exchange."""
        order.submitted_at = datetime.now()
        order.status = OrderStatus.SUBMITTED
        
        self._orders[order.order_id] = order
        
        # Execute based on order type
        if order.order_type == OrderType.MARKET:
            order = await self._execute_market_order(order)
        elif order.order_type == OrderType.LIMIT:
            order.status = OrderStatus.ACCEPTED
            # Limit orders checked periodically
        elif order.order_type == OrderType.STOP:
            order.status = OrderStatus.ACCEPTED
        
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if order_id not in self._orders:
            return False
        
        order = self._orders[order_id]
        if not order.is_active:
            return False
        
        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now()
        self._emit("on_order_update", order)
        
        return True
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self._orders.get(order_id)
    
    async def get_orders(
        self,
        status: Optional[OrderStatus] = None,
        symbol: Optional[str] = None
    ) -> List[Order]:
        """Get orders with optional filtering."""
        orders = list(self._orders.values())
        
        if status:
            orders = [o for o in orders if o.status == status]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return orders
    
    async def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get current quote for a symbol."""
        price = self._prices.get(symbol, 0.0)
        spread = price * 0.0001  # 1 bps spread
        
        return {
            "bid": price - spread,
            "ask": price + spread,
            "last": price,
            "volume": 1000000,  # Simulated
        }
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get historical bars (returns empty for paper trading)."""
        return []
    
    def get_trades(self) -> List[Trade]:
        """Get all executed trades."""
        return self._trades.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trading statistics."""
        account = asyncio.get_event_loop().run_until_complete(self.get_account())
        
        total_pnl = account.equity - self.initial_cash
        total_return = (total_pnl / self.initial_cash) * 100
        
        winning_trades = [t for t in self._trades if t.side == OrderSide.SELL]
        # Simplified win rate calculation
        
        return {
            "initial_cash": self.initial_cash,
            "current_equity": account.equity,
            "total_pnl": total_pnl,
            "total_return_pct": total_return,
            "total_trades": self._total_trades,
            "total_fees": self._total_fees,
            "open_positions": len(self._positions),
        }
    
    def reset(self) -> None:
        """Reset the paper exchange to initial state."""
        self._cash = self.initial_cash
        self._positions = {}
        self._orders = {}
        self._trades = []
        self._prices = {}
        self._total_fees = 0.0
        self._total_trades = 0

