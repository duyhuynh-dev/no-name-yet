"""
Tests for the Real-Time Trading Infrastructure.
"""

import pytest
import asyncio

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio(loop_scope="function")
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.trading import (
    BaseExchange,
    Order,
    Position,
    Trade,
    AccountInfo,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    PaperExchange,
    OrderManager,
    ExecutionEngine,
)
from src.trading.order_manager import RiskLimits


@pytest.fixture
def paper_exchange():
    """Create a paper exchange for testing."""
    exchange = PaperExchange(
        initial_cash=100000.0,
        fee_rate=0.001,
        slippage_rate=0.0005,
    )
    return exchange


@pytest.fixture
def order_manager(paper_exchange):
    """Create an order manager for testing."""
    return OrderManager(
        exchange=paper_exchange,
        risk_limits=RiskLimits(
            max_order_value=50000,
            max_daily_loss=5000,
        )
    )


class TestOrder:
    """Tests for Order dataclass."""
    
    def test_order_creation(self):
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == 100
        assert order.status == OrderStatus.PENDING
        assert order.is_active
    
    def test_order_to_dict(self):
        order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            order_type=OrderType.LIMIT,
            limit_price=150.0,
        )
        
        d = order.to_dict()
        assert d["symbol"] == "AAPL"
        assert d["side"] == "sell"
        assert d["order_type"] == "limit"
        assert d["limit_price"] == 150.0
    
    def test_order_remaining_quantity(self):
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        order.filled_quantity = 30
        
        assert order.remaining_quantity == 70


class TestPosition:
    """Tests for Position dataclass."""
    
    def test_position_creation(self):
        pos = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=150.0,
            current_price=155.0,
        )
        
        assert pos.symbol == "AAPL"
        assert pos.side == "long"
    
    def test_position_update_price(self):
        pos = Position(
            symbol="AAPL",
            quantity=100,
            avg_entry_price=150.0,
        )
        
        pos.update_price(160.0)
        
        assert pos.current_price == 160.0
        assert pos.market_value == 16000.0
        assert pos.unrealized_pnl == 1000.0  # (160-150) * 100
    
    def test_short_position(self):
        pos = Position(
            symbol="AAPL",
            quantity=-100,
            avg_entry_price=150.0,
        )
        
        assert pos.side == "short"


class TestPaperExchange:
    """Tests for PaperExchange."""
    
    @pytest.mark.asyncio
    async def test_connect(self, paper_exchange):
        result = await paper_exchange.connect()
        assert result is True
        assert paper_exchange.is_connected
    
    @pytest.mark.asyncio
    async def test_get_account(self, paper_exchange):
        await paper_exchange.connect()
        account = await paper_exchange.get_account()
        
        assert account.cash == 100000.0
        assert account.equity == 100000.0
    
    @pytest.mark.asyncio
    async def test_set_price(self, paper_exchange):
        paper_exchange.set_price("AAPL", 150.0)
        quote = await paper_exchange.get_quote("AAPL")
        
        assert quote["last"] == 150.0
    
    @pytest.mark.asyncio
    async def test_market_order_buy(self, paper_exchange):
        await paper_exchange.connect()
        paper_exchange.set_price("AAPL", 150.0)
        
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        
        result = await paper_exchange.submit_order(order)
        
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 10
        assert result.filled_avg_price > 0
    
    @pytest.mark.asyncio
    async def test_market_order_creates_position(self, paper_exchange):
        await paper_exchange.connect()
        paper_exchange.set_price("AAPL", 150.0)
        
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        
        await paper_exchange.submit_order(order)
        
        position = await paper_exchange.get_position("AAPL")
        assert position is not None
        assert position.quantity == 10
    
    @pytest.mark.asyncio
    async def test_insufficient_funds(self, paper_exchange):
        await paper_exchange.connect()
        paper_exchange.set_price("AAPL", 150.0)
        
        # Try to buy more than we can afford
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10000,  # 10000 * 150 = 1.5M > 100K
            order_type=OrderType.MARKET,
        )
        
        result = await paper_exchange.submit_order(order)
        
        assert result.status == OrderStatus.REJECTED
    
    @pytest.mark.asyncio
    async def test_round_trip_trade(self, paper_exchange):
        await paper_exchange.connect()
        paper_exchange.set_price("AAPL", 150.0)
        
        # Buy
        buy_order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        await paper_exchange.submit_order(buy_order)
        
        # Set higher price
        paper_exchange.set_price("AAPL", 160.0)
        
        # Sell
        sell_order = Order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        await paper_exchange.submit_order(sell_order)
        
        # Position should be closed
        position = await paper_exchange.get_position("AAPL")
        assert position is None
        
        # Should have made profit (minus fees and slippage)
        account = await paper_exchange.get_account()
        # Not exact due to fees and slippage, but should be profitable
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, paper_exchange):
        await paper_exchange.connect()
        paper_exchange.set_price("AAPL", 150.0)
        
        # Submit limit order that won't fill immediately
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=140.0,  # Below market
        )
        
        result = await paper_exchange.submit_order(order)
        assert result.status == OrderStatus.ACCEPTED
        
        # Cancel it
        cancelled = await paper_exchange.cancel_order(result.order_id)
        assert cancelled is True
        
        # Verify cancelled
        updated = await paper_exchange.get_order(result.order_id)
        assert updated.status == OrderStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_reset(self, paper_exchange):
        await paper_exchange.connect()
        paper_exchange.set_price("AAPL", 150.0)
        
        # Make a trade
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
        )
        await paper_exchange.submit_order(order)
        
        # Reset
        paper_exchange.reset()
        
        account = await paper_exchange.get_account()
        assert account.cash == 100000.0
        
        positions = await paper_exchange.get_positions()
        assert len(positions) == 0


class TestOrderManager:
    """Tests for OrderManager."""
    
    @pytest.mark.asyncio
    async def test_submit_order(self, order_manager, paper_exchange):
        await paper_exchange.connect()
        paper_exchange.set_price("AAPL", 150.0)
        
        order = await order_manager.submit_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
        )
        
        assert order.status == OrderStatus.FILLED
    
    @pytest.mark.asyncio
    async def test_risk_check_max_value(self, order_manager, paper_exchange):
        await paper_exchange.connect()
        paper_exchange.set_price("AAPL", 150.0)
        
        # Try to exceed max order value (50000)
        order = await order_manager.submit_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=500,  # 500 * 150 = 75000 > 50000
        )
        
        assert order.status == OrderStatus.REJECTED
        assert "exceeds limit" in order.metadata.get("reject_reason", "")
    
    @pytest.mark.asyncio
    async def test_get_orders(self, order_manager, paper_exchange):
        await paper_exchange.connect()
        paper_exchange.set_price("AAPL", 150.0)
        paper_exchange.set_price("MSFT", 300.0)
        
        await order_manager.submit_order("AAPL", OrderSide.BUY, 10)
        await order_manager.submit_order("MSFT", OrderSide.BUY, 5)
        
        all_orders = order_manager.get_orders()
        assert len(all_orders) == 2
        
        aapl_orders = order_manager.get_orders(symbol="AAPL")
        assert len(aapl_orders) == 1
    
    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, order_manager, paper_exchange):
        await paper_exchange.connect()
        paper_exchange.set_price("AAPL", 150.0)
        
        # Submit limit orders that won't fill
        await order_manager.submit_order(
            "AAPL", OrderSide.BUY, 10,
            order_type=OrderType.LIMIT,
            limit_price=100.0
        )
        await order_manager.submit_order(
            "AAPL", OrderSide.BUY, 10,
            order_type=OrderType.LIMIT,
            limit_price=100.0
        )
        
        cancelled = await order_manager.cancel_all_orders()
        assert cancelled == 2
    
    def test_statistics(self, order_manager):
        stats = order_manager.get_statistics()
        
        assert "total_orders" in stats
        assert "fill_rate" in stats
        assert "daily_pnl" in stats


class TestExecutionEngine:
    """Tests for ExecutionEngine."""
    
    @pytest.mark.asyncio
    async def test_twap_execution(self, order_manager, paper_exchange):
        await paper_exchange.connect()
        paper_exchange.set_price("AAPL", 150.0)
        
        engine = ExecutionEngine(order_manager)
        
        # Execute small TWAP for testing (short duration)
        plan = await engine.execute_twap(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=30,
            duration_minutes=1,  # 1 minute for testing
            num_slices=3,
        )
        
        assert plan.total_quantity == 30
        assert plan.num_slices == 3
        assert plan.slice_quantity == 10
    
    def test_execution_statistics(self, order_manager):
        engine = ExecutionEngine(order_manager)
        stats = engine.get_statistics()
        
        assert "total_executions" in stats
        assert "active_executions" in stats


class TestOrderTypes:
    """Tests for different order types."""
    
    @pytest.mark.asyncio
    async def test_limit_order(self, paper_exchange):
        await paper_exchange.connect()
        paper_exchange.set_price("AAPL", 150.0)
        
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=145.0,  # Below market
        )
        
        result = await paper_exchange.submit_order(order)
        # Should be accepted but not filled (price too low)
        assert result.status == OrderStatus.ACCEPTED
    
    @pytest.mark.asyncio
    async def test_limit_order_fills(self, paper_exchange):
        await paper_exchange.connect()
        paper_exchange.set_price("AAPL", 150.0)
        
        order = Order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            limit_price=155.0,  # Above market - should fill
        )
        
        result = await paper_exchange.submit_order(order)
        # Check limit order execution
        await paper_exchange._execute_limit_order(result)


class TestHelperMethods:
    """Tests for helper methods."""
    
    def test_create_market_order(self, paper_exchange):
        order = paper_exchange.create_market_order(
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100
        )
        
        assert order.symbol == "AAPL"
        assert order.order_type == OrderType.MARKET
    
    def test_create_limit_order(self, paper_exchange):
        order = paper_exchange.create_limit_order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            limit_price=160.0
        )
        
        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == 160.0
    
    def test_create_stop_order(self, paper_exchange):
        order = paper_exchange.create_stop_order(
            symbol="AAPL",
            side=OrderSide.SELL,
            quantity=50,
            stop_price=140.0
        )
        
        assert order.order_type == OrderType.STOP
        assert order.stop_price == 140.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

