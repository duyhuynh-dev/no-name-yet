"""
Execution Engine

Handles order execution strategies including TWAP, VWAP, and smart routing.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

from .base import Order, OrderSide, OrderType, OrderStatus
from .order_manager import OrderManager

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Order execution strategies."""
    MARKET = "market"           # Execute immediately at market
    TWAP = "twap"              # Time-Weighted Average Price
    VWAP = "vwap"              # Volume-Weighted Average Price
    ICEBERG = "iceberg"        # Hidden quantity
    ADAPTIVE = "adaptive"      # Adapt to market conditions


@dataclass
class ExecutionPlan:
    """Execution plan for an order."""
    strategy: ExecutionStrategy
    total_quantity: float
    symbol: str
    side: OrderSide
    
    # Timing
    start_time: datetime = None
    end_time: datetime = None
    duration_minutes: int = 30
    
    # Slicing
    num_slices: int = 10
    slice_quantity: float = 0.0
    slice_interval_seconds: float = 0.0
    
    # State
    executed_quantity: float = 0.0
    avg_price: float = 0.0
    child_orders: List[str] = None
    
    def __post_init__(self):
        if self.child_orders is None:
            self.child_orders = []
        if self.start_time is None:
            self.start_time = datetime.now()
        if self.end_time is None:
            self.end_time = self.start_time + timedelta(minutes=self.duration_minutes)
        
        # Calculate slicing
        self.slice_quantity = self.total_quantity / self.num_slices
        total_seconds = self.duration_minutes * 60
        self.slice_interval_seconds = total_seconds / self.num_slices
    
    @property
    def remaining_quantity(self) -> float:
        return self.total_quantity - self.executed_quantity
    
    @property
    def completion_pct(self) -> float:
        return (self.executed_quantity / self.total_quantity * 100) if self.total_quantity > 0 else 0
    
    @property
    def is_complete(self) -> bool:
        return self.executed_quantity >= self.total_quantity


class ExecutionEngine:
    """
    Execution Engine for algorithmic order execution.
    
    Features:
    - TWAP execution
    - VWAP execution (simulated)
    - Iceberg orders
    - Execution monitoring
    """
    
    def __init__(self, order_manager: OrderManager):
        """
        Initialize the Execution Engine.
        
        Args:
            order_manager: Order manager for submitting orders
        """
        self.order_manager = order_manager
        
        # Active executions
        self._active_plans: Dict[str, ExecutionPlan] = {}
        self._execution_tasks: Dict[str, asyncio.Task] = {}
        
        # Execution history
        self._completed_plans: List[ExecutionPlan] = []
    
    async def execute_twap(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        duration_minutes: int = 30,
        num_slices: int = 10,
    ) -> ExecutionPlan:
        """
        Execute an order using TWAP strategy.
        
        Spreads the order evenly over time to minimize market impact.
        
        Args:
            symbol: Symbol to trade
            side: Buy or sell
            quantity: Total quantity
            duration_minutes: Execution window
            num_slices: Number of child orders
            
        Returns:
            Execution plan
        """
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.TWAP,
            total_quantity=quantity,
            symbol=symbol,
            side=side,
            duration_minutes=duration_minutes,
            num_slices=num_slices,
        )
        
        plan_id = f"twap_{symbol}_{datetime.now().strftime('%H%M%S')}"
        self._active_plans[plan_id] = plan
        
        # Start execution task
        task = asyncio.create_task(self._execute_twap_plan(plan_id, plan))
        self._execution_tasks[plan_id] = task
        
        logger.info(f"Started TWAP execution: {plan_id}")
        return plan
    
    async def _execute_twap_plan(self, plan_id: str, plan: ExecutionPlan) -> None:
        """Execute TWAP plan."""
        try:
            for i in range(plan.num_slices):
                if plan.is_complete:
                    break
                
                # Calculate slice quantity (adjust last slice)
                slice_qty = min(plan.slice_quantity, plan.remaining_quantity)
                
                if slice_qty <= 0:
                    break
                
                # Submit child order
                order = await self.order_manager.submit_order(
                    symbol=plan.symbol,
                    side=plan.side,
                    quantity=slice_qty,
                    order_type=OrderType.MARKET,
                )
                
                plan.child_orders.append(order.order_id)
                
                if order.is_filled:
                    plan.executed_quantity += order.filled_quantity
                    # Update average price
                    if plan.avg_price == 0:
                        plan.avg_price = order.filled_avg_price
                    else:
                        total_value = (plan.avg_price * (plan.executed_quantity - order.filled_quantity) +
                                      order.filled_avg_price * order.filled_quantity)
                        plan.avg_price = total_value / plan.executed_quantity
                
                logger.info(
                    f"TWAP {plan_id}: Slice {i+1}/{plan.num_slices}, "
                    f"executed {plan.executed_quantity}/{plan.total_quantity}"
                )
                
                # Wait for next slice (except for last)
                if i < plan.num_slices - 1:
                    await asyncio.sleep(plan.slice_interval_seconds)
            
            logger.info(f"TWAP execution complete: {plan_id}")
            
        except asyncio.CancelledError:
            logger.info(f"TWAP execution cancelled: {plan_id}")
        except Exception as e:
            logger.error(f"TWAP execution error: {e}")
        finally:
            self._active_plans.pop(plan_id, None)
            self._execution_tasks.pop(plan_id, None)
            self._completed_plans.append(plan)
    
    async def execute_iceberg(
        self,
        symbol: str,
        side: OrderSide,
        total_quantity: float,
        visible_quantity: float,
        limit_price: float,
    ) -> ExecutionPlan:
        """
        Execute an iceberg order.
        
        Shows only a portion of the total quantity at a time.
        
        Args:
            symbol: Symbol to trade
            side: Buy or sell
            total_quantity: Total quantity to execute
            visible_quantity: Quantity to show at a time
            limit_price: Limit price for orders
            
        Returns:
            Execution plan
        """
        num_slices = int(np.ceil(total_quantity / visible_quantity))
        
        plan = ExecutionPlan(
            strategy=ExecutionStrategy.ICEBERG,
            total_quantity=total_quantity,
            symbol=symbol,
            side=side,
            num_slices=num_slices,
            slice_quantity=visible_quantity,
        )
        
        plan_id = f"iceberg_{symbol}_{datetime.now().strftime('%H%M%S')}"
        self._active_plans[plan_id] = plan
        
        # Start execution task
        task = asyncio.create_task(
            self._execute_iceberg_plan(plan_id, plan, limit_price)
        )
        self._execution_tasks[plan_id] = task
        
        logger.info(f"Started Iceberg execution: {plan_id}")
        return plan
    
    async def _execute_iceberg_plan(
        self,
        plan_id: str,
        plan: ExecutionPlan,
        limit_price: float
    ) -> None:
        """Execute iceberg plan."""
        try:
            while not plan.is_complete:
                slice_qty = min(plan.slice_quantity, plan.remaining_quantity)
                
                if slice_qty <= 0:
                    break
                
                # Submit visible slice as limit order
                order = await self.order_manager.submit_order(
                    symbol=plan.symbol,
                    side=plan.side,
                    quantity=slice_qty,
                    order_type=OrderType.LIMIT,
                    limit_price=limit_price,
                )
                
                plan.child_orders.append(order.order_id)
                
                # Wait for fill or timeout
                timeout = 60  # 1 minute timeout per slice
                start = datetime.now()
                
                while not order.is_filled and (datetime.now() - start).seconds < timeout:
                    # Check order status
                    updated_order = await self.order_manager.exchange.get_order(order.order_id)
                    if updated_order:
                        order = updated_order
                    
                    if order.is_filled:
                        break
                    
                    await asyncio.sleep(1)
                
                if order.is_filled:
                    plan.executed_quantity += order.filled_quantity
                    if plan.avg_price == 0:
                        plan.avg_price = order.filled_avg_price
                    else:
                        total_value = (plan.avg_price * (plan.executed_quantity - order.filled_quantity) +
                                      order.filled_avg_price * order.filled_quantity)
                        plan.avg_price = total_value / plan.executed_quantity
                else:
                    # Cancel unfilled order and continue
                    await self.order_manager.cancel_order(order.order_id)
                
                logger.info(f"Iceberg {plan_id}: {plan.completion_pct:.1f}% complete")
            
            logger.info(f"Iceberg execution complete: {plan_id}")
            
        except asyncio.CancelledError:
            logger.info(f"Iceberg execution cancelled: {plan_id}")
        except Exception as e:
            logger.error(f"Iceberg execution error: {e}")
        finally:
            self._active_plans.pop(plan_id, None)
            self._execution_tasks.pop(plan_id, None)
            self._completed_plans.append(plan)
    
    async def cancel_execution(self, plan_id: str) -> bool:
        """
        Cancel an active execution.
        
        Args:
            plan_id: ID of execution plan to cancel
            
        Returns:
            True if cancelled successfully
        """
        if plan_id not in self._active_plans:
            return False
        
        # Cancel the task
        if plan_id in self._execution_tasks:
            self._execution_tasks[plan_id].cancel()
        
        # Cancel pending child orders
        plan = self._active_plans[plan_id]
        for order_id in plan.child_orders:
            order = self.order_manager.get_order(order_id)
            if order and order.is_active:
                await self.order_manager.cancel_order(order_id)
        
        return True
    
    def get_active_executions(self) -> Dict[str, ExecutionPlan]:
        """Get all active execution plans."""
        return self._active_plans.copy()
    
    def get_execution_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an execution plan."""
        plan = self._active_plans.get(plan_id)
        if not plan:
            # Check completed plans
            for p in self._completed_plans:
                if f"{p.strategy.value}_{p.symbol}" in plan_id:
                    plan = p
                    break
        
        if not plan:
            return None
        
        return {
            "plan_id": plan_id,
            "strategy": plan.strategy.value,
            "symbol": plan.symbol,
            "side": plan.side.value,
            "total_quantity": plan.total_quantity,
            "executed_quantity": plan.executed_quantity,
            "remaining_quantity": plan.remaining_quantity,
            "avg_price": plan.avg_price,
            "completion_pct": plan.completion_pct,
            "is_complete": plan.is_complete,
            "num_child_orders": len(plan.child_orders),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        total_executions = len(self._completed_plans) + len(self._active_plans)
        completed = len(self._completed_plans)
        
        total_executed = sum(p.executed_quantity for p in self._completed_plans)
        
        return {
            "total_executions": total_executions,
            "completed_executions": completed,
            "active_executions": len(self._active_plans),
            "total_quantity_executed": total_executed,
        }

