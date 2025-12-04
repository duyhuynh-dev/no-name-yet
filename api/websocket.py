"""
WebSocket Handler for Real-Time Updates

Provides real-time streaming of:
- Portfolio updates
- Position changes
- Trade executions
- Market data
- Agent signals
- System alerts
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types"""
    PORTFOLIO_UPDATE = "portfolio_update"
    POSITION_UPDATE = "position_update"
    TRADE_EXECUTED = "trade_executed"
    MARKET_DATA = "market_data"
    AGENT_SIGNAL = "agent_signal"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


@dataclass
class WSMessage:
    """WebSocket message structure"""
    type: MessageType
    data: Any
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "timestamp": self.timestamp,
            "data": self.data
        })


class ConnectionManager:
    """
    Manages WebSocket connections and subscriptions
    
    Features:
    - Connection lifecycle management
    - Channel-based subscriptions
    - Broadcast and targeted messaging
    - Heartbeat monitoring
    """
    
    def __init__(self):
        # Active connections
        self.active_connections: Set[WebSocket] = set()
        
        # Subscriptions: channel -> set of websockets
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        
        # Connection metadata
        self.connection_info: Dict[WebSocket, dict] = {}
        
        # Heartbeat tracking
        self.last_heartbeat: Dict[WebSocket, datetime] = {}
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_info[websocket] = {
            "connected_at": datetime.now().isoformat(),
            "subscriptions": set(),
        }
        self.last_heartbeat[websocket] = datetime.now()
        
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        
        # Remove from all subscriptions
        for channel, subscribers in self.subscriptions.items():
            subscribers.discard(websocket)
        
        # Cleanup metadata
        self.connection_info.pop(websocket, None)
        self.last_heartbeat.pop(websocket, None)
        
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    def subscribe(self, websocket: WebSocket, channel: str) -> None:
        """Subscribe a connection to a channel"""
        if channel not in self.subscriptions:
            self.subscriptions[channel] = set()
        
        self.subscriptions[channel].add(websocket)
        
        if websocket in self.connection_info:
            self.connection_info[websocket]["subscriptions"].add(channel)
        
        logger.debug(f"Subscribed to channel: {channel}")
    
    def unsubscribe(self, websocket: WebSocket, channel: str) -> None:
        """Unsubscribe a connection from a channel"""
        if channel in self.subscriptions:
            self.subscriptions[channel].discard(websocket)
        
        if websocket in self.connection_info:
            self.connection_info[websocket]["subscriptions"].discard(channel)
    
    async def send_to_connection(
        self, 
        websocket: WebSocket, 
        message: WSMessage
    ) -> bool:
        """Send a message to a specific connection"""
        try:
            await websocket.send_text(message.to_json())
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def broadcast(self, message: WSMessage) -> int:
        """Broadcast a message to all connected clients"""
        sent_count = 0
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message.to_json())
                sent_count += 1
            except Exception:
                disconnected.append(connection)
        
        # Cleanup disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
        
        return sent_count
    
    async def broadcast_to_channel(
        self, 
        channel: str, 
        message: WSMessage
    ) -> int:
        """Broadcast a message to all subscribers of a channel"""
        if channel not in self.subscriptions:
            return 0
        
        sent_count = 0
        disconnected = []
        
        for connection in self.subscriptions[channel]:
            try:
                await connection.send_text(message.to_json())
                sent_count += 1
            except Exception:
                disconnected.append(connection)
        
        # Cleanup disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
        
        return sent_count
    
    def update_heartbeat(self, websocket: WebSocket) -> None:
        """Update the last heartbeat time for a connection"""
        self.last_heartbeat[websocket] = datetime.now()
    
    def get_stats(self) -> dict:
        """Get connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "channels": {
                channel: len(subscribers) 
                for channel, subscribers in self.subscriptions.items()
            },
        }


# Global connection manager
manager = ConnectionManager()


async def handle_websocket(websocket: WebSocket):
    """
    Main WebSocket handler
    
    Handles:
    - Connection lifecycle
    - Message routing
    - Subscription management
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                msg_type = message.get("type", "")
                
                if msg_type == "ping":
                    # Heartbeat
                    manager.update_heartbeat(websocket)
                    await websocket.send_text(
                        WSMessage(
                            type=MessageType.HEARTBEAT,
                            data={"status": "ok"}
                        ).to_json()
                    )
                
                elif msg_type == "subscribe":
                    channel = message.get("channel", "")
                    if channel:
                        manager.subscribe(websocket, channel)
                        await websocket.send_text(json.dumps({
                            "type": "subscribed",
                            "channel": channel,
                            "timestamp": datetime.now().isoformat()
                        }))
                
                elif msg_type == "unsubscribe":
                    channel = message.get("channel", "")
                    if channel:
                        manager.unsubscribe(websocket, channel)
                        await websocket.send_text(json.dumps({
                            "type": "unsubscribed",
                            "channel": channel,
                            "timestamp": datetime.now().isoformat()
                        }))
                
                else:
                    # Unknown message type
                    await websocket.send_text(
                        WSMessage(
                            type=MessageType.ERROR,
                            data={"error": f"Unknown message type: {msg_type}"}
                        ).to_json()
                    )
            
            except json.JSONDecodeError:
                await websocket.send_text(
                    WSMessage(
                        type=MessageType.ERROR,
                        data={"error": "Invalid JSON"}
                    ).to_json()
                )
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# Helper functions for publishing updates

async def publish_portfolio_update(data: dict) -> int:
    """Publish a portfolio update to all subscribers"""
    message = WSMessage(
        type=MessageType.PORTFOLIO_UPDATE,
        data=data
    )
    return await manager.broadcast_to_channel("portfolio", message)


async def publish_position_update(data: dict) -> int:
    """Publish a position update to all subscribers"""
    message = WSMessage(
        type=MessageType.POSITION_UPDATE,
        data=data
    )
    return await manager.broadcast_to_channel("positions", message)


async def publish_trade(data: dict) -> int:
    """Publish a trade execution to all subscribers"""
    message = WSMessage(
        type=MessageType.TRADE_EXECUTED,
        data=data
    )
    return await manager.broadcast_to_channel("trades", message)


async def publish_market_data(symbol: str, data: dict) -> int:
    """Publish market data for a specific symbol"""
    message = WSMessage(
        type=MessageType.MARKET_DATA,
        data=data
    )
    return await manager.broadcast_to_channel(f"market:{symbol}", message)


async def publish_agent_signal(data: dict) -> int:
    """Publish an agent signal to all subscribers"""
    message = WSMessage(
        type=MessageType.AGENT_SIGNAL,
        data=data
    )
    return await manager.broadcast_to_channel("agents", message)


async def publish_alert(data: dict) -> int:
    """Publish an alert to all subscribers"""
    message = WSMessage(
        type=MessageType.ALERT,
        data=data
    )
    # Alerts go to everyone
    return await manager.broadcast(message)

