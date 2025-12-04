"""
Agent Registry

Manages multiple trading agents, tracks their performance, and provides
utilities for agent selection and comparison.
"""

from typing import Dict, List, Optional, Any, Type
import pandas as pd
import numpy as np
from datetime import datetime

from .base import BaseAgent, AgentSignal


class AgentRegistry:
    """
    Registry for managing multiple trading agents.
    
    Features:
    - Register/unregister agents
    - Track agent performance
    - Dynamic agent weighting
    - Agent comparison and ranking
    """
    
    def __init__(self):
        """Initialize the agent registry."""
        self._agents: Dict[str, BaseAgent] = {}
        self._weights: Dict[str, float] = {}
        self._performance_log: Dict[str, List[Dict[str, Any]]] = {}
        self._created_at = datetime.now()
    
    def register(
        self,
        agent: BaseAgent,
        initial_weight: float = 1.0
    ) -> None:
        """
        Register an agent with the registry.
        
        Args:
            agent: The agent to register
            initial_weight: Initial weight for ensemble voting
        """
        if agent.name in self._agents:
            raise ValueError(f"Agent '{agent.name}' already registered")
        
        self._agents[agent.name] = agent
        self._weights[agent.name] = initial_weight
        self._performance_log[agent.name] = []
    
    def unregister(self, name: str) -> Optional[BaseAgent]:
        """
        Remove an agent from the registry.
        
        Args:
            name: Name of the agent to remove
            
        Returns:
            The removed agent, or None if not found
        """
        if name in self._agents:
            agent = self._agents.pop(name)
            self._weights.pop(name, None)
            self._performance_log.pop(name, None)
            return agent
        return None
    
    def get(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self._agents.get(name)
    
    def list_agents(self) -> List[str]:
        """Get list of registered agent names."""
        return list(self._agents.keys())
    
    def get_all_agents(self) -> Dict[str, BaseAgent]:
        """Get all registered agents."""
        return self._agents.copy()
    
    def get_signals(
        self,
        data: pd.DataFrame,
        position: int = 0,
        agent_names: Optional[List[str]] = None
    ) -> Dict[str, AgentSignal]:
        """
        Get signals from multiple agents.
        
        Args:
            data: Market data DataFrame
            position: Current position
            agent_names: Specific agents to query (None = all)
            
        Returns:
            Dictionary mapping agent names to their signals
        """
        signals = {}
        agents_to_query = agent_names or list(self._agents.keys())
        
        for name in agents_to_query:
            if name in self._agents:
                try:
                    signals[name] = self._agents[name].generate_signal(
                        data, position
                    )
                except Exception as e:
                    # Log error but continue with other agents
                    print(f"Error getting signal from {name}: {e}")
        
        return signals
    
    def update_weights(
        self,
        performance_window: int = 50,
        min_weight: float = 0.1,
        max_weight: float = 3.0
    ) -> None:
        """
        Update agent weights based on recent performance.
        
        Args:
            performance_window: Number of recent trades to consider
            min_weight: Minimum weight
            max_weight: Maximum weight
        """
        for name, agent in self._agents.items():
            metrics = agent.get_performance_metrics()
            
            if metrics["num_trades"] < 5:
                # Not enough data, keep current weight
                continue
            
            # Calculate weight based on Sharpe ratio and win rate
            sharpe = metrics["sharpe"]
            win_rate = metrics["win_rate"]
            
            # Combine metrics (higher is better)
            score = (sharpe + 1) * (win_rate + 0.5)
            
            # Normalize to weight
            new_weight = np.clip(score, min_weight, max_weight)
            self._weights[name] = new_weight
    
    def get_weight(self, name: str) -> float:
        """Get current weight for an agent."""
        return self._weights.get(name, 1.0)
    
    def set_weight(self, name: str, weight: float) -> None:
        """Set weight for an agent."""
        if name in self._agents:
            self._weights[name] = max(0.0, weight)
    
    def log_performance(
        self,
        agent_name: str,
        signal: AgentSignal,
        actual_return: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Log performance for an agent.
        
        Args:
            agent_name: Name of the agent
            signal: The signal that was generated
            actual_return: The actual return achieved
            timestamp: When this occurred
        """
        if agent_name not in self._performance_log:
            return
        
        self._performance_log[agent_name].append({
            "timestamp": timestamp or datetime.now(),
            "action": signal.action.name,
            "confidence": signal.confidence,
            "strength": signal.strength,
            "actual_return": actual_return,
        })
        
        # Also update agent's performance
        self._agents[agent_name].update_performance(actual_return)
    
    def get_rankings(self) -> List[Dict[str, Any]]:
        """
        Get agents ranked by performance.
        
        Returns:
            List of agents sorted by total PnL (descending)
        """
        rankings = []
        
        for name, agent in self._agents.items():
            metrics = agent.get_performance_metrics()
            rankings.append({
                "name": name,
                "weight": self._weights.get(name, 1.0),
                **metrics
            })
        
        # Sort by Sharpe ratio (or total_pnl as fallback)
        rankings.sort(
            key=lambda x: (x["sharpe"], x["total_pnl"]),
            reverse=True
        )
        
        return rankings
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all registered agents."""
        return {
            "num_agents": len(self._agents),
            "agent_names": list(self._agents.keys()),
            "weights": self._weights.copy(),
            "rankings": self.get_rankings(),
            "created_at": self._created_at.isoformat(),
        }
    
    def reset_all(self) -> None:
        """Reset all agents."""
        for agent in self._agents.values():
            agent.reset()
        self._performance_log = {name: [] for name in self._agents}
    
    def __len__(self) -> int:
        return len(self._agents)
    
    def __contains__(self, name: str) -> bool:
        return name in self._agents
    
    def __repr__(self) -> str:
        return f"AgentRegistry(agents={list(self._agents.keys())})"

