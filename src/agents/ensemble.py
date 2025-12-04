"""
Ensemble Agent

Combines signals from multiple specialized agents using various
voting and weighting mechanisms.
"""

from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np
import pandas as pd

from .base import BaseAgent, AgentSignal, AgentAction
from .registry import AgentRegistry


class VotingMethod(Enum):
    """Methods for combining agent votes."""
    MAJORITY = "majority"           # Simple majority vote
    WEIGHTED = "weighted"           # Weight by agent confidence
    PERFORMANCE_WEIGHTED = "performance"  # Weight by historical performance
    UNANIMOUS = "unanimous"         # All agents must agree
    CONFIDENCE_THRESHOLD = "confidence"   # Only count high-confidence signals


class EnsembleAgent(BaseAgent):
    """
    Ensemble agent that combines signals from multiple agents.
    
    Features:
    - Multiple voting methods
    - Dynamic weight adjustment
    - Conflict resolution
    - Confidence aggregation
    """
    
    def __init__(
        self,
        name: str = "ensemble_agent",
        registry: Optional[AgentRegistry] = None,
        voting_method: VotingMethod = VotingMethod.WEIGHTED,
        confidence_threshold: float = 0.6,
        min_agreement: float = 0.5,
        **kwargs
    ):
        """
        Initialize the Ensemble Agent.
        
        Args:
            name: Agent identifier
            registry: Registry containing agents to ensemble
            voting_method: Method for combining votes
            confidence_threshold: Minimum confidence for a vote to count
            min_agreement: Minimum agreement ratio for action
        """
        super().__init__(
            name=name,
            description="Ensemble of multiple trading agents",
            lookback_period=50,  # Use max of sub-agents
            **kwargs
        )
        self.registry = registry or AgentRegistry()
        self.voting_method = voting_method
        self.confidence_threshold = confidence_threshold
        self.min_agreement = min_agreement
        self._last_signals: Dict[str, AgentSignal] = {}
    
    def add_agent(self, agent: BaseAgent, weight: float = 1.0) -> None:
        """Add an agent to the ensemble."""
        self.registry.register(agent, weight)
    
    def remove_agent(self, name: str) -> Optional[BaseAgent]:
        """Remove an agent from the ensemble."""
        return self.registry.unregister(name)
    
    def _aggregate_majority(
        self,
        signals: Dict[str, AgentSignal]
    ) -> AgentSignal:
        """Simple majority voting."""
        action_votes: Dict[AgentAction, int] = {}
        total_confidence = 0.0
        total_strength = 0.0
        
        for name, signal in signals.items():
            # Map strong buy/sell to buy/sell for voting
            vote_action = signal.action
            if vote_action == AgentAction.STRONG_BUY:
                vote_action = AgentAction.BUY
            elif vote_action == AgentAction.STRONG_SELL:
                vote_action = AgentAction.SELL
            
            action_votes[vote_action] = action_votes.get(vote_action, 0) + 1
            total_confidence += signal.confidence
            total_strength += signal.strength
        
        # Find winning action
        if not action_votes:
            return AgentSignal(AgentAction.HOLD, 0.5, 0.0)
        
        winning_action = max(action_votes.keys(), key=lambda k: action_votes[k])
        vote_ratio = action_votes[winning_action] / len(signals)
        
        avg_confidence = total_confidence / len(signals)
        avg_strength = total_strength / len(signals)
        
        # Require minimum agreement
        if vote_ratio < self.min_agreement:
            return AgentSignal(
                action=AgentAction.HOLD,
                confidence=0.5,
                strength=0.0,
                metadata={"reason": "Insufficient agreement", "vote_ratio": vote_ratio}
            )
        
        return AgentSignal(
            action=winning_action,
            confidence=avg_confidence * vote_ratio,
            strength=avg_strength,
            metadata={"vote_ratio": vote_ratio, "votes": dict(action_votes)}
        )
    
    def _aggregate_weighted(
        self,
        signals: Dict[str, AgentSignal]
    ) -> AgentSignal:
        """Weighted voting by confidence and registry weight."""
        action_scores: Dict[AgentAction, float] = {}
        total_weight = 0.0
        weighted_strength = 0.0
        
        for name, signal in signals.items():
            # Get registry weight
            registry_weight = self.registry.get_weight(name)
            
            # Combined weight = confidence * registry_weight
            combined_weight = signal.confidence * registry_weight
            
            # Map strong actions
            vote_action = signal.action
            if vote_action == AgentAction.STRONG_BUY:
                vote_action = AgentAction.BUY
                combined_weight *= 1.5  # Boost strong signals
            elif vote_action == AgentAction.STRONG_SELL:
                vote_action = AgentAction.SELL
                combined_weight *= 1.5
            
            action_scores[vote_action] = (
                action_scores.get(vote_action, 0.0) + combined_weight
            )
            total_weight += combined_weight
            weighted_strength += signal.strength * combined_weight
        
        if total_weight == 0:
            return AgentSignal(AgentAction.HOLD, 0.5, 0.0)
        
        # Find winning action
        winning_action = max(action_scores.keys(), key=lambda k: action_scores[k])
        winning_score = action_scores[winning_action]
        
        # Calculate confidence as ratio of winning score to total
        confidence = winning_score / total_weight
        avg_strength = weighted_strength / total_weight
        
        # Check minimum agreement
        if confidence < self.min_agreement:
            return AgentSignal(
                action=AgentAction.HOLD,
                confidence=0.5,
                strength=0.0,
                metadata={"reason": "Low weighted agreement", "score_ratio": confidence}
            )
        
        return AgentSignal(
            action=winning_action,
            confidence=confidence,
            strength=avg_strength,
            metadata={"action_scores": dict(action_scores), "total_weight": total_weight}
        )
    
    def _aggregate_performance_weighted(
        self,
        signals: Dict[str, AgentSignal]
    ) -> AgentSignal:
        """Weight by historical performance."""
        action_scores: Dict[AgentAction, float] = {}
        total_weight = 0.0
        weighted_strength = 0.0
        
        for name, signal in signals.items():
            agent = self.registry.get(name)
            if not agent:
                continue
            
            # Get performance metrics
            metrics = agent.get_performance_metrics()
            
            # Performance weight based on Sharpe and win rate
            perf_weight = max(0.1, (metrics["sharpe"] + 1) * metrics["win_rate"])
            combined_weight = signal.confidence * perf_weight
            
            vote_action = signal.action
            if vote_action in [AgentAction.STRONG_BUY, AgentAction.STRONG_SELL]:
                vote_action = (
                    AgentAction.BUY if vote_action == AgentAction.STRONG_BUY
                    else AgentAction.SELL
                )
                combined_weight *= 1.3
            
            action_scores[vote_action] = (
                action_scores.get(vote_action, 0.0) + combined_weight
            )
            total_weight += combined_weight
            weighted_strength += signal.strength * combined_weight
        
        if total_weight == 0:
            return AgentSignal(AgentAction.HOLD, 0.5, 0.0)
        
        winning_action = max(action_scores.keys(), key=lambda k: action_scores[k])
        confidence = action_scores[winning_action] / total_weight
        avg_strength = weighted_strength / total_weight
        
        return AgentSignal(
            action=winning_action,
            confidence=confidence,
            strength=avg_strength,
            metadata={"action_scores": dict(action_scores), "method": "performance_weighted"}
        )
    
    def _aggregate_confidence_threshold(
        self,
        signals: Dict[str, AgentSignal]
    ) -> AgentSignal:
        """Only count signals above confidence threshold."""
        filtered_signals = {
            name: sig for name, sig in signals.items()
            if sig.confidence >= self.confidence_threshold
        }
        
        if not filtered_signals:
            return AgentSignal(
                action=AgentAction.HOLD,
                confidence=0.5,
                strength=0.0,
                metadata={"reason": "No signals above threshold"}
            )
        
        # Use weighted voting on filtered signals
        return self._aggregate_weighted(filtered_signals)
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        position: int = 0,
        **kwargs
    ) -> AgentSignal:
        """
        Generate ensemble signal by combining all agent signals.
        
        Args:
            data: OHLCV DataFrame
            position: Current position
            
        Returns:
            Combined AgentSignal
        """
        if len(self.registry) == 0:
            return AgentSignal(
                action=AgentAction.HOLD,
                confidence=0.0,
                strength=0.0,
                metadata={"reason": "No agents registered"}
            )
        
        # Get signals from all agents
        signals = self.registry.get_signals(data, position)
        self._last_signals = signals
        
        if not signals:
            return AgentSignal(
                action=AgentAction.HOLD,
                confidence=0.0,
                strength=0.0,
                metadata={"reason": "No signals generated"}
            )
        
        # Aggregate based on voting method
        if self.voting_method == VotingMethod.MAJORITY:
            result = self._aggregate_majority(signals)
        elif self.voting_method == VotingMethod.WEIGHTED:
            result = self._aggregate_weighted(signals)
        elif self.voting_method == VotingMethod.PERFORMANCE_WEIGHTED:
            result = self._aggregate_performance_weighted(signals)
        elif self.voting_method == VotingMethod.CONFIDENCE_THRESHOLD:
            result = self._aggregate_confidence_threshold(signals)
        else:
            result = self._aggregate_weighted(signals)
        
        # Add individual signals to metadata
        result.metadata["individual_signals"] = {
            name: sig.to_dict() for name, sig in signals.items()
        }
        result.metadata["num_agents"] = len(signals)
        result.metadata["voting_method"] = self.voting_method.value
        
        return result
    
    def get_last_signals(self) -> Dict[str, AgentSignal]:
        """Get the last signals from all agents."""
        return self._last_signals.copy()
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get ensemble strategy parameters."""
        return {
            "voting_method": self.voting_method.value,
            "confidence_threshold": self.confidence_threshold,
            "min_agreement": self.min_agreement,
            "num_agents": len(self.registry),
            "agent_names": self.registry.list_agents(),
        }
    
    def reset(self) -> None:
        """Reset ensemble and all sub-agents."""
        super().reset()
        self.registry.reset_all()
        self._last_signals = {}

