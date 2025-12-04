"""
Advanced Reinforcement Learning Algorithms

Implements SAC (Soft Actor-Critic) and TD3 (Twin Delayed DDPG)
for continuous action trading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class Transition:
    """Experience tuple for replay buffer."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity: int = 1_000_000):
        """Initialize replay buffer."""
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Add transition to buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample batch of transitions."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class MLPNetwork(nn.Module):
    """Multi-layer perceptron network."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for SAC.
    
    Outputs mean and log_std for action distribution.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean and log_std."""
        h = self.shared(state)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action with reparameterization trick.
        
        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Reparameterization
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        
        # Squash through tanh
        action = torch.tanh(x_t)
        
        # Log probability with tanh squashing correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for inference."""
        mean, log_std = self.forward(state)
        
        if deterministic:
            return torch.tanh(mean)
        
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        action = torch.tanh(normal.sample())
        
        return action


class TwinQNetwork(nn.Module):
    """
    Twin Q-networks for SAC/TD3.
    
    Uses two Q-networks to reduce overestimation bias.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__()
        
        self.q1 = MLPNetwork(state_dim + action_dim, 1, hidden_dims)
        self.q2 = MLPNetwork(state_dim + action_dim, 1, hidden_dims)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get Q-values from both networks."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
    
    def q1_forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get Q-value from first network only."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class SACAgent:
    """
    Soft Actor-Critic Agent.
    
    SAC is an off-policy algorithm that maximizes both
    expected return and entropy for exploration.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        auto_entropy: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize SAC Agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Entropy coefficient
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            lr_alpha: Alpha learning rate
            buffer_size: Replay buffer size
            batch_size: Training batch size
            auto_entropy: Whether to auto-tune alpha
            device: Compute device
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim
        
        # Networks
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Copy weights to target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Entropy
        self.auto_entropy = auto_entropy
        if auto_entropy:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
        else:
            self.alpha = alpha
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.training_steps = 0
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Select action for given state."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor.get_action(state, deterministic)
        
        return action.cpu().numpy()[0]
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """
        Update networks using a batch from replay buffer.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Sample batch
        batch = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([t.action for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t.reward for t in batch])).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t.done for t in batch])).unsqueeze(-1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        alpha_loss = 0.0
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.training_steps += 1
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss.item() if self.auto_entropy else 0.0,
        }
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha if self.auto_entropy else None,
            "training_steps": self.training_steps,
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if checkpoint["log_alpha"] is not None:
            self.log_alpha = checkpoint["log_alpha"]
            self.alpha = self.log_alpha.exp().item()
        self.training_steps = checkpoint["training_steps"]


class DeterministicPolicy(nn.Module):
    """Deterministic policy network for TD3."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        max_action: float = 1.0,
    ):
        super().__init__()
        
        self.max_action = max_action
        self.network = MLPNetwork(state_dim, action_dim, hidden_dims, activation="relu")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * torch.tanh(self.network(state))


class TD3Agent:
    """
    Twin Delayed DDPG Agent.
    
    TD3 improves on DDPG with:
    - Twin Q-networks
    - Delayed policy updates
    - Target policy smoothing
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float = 1.0,
        hidden_dims: List[int] = [256, 256],
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        exploration_noise: float = 0.1,
        device: str = "cpu",
    ):
        """
        Initialize TD3 Agent.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            max_action: Maximum action value
            hidden_dims: Hidden layer dimensions
            gamma: Discount factor
            tau: Soft update coefficient
            policy_noise: Noise added to target policy
            noise_clip: Range to clip noise
            policy_delay: Frequency of delayed policy updates
            lr_actor: Actor learning rate
            lr_critic: Critic learning rate
            buffer_size: Replay buffer size
            batch_size: Training batch size
            exploration_noise: Exploration noise std
            device: Compute device
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.action_dim = action_dim
        
        # Networks
        self.actor = DeterministicPolicy(state_dim, action_dim, hidden_dims, max_action).to(self.device)
        self.actor_target = DeterministicPolicy(state_dim, action_dim, hidden_dims, max_action).to(self.device)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Copy weights to targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.training_steps = 0
    
    def select_action(
        self,
        state: np.ndarray,
        add_noise: bool = True,
    ) -> np.ndarray:
        """Select action for given state."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """
        Update networks using a batch from replay buffer.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < self.batch_size:
            return {}
        
        # Sample batch
        batch = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([t.action for t in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([t.reward for t in batch])).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t.done for t in batch])).unsqueeze(-1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Twin Q-values
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.training_steps += 1
        actor_loss = 0.0
        
        # Delayed policy updates
        if self.training_steps % self.policy_delay == 0:
            # Update actor
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update targets
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            actor_loss = actor_loss.item()
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss,
        }
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "training_steps": self.training_steps,
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.training_steps = checkpoint["training_steps"]

