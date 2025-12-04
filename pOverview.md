Project overview: HFT market simulator with reinforcement learning
Core concept
Build a market simulator where RL agents learn to trade by interacting with a simulated market. The agents observe market signals, choose actions (buy/sell/hold), and receive rewards based on risk-adjusted performance. The goal is to train agents that can make profitable, risk-aware trading decisions in a high-frequency context.
Why it matters
CS: RL algorithms, neural networks, distributed systems
Economics: Risk-adjusted returns, market microstructure, game theory
Engineering: MLOps, low-latency serving, scalable training
Components
RL agent
Algorithm: PPO or DQN
State: Order book depth, RSI, MACD, and other market signals
Actions: Buy, Sell, Hold (or continuous order sizing)
Reward: Risk-adjusted returns (e.g., Sharpe ratio, drawdown penalties)
Market simulator
Simulates order book dynamics, price movements, and order execution
Provides realistic market conditions for training
Training infrastructure
Distributed training with Ray RLlib (50+ parallel simulations)
Experiment tracking with MLflow or W&B
Logs training metrics, hyperparameters, and model checkpoints
Production serving
Microservice that accepts market state (JSON) and returns trading actions
Latency target: <50ms
Learning objective
The agent learns a policy that maximizes risk-adjusted returns, not just raw profit. The reward function penalizes volatility and drawdowns, encouraging consistent, risk-aware trading.
Success metrics
Training: Decreasing loss, improving Sharpe ratio over episodes
Validation: Strong out-of-sample performance
Production: <50ms inference latency, consistent risk-adjusted returns
In short: train RL agents to trade in a simulated HFT environment, optimize for risk-adjusted returns, and deploy them as a low-latency service.
