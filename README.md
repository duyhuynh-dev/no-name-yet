# HFT Market Simulator with Reinforcement Learning

A High-Frequency Trading Market Simulator using Reinforcement Learning (PPO) with risk-adjusted reward functions, MLOps pipeline, and production deployment.

## Project Overview

This project implements a complete HFT trading system that:
- Uses **Proximal Policy Optimization (PPO)** to train trading agents
- Implements **risk-adjusted reward functions** (Sharpe ratio, volatility penalties)
- Provides a **custom Gymnasium environment** for market simulation
- Includes **MLOps pipeline** with MLflow for experiment tracking
- Deploys trained models via **FastAPI** with <50ms latency
- Features a **React/Next.js dashboard** for visualization

## Project Structure

```
.
├── src/              # Source code
├── data/             # Data storage (raw and processed)
│   ├── raw/         # Raw market data
│   └── processed/   # Processed features
├── models/           # Trained model checkpoints
├── notebooks/        # Jupyter notebooks for exploration
├── tests/            # Unit and integration tests
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Setup

### Prerequisites

- Python 3.8-3.11
- pip
- (Optional) TA-Lib system library for technical indicators

### Installation

1. **Clone the repository** (if applicable)

2. **Run the setup script:**
   ```bash
   ./setup_env.sh
   ```

   Or manually:
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate virtual environment
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Install TA-Lib** (if needed):
   
   **macOS:**
   ```bash
   brew install ta-lib
   ```
   
   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install ta-lib
   ```
   
   **Windows:**
   Download from [Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

## Usage

*Coming soon - will be updated as we implement features*

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
```

## Project Status

See [TODO.md](TODO.md) for detailed progress tracking.

## License

*To be determined*
