# RL Algorithms in PyTorch

A modular implementation of reinforcement learning algorithms in PyTorch, starting with Deep Q-Networks (DQN).

## Project Structure
```
rl-algorithms-pytorch/
├── src/
│   ├── algorithms/     # RL algorithm implementations
│   ├── networks/       # Neural network architectures
│   ├── environments/   # Environment wrappers and utilities
│   ├── utils/         # Helper functions and utilities
│   └── config/        # Configuration files
├── tests/             # Unit tests
└── notebooks/         # Example notebooks
```

## Setup
1. Create a virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Usage
See the notebooks directory for example usage of different algorithms.

## Contributing
New algorithms should follow the base algorithm interface in `src/algorithms/base.py`. 