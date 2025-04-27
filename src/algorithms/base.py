from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import gymnasium as gym

class BaseAlgorithm(ABC):
    """Base class for all reinforcement learning algorithms."""
    
    def __init__(self, env: gym.Env, config: Dict[str, Any]):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def train(self, num_episodes: int) -> Dict[str, Any]:
        """Train the algorithm for a specified number of episodes."""
        pass
    
    @abstractmethod
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current policy."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model and algorithm state."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model and algorithm state."""
        pass 