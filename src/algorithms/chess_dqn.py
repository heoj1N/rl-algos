import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, Any
from .base import BaseAlgorithm
from ..networks.chess_network import ChessNetwork

class ChessDQN(BaseAlgorithm):
    """Deep Q-Network implementation for chess."""
    
    def __init__(self, env: Any, config: Dict[str, Any]):
        super().__init__(env, config)
        
        # Network parameters
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.n
        
        # Create networks
        self.policy_net = ChessNetwork(self.action_dim).to(self.device)
        self.target_net = ChessNetwork(self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.get('learning_rate', 0.001))
        
        # Replay buffer
        self.memory = deque(maxlen=config.get('memory_size', 100000))
        
        # Training parameters
        self.batch_size = config.get('batch_size', 128)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.target_update = config.get('target_update', 1000)
        self.steps_done = 0
    
    def _get_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        self.steps_done += 1
        
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            return q_values.argmax().item()
    
    def _update_network(self) -> float:
        """Update the Q-network using experience replay."""
        if len(self.memory) < self.batch_size:
            return 0.0
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q(s_t, a)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute Q(s_{t+1}, a)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train(self, num_episodes: int) -> Dict[str, Any]:
        """Train the DQN agent."""
        episode_rewards = []
        losses = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self._get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward
                
                loss = self._update_network()
                if loss > 0:
                    losses.append(loss)
            
            episode_rewards.append(episode_reward)
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Epsilon: {self.epsilon:.2f}, "
                  f"Avg Loss: {avg_loss:.4f}")
        
        return {
            "episode_rewards": episode_rewards,
            "losses": losses
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current policy."""
        rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    action = self.policy_net(state_tensor).argmax().item()
                
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            rewards.append(episode_reward)
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards)
        }
    
    def save(self, path: str) -> None:
        """Save the model and algorithm state."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
    
    def load(self, path: str) -> None:
        """Load the model and algorithm state."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done'] 