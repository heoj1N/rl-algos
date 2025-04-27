import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List

class TetrisDQN:
    """Deep Q-Network agent for Tetris."""
    
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        self.policy_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config['learning_rate'])
        self.memory = deque(maxlen=config['memory_size'])
        self.epsilon = config['epsilon_start']
        self.steps_done = 0
    
    def _build_network(self) -> nn.Module:
        """Build the neural network architecture."""
        return nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * self.env.height * self.env.width, self.config['hidden_size']),
            nn.ReLU(),
            nn.Linear(self.config['hidden_size'], self.env.action_space.n)
        )
    
    def _select_action(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def _optimize_model(self) -> float:
        """Perform one step of optimization."""
        if len(self.memory) < self.config['batch_size']:
            return 0.0
        
        batch = random.sample(self.memory, self.config['batch_size'])
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        state_batch = torch.FloatTensor(state_batch).unsqueeze(1).to(self.device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute V(s_{t+1})
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            expected_q_values = reward_batch + (1 - done_batch) * self.config['gamma'] * next_q_values
        
        # Compute loss and optimize
        loss = nn.MSELoss()(q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes: int) -> Dict[str, List[float]]:
        """Train the agent."""
        episode_rewards = []
        losses = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_losses = []
            steps = 0
            
            while True:
                action = self._select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                
                # Store transition in memory
                self.memory.append((state, action, reward, next_state, done))
                
                # Optimize model
                loss = self._optimize_model()
                if loss > 0:  # Only track non-zero losses
                    episode_losses.append(loss)
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                # Update target network
                if steps % self.config['target_update'] == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                if done:
                    break
            
            # Update epsilon
            self.epsilon = max(self.config['epsilon_end'], 
                             self.epsilon * self.config['epsilon_decay'])
            
            # Record episode results
            episode_rewards.append(episode_reward)
            avg_loss = np.mean(episode_losses) if episode_losses else 0
            losses.append(avg_loss)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Reward: {episode_reward:.2f}, "
                      f"Loss: {avg_loss:.4f}, "
                      f"Epsilon: {self.epsilon:.2f}")
        
        return {
            'episode_rewards': episode_rewards,
            'losses': losses
        }
    
    def evaluate(self, num_episodes: int) -> Dict[str, float]:
        """Evaluate the trained agent."""
        rewards = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            while True:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
                    q_values = self.policy_net(state_tensor)
                    action = q_values.argmax().item()
                
                state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                
                if done:
                    break
            
            rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards)
        }
    
    def save(self, path: str) -> None:
        """Save the trained model."""
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load a trained model."""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict()) 