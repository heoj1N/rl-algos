"""Default configuration for DQN algorithm."""

DQN_CONFIG = {
    'learning_rate': 0.001,
    'memory_size': 10000,
    'batch_size': 64,
    'gamma': 0.99,  # Discount factor
    'epsilon_start': 1.0,
    'epsilon_min': 0.01,
    'epsilon_decay': 0.995,
    'target_update': 10,  # Update target network every N episodes
} 